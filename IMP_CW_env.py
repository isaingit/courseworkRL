import gymnasium as gym
import numpy as np
from scipy.stats import poisson
import copy

class MESCEnv():   
    def __init__(self, structure, *args, **kwargs):
        '''
        Args:
            structure : List of integers whose length is the number of stages and each element is the number of participants in that stage.
        '''
        self.name = "InventoryManagement"
        # Default values for the initial conditions (overriden if kwargs are passed)
        self.n_periods = 52*7 # 52 weeks * 7 days
        self.seed = 0
        self.demand_dist_fcn = poisson
        self.demand_dist_param = [{'mu': 3}, # Mo-Thu
                                  {'mu': 10}, # Fri
                                  {'mu': 15}] # Sat-Sun
        
        # Buffers to store data
        self.demand_dataset = None # Attribute to store the demands of the episode if externally provided (e.g., during evaluation)

        # Assigned the sub-classed frozen distribution to the demand distribution function
        if self.demand_dist_fcn is poisson:
            self.demand_dist = frozen_poisson(random_state=self.seed)

        # Update initial values if kwargs are passed
        for key, value in kwargs.items():
            setattr(self, key, value)  

        # Definition of the supply chain
        self.structure = structure # Number of retailers and DCs in the system
        # Instantiate retailers        
        self.retailers = []
        for i in range(sum(structure[0])):
            self.retailers.append(Retailer())

        # Instantiate DCs  
        self.DCs = []
        for i in range(len(structure[1])):
            self.DCs.append(DC(self.retailers[(i*structure[0][i]):(i*structure[0][i]+structure[0][i])]))

        self.reset()

        # Define action space: reorder quantities of each stage m at each period t
        self.n_actions = len(self.retailers) + len(self.DCs) # Number of actions = number of retailers + number of DCs
        self.action_space_UB = np.array([retailer.order_quantity_limit for retailer in self.retailers] + [DC.order_quantity_limit for DC in self.DCs])
        self.action_space = gym.spaces.Box(low=np.zeros(self.n_actions, dtype=np.int16), 
                                           high=self.action_space_UB, 
                                           dtype=np.int16)

        # Define observation space: inventory position of each entity + day of the week
        self.n_states  = len(self.retailers) + len(self.DCs) + 1 # Number of states = number of retailers + number of DCs + day of the week 
        self.observation_space_UB = np.array([DC.capacity + DC.order_quantity_limit for DC in self.DCs] + [retailer.capacity + retailer.order_quantity_limit for retailer in self.retailers] + [6])
        self.observation_space = gym.spaces.Box(low=np.zeros(self.n_states, dtype=np.int16), 
                                                high=self.observation_space_UB, 
                                                dtype=np.int16)
    
    def _RESET(self):
        self.current_period = 1
        self.dow = 0
        
        # Definition of the self.demands_episode attribute, which stores the demand information that will be actually used during self.step()
        if self.demand_dataset is None:
            self.demands_episode , self.prob_per_scenario = self.sample_demands_episode()
        else:
            self.demands_episode = copy.deepcopy(self.demand_dataset)

        for i, retailer in enumerate(self.retailers):
            retailer.reset(self.demands_episode[:,i])
        for DC in self.DCs:
            DC.reset()
        
        self._update_state()
    def reset(self):
        return self._RESET()

    def _STEP(self, action):
        #> 0. Ensure action is of the correct dtype
        action = action.astype(np.int16) 

        for i in range(len(self.DCs)):
            #> 1.1 DCs place orders to supplier
            # TODO: revise it's being done correctly. Depending on the meaning, it should take one action or the other.
            self.DCs[i].place_order(action[i+len(self.retailers)], self.current_period)

            #> 1.2 DCs receive orders from supplier
            self.DCs[i].receive_order(self.current_period)

            #> 1.3 DCs satisfy demand from retailers
            self.DCs[i].satisfy_demand(self.DCs[i].retailers, action[:len(self.retailers)], self.current_period)
        
        for i in range(len(self.retailers)):
            #> 2.1 Retailers receive orders from DCs
            self.retailers[i].receive_order(self.current_period)

            #> 2.2 Retailers satisfy demand from customers
            self.retailers[i].satisfy_demand(self.dow)
        
        #> 3. Compute reward
        revenue = np.sum(retailer.SD * retailer.unit_price for retailer in self.retailers) # Revenue from sale
        holding_cost = np.sum(DC.I * DC.holding_cost for DC in self.DCs) + np.sum(retailer.I * retailer.holding_cost for retailer in self.retailers) 
        fixed_order_cost = np.sum(DC.fix_order_cost*DC.n_orders for DC in self.DCs) + np.sum(retailer.fix_order_cost*retailer.n_orders for retailer in self.retailers)
        var_order_cost = np.sum(DC.var_order_cost * action[i+len(self.retailers)] for i, DC in enumerate(self.DCs))
        penalty_unsatisfied_demand = np.sum(retailer.UD * retailer.lost_sales_cost for retailer in self.retailers) 
        penalty_capacity_violation = np.sum(DC.I_surplus * DC.capacity_violation_cost for DC in self.DCs) + np.sum(retailer.I_surplus * retailer.capacity_violation_cost for retailer in self.retailers) # Penalty for exceeding storage capacity
        reward = revenue - holding_cost - fixed_order_cost - var_order_cost - penalty_unsatisfied_demand - penalty_capacity_violation
        # print(f"Period {self.current_period} - Reward: {reward}, Revenue: {revenue}, Holding Cost: {holding_cost}, Fixed Order Cost: {fixed_order_cost}, Variable Order Cost: {var_order_cost}, Penalty Unsatisfied Demand: {penalty_unsatisfied_demand}, Penalty Capacity Violation: {penalty_capacity_violation}")
        #> 4. Update period
        self.current_period += 1

        #> 5. Update state
        self._update_state()

        #> 6. Check if episode is finished
        if self.current_period > self.n_periods:
            episode_ended = True
        else:
            episode_ended = False

        return self.state , reward, episode_ended , {}
    def step(self, action):
        return self._STEP(action)
    
    def _update_state(self):
        self.dow = (self.current_period-1) % 7 # Day of the week (0-6)
        self.state = np.array([retailer.I for retailer in self.retailers]+ [DC.inv_pos for DC in self.DCs] + [self.dow])

    def sample_demands_episode(self):
        '''
        Sample demand for each retailer in the current period. The demand distribution function is defined in the __init__ method.
        '''
        demands_episode = []
        prob_per_scenario = np.zeros(self.n_periods, dtype=np.float32)
        for i in range(self.n_periods):
            if self.dow < 4:
                demand_dist_param = self.demand_dist_param[0]
            elif self.dow == 4:
                demand_dist_param = self.demand_dist_param[1]
            else:
                demand_dist_param = self.demand_dist_param[2]

            scenario = self.demand_dist.rvs(**demand_dist_param, size=len(self.retailers))
            prob_per_scenario[i] = np.prod([self.demand_dist.pmf(scenario[j], **demand_dist_param) for j in range(len(scenario))])
            demands_episode.append(scenario)

        return np.array(demands_episode), prob_per_scenario 

class DC():
    def __init__(self, retailers , *args, **kwargs):
        # Defaul values for the initial conditions (overriden if kwargs are passed)
        self.I0 = 100 # Initial on-hand inventory
        self.lead_time = 5
        self.capacity = 150 # storage capacity of the DC
        self.order_quantity_limit = 200 # productive capacity of DC's supplier. TBD: check how now it's unused
        self.holding_cost = 1
        self.fix_order_cost = 75
        self.var_order_cost = 10 # It was set in the parent class in the original article
        self.capacity_violation_cost = 30 # Penalty of exceeding storage capacity expressed in cost/item

        self.retailers = retailers # List of retailers served by the DC
        
        self.reset()
        
    def reset(self):
        # Use the correct dtype
        self.order_quantity_limit = np.array(self.order_quantity_limit, dtype=np.int16)
        self.capacity = np.array(self.capacity, dtype=np.int16)
        # Initialize variables
        self.I = np.array(self.I0, dtype = np.int16) # On-hand inventory at the start of each period
        self.I_surplus = np.array(0, dtype=np.int16) # Number of units by which capacity constraint is violated at the start of each period
        self.T = np.array(0, dtype=np.int16) # Pipeline inventory at the start of each period
        self.backlog = np.array(0, dtype=np.int16) # Quantity that DC requests to supplier but supplier denies/is unable to provide
        self.inv_pos = self.I+self.T-self.backlog # Inventory position = On hand inventory + Pipeline inventory - Backlogged demand
        # Initialize action record
        self.action_log = [] # Orders placed to supplier
        # Initialize counters
        self.n_orders = 0 # Number of orders placed by the DC to the supplier in the current period

    def place_order(self, action , current_period):
        '''
        Args:
            action : ordered quantity from the DC to the supplier
            current period
        Returns: 
            accepted order: stored in action_log as a list [arrival_time , ordered_quantity]
            backlog : difference between actual and accepted order
        '''
        if action > 0 :
            self.action_log.append([current_period + self.lead_time , min(self.order_quantity_limit, action)])
            self.backlog = np.minimum(0 , action - self.order_quantity_limit, dtype=np.int16) # Backlog is the difference between the order placed and the order accepted
            self.n_orders = 1 # Reset the number of orders placed by the DC to the supplier in the current period
        else:
            self.n_orders = 0 # Reset the number of orders placed by the DC to the supplier in the current period

    def receive_order(self, current_period):
        # Receive order
        if len(self.action_log) > 0 :
            if current_period == self.action_log[0][0] :
                # self.I = min(self.capacity , self.I + self.action_log[0][1])
                self.I += self.action_log[0][1]
                self.action_log.pop(0)
        
        # Check if inventory at hand surpasses storage capacity
        self.I_surplus = max(0, self.I-self.capacity)

    def satisfy_demand(self, retailers, actions, current_period):

        #> 0. Initialize arrarys to accumulate demnand satisfaction to compute the number of orders placed
        total_demand_satisfied = np.zeros(len(retailers), dtype=np.int16) # sum of backlogged and current demand satisfied in current period

        #> 1. Prioritize satisfaction of backlogged demand
        if np.sum(retailer.backlog for retailer in retailers) > 0 :
            for i, retailer in enumerate(retailers):
                if self.I <= retailer.backlog :
                    retailer.backlog -= self.I
                    total_demand_satisfied[i] += self.I
                    self.I = np.array(0, dtype=np.int16)
                else:
                    self.I -= retailer.backlog
                    total_demand_satisfied[i] += retailer.backlog
                    retailer.backlog = np.array(0, dtype=np.int16)

        #> 2. Satisfy current period demand
        if self.I > 0 :
            for i, retailer in enumerate(retailers):
                if self.I <= actions[i]:
                    retailer.backlog = actions[i] - self.I
                    total_demand_satisfied[i] += self.I
                    self.I = np.array(0, dtype=np.int16)
                else:
                    self.I -= actions[i]
                    total_demand_satisfied[i] += actions[i]
                    retailer.backlog = np.array(0, dtype=np.int16)
        else:
            for i, retailer in enumerate(retailers):
                retailer.backlog = actions[i]

        #> 3. Assess the number of batches into which the order placed by each retailer must be divided
        for i, retailer in enumerate(retailers):
            quantity_left = total_demand_satisfied[i]
            n_orders = 0
            while quantity_left > 0:
                if quantity_left > retailer.order_quantity_limit:
                    retailer.order_arrival_list.append((current_period + retailer.lead_time , retailer.order_quantity_limit))
                    quantity_left -= retailer.order_quantity_limit
                else:
                    retailer.order_arrival_list.append((current_period + retailer.lead_time , quantity_left))
                    quantity_left = 0
                n_orders += 1
            retailer.n_orders = n_orders # Store the number of orders placed by each retailer to the DC in the current period
        
        #> 4. Update pipeline inventory and total inventory position
        self.T = np.sum(order[1] for order in self.action_log)
        self.inv_pos = self.I + self.T - np.sum(retailer.backlog for retailer in retailers) # Inventory position = On hand inventory + Pipeline inventory - Backlogged demand

class Retailer():
    #TBD add a penalty for unsatisfied demand (lost sales) instead of a backlog to retailer
    def __init__(self, *args, **kwargs):
        self.I0 = 25
        self.lead_time = 2
        self.capacity = 75
        self.order_quantity_limit = 50
        self.holding_cost = 3
        self.fix_order_cost = 50
        self.unit_price = 50
        self.lost_sales_cost = 5
        self.capacity_violation_cost = self.unit_price/4 # Penalty for exceeding storage capacity expressed in cost/item
        
        self.reset()

    def reset(self, *args):
        # Use the correct dtype
        self.order_quantity_limit = np.array(self.order_quantity_limit, dtype=np.int16)
        self.capacity = np.array(self.capacity, dtype=np.int16)
        # Initialize variables
        self.I = np.array(self.I0, dtype=np.int16)
        self.I_surplus = np.array(0, dtype=np.int16) # Number of units by which capacity constraint is violated at the start of each period
        self.T = np.array(0, dtype=np.int16) # Pipeline inventory at the start of each period
        self.SD = np.array(0, dtype=np.int16) # Sales performed at each period 
        self.UD = np.array(0, dtype=np.int16) # Unsatisfied demand at each period
        self.backlog = np.array(0, dtype=np.int16) # Quantity that retailer requests to DC but DC denies/is unable to provide
        self.inv_pos = self.I + self.T - self.backlog # Inventory position = On hand inventory + Pipeline inventory - Backlogged demand
        self.order_arrival_list = [] # Orders placed to supplier
        # Initialize counters
        self.n_orders = 0 # Number of orders placed by the retailer to the supplier in the current period
        # Initialize action record
        if len(args) > 0:
            self.demands_episode = args[0] # Demand log for the current episode

    def place_order(self):
        '''
        orders to be sent from DC to retailer are placed and computed in the method "place_order" of the DC class
        '''

    def receive_order(self, current_period):
        n_orders = 0
        new_order_arrival_list = []

        for arrival_time, order_quantity in self.order_arrival_list:
            if arrival_time == current_period:
                self.I += order_quantity
                n_orders += 1
            else:
                new_order_arrival_list.append((arrival_time, order_quantity))
        self.order_arrival_list = new_order_arrival_list

        # Check if inventory at hand surpasses storage capacity
        self.I_surplus = max(0, self.I-self.capacity)

        return n_orders
    
    def satisfy_demand(self, current_period):
        demand = self.demands_episode[current_period] 
        self.SD = np.minimum(demand, self.I, dtype=self.I.dtype)
        self.I -= self.SD
        self.UD = demand - self.SD
        self.T = np.sum(order for _ , order in self.order_arrival_list)
        self.inv_pos = self.I + self.T - 0 # Inventory position = On hand inventory + Pipeline inventory. Retailers do not have backlogged demand; unfulfilled final customer demand is lost at a penalty cost.

class frozen_poisson():
    '''
    This class is used to freeze the parameters of the poisson distribution. It is used to avoid passing the parameters every time a new sample is generated.
    '''
    def __init__(self, random_state=None):
        self.random_state = random_state

    def rvs(self, mu = 0 , size=1):
        if not hasattr(self, 'rng'):
            if self.random_state is None:
                self.rng = np.random.Generator(np.random.PCG64())
            else:
                self.rng = np.random.Generator(np.random.PCG64(seed=self.random_state))
        return self.rng.poisson(mu, size=size)

    def pmf(self, value , mu = 0):
        return poisson.pmf(value, mu=mu)