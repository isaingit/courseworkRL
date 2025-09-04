from IMP_CW_env import MESCEnv
import numpy as np 
import multiprocessing as mp
import time

from heuristic_policy import HeuristicPolicy, Optimizer
from PPO_file import PPO
from utils import plot_reward_evolution, read_log_file, plot_reward_distribution

import matplotlib.pyplot as plt

#region ALTERNATIVE ENVIRONMENT DEFINITION

## Instantiate environment with a specific supply chain structure and demand distribution parameters
n_retailers = [2]
n_DCs = [1]
n_suppliers = 1
supply_chain_structure = [n_retailers , n_DCs, n_suppliers]

demand_dist_param = [{'mu': 5}, # Mo-Thu
                     {'mu': 7}, # Fri
                     {'mu': 20}] # Sat-Sun

env_test = MESCEnv(supply_chain_structure, demand_dist_param=demand_dist_param)

## Customize the environment with specific parameters
# Stress test capacity and lead times at distribution center
env_test.DCs[0].capacity = env_test.DCs[0].capacity/2
env_test.DCs[0].capacity_violation_cost += 20
env_test.DCs[0].lead_time += 1 
env_test.DCs[0].holding_cost = env_test.DCs[0].holding_cost * 2
# Retailer 0: premium customer 
env_test.retailers[0].lead_time = 1
env_test.retailers[0].lost_sales_cost = env_test.retailers[0].lost_sales_cost * 2 
env_test.retailers[0].capacity_violation_cost = env_test.retailers[0].capacity_violation_cost * 2
# Retailer 2: higher rotating inventory
env_test.retailers[0].capacity -= 25
env_test.retailers[1].fix_order_cost = env_test.retailers[1].fix_order_cost / 2
# Sold product is more expensive
env_test.retailers[0].unit_price += 10
env_test.retailers[1].unit_price += 10

## Topography settings
num_states = env_test.observation_space.shape[0]
num_actions = env_test.action_space.shape[0]

#endregion ALTERNATIVE ENVIRONMENT DEFINITION

#region FUNCION TO RUN PPO AGENT
def run_ppo_with_timeout(func, args, timeout):
    entering_time = time.time()
    print(f"Entering run_ppo_with_timeout.")
    manager = mp.Manager()
    model_dict = manager.dict()

    p = mp.Process(target=func, 
                args = (args,),
                kwargs= {'model_dict': model_dict,
                         'logging': False, 
                         'verbose': False})
    
    start_time = time.time() 
    print("Starting PPO training with timeout after {:.2f} seconds...".format(start_time - entering_time))

    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        print("Timeout reached. Saving the latest policy.")
        p.terminate()
        p.join()

    print(f"PPO training completed in {time.time() - start_time:.2f} seconds.")

    return model_dict['actor_model'].copy()
#endregion FUNCTION TO RUN PPO AGENT

#region EXECUTION
if __name__ == "__main__":
    
    # STEP 1: Instantiate PPO agent using students' hyperparameter choices
    #region PPO AGENT
    
    ## Paste students' choice of hyperparameters here
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    num_epochs = 20 # Number of actor and critic network updates per batch of data
    max_train_steps =  100 * env_test.n_periods # Stopping criteria: number of environment time steps that PPO algorithm will be executed
    update_timestep = env_test.n_periods * 3 # How often to update actor and critic networks

    lr_actor = 0.00005
    lr_critic = 0.0001

    eps_clip = 0.2 # Value of epsilon in PPO clipping
    gamma = 0.99 # Discount factor to evaluate return

    action_std_init = .5
    decay_freq = 1000
    decay_rate = 0.03
    action_std_min = 0.05

    log_freq = 1 # Number of episodes after which the log file is updated
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    ## Create PPO agent
    ## TODO: alternative to get hyperparameters as a dict and unpack it upon instantiation
    PPOagent = PPO( num_states, num_actions, lr_actor, lr_critic,
                    action_std_init=action_std_init, decay_freq=decay_freq, decay_rate=decay_rate, action_std_min=action_std_min,
                    num_epochs=num_epochs, gamma=gamma, eps_clip=eps_clip, max_train_steps=max_train_steps, update_timestep=update_timestep,
                    log_freq=log_freq)
    ## Activate test mode to to save checkpoints of the actor model in order to retrieve the latest policy if timeout is reached.
    PPOagent.test_mode = True 

    #endregion INSTANTIATE PPO AGENT

    # STEP 2: Run PPO agent using students' function
    timeout = 60 # Timeout in seconds
    students_function = PPOagent.train
    output = run_ppo_with_timeout(students_function, env_test, timeout)

    # STEP 3: Load the latest policy into PPO agent
    PPOagent.policy.actor.load_state_dict(output)
    PPOagent.policy_old.actor.load_state_dict(output)

    # STEP 3: Generate test dataset
    ## TODO: instead of generating a test dataset everytime, either load it or just specify the seed (ok if not to be compared to (s,S) policy)
    test_demand_dataset = []
    env_test.seed = 42 # Set seed for reproducibility
    for _ in range(0,100):
        demands_episode, _ = env_test.sample_demands_episode()
        test_demand_dataset.append(demands_episode)

    # STEP 4: Evaluate PPO agent on test dataset
    reward_list = PPOagent.evaluate_policy(env_test, test_demand_dataset)
    print("PPO performance:\n - Average reward: {:.0f}\n - Reward standard deviation: {:.2f}".format(np.mean(reward_list), np.std(reward_list)))
