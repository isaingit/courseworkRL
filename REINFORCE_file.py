import numpy as np 
import torch
from torch.distributions import MultivariateNormal

from utils import *
from IMP_CW_env import MESCEnv

#region NEURAL NETWORKS DEFINITION: POLICY AND VALUE NETWORKS
class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, h1_size = 128, h2_size = 64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.fc3 = torch.nn.Linear(h2_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

class ValueNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#endregion

#region REINFORCE IMPLEMENTATION
class REINFORCE():
    def __init__(self, lr_policy_net= 5e-5, lr_value_net = 1e-4, discount_factor = .99 , max_steps = 1e5, weight_entropy = 0.001, action_std_init = .5):
        # Set hyperparameter values
        self.lr_policy_net = lr_policy_net
        self.lr_value_net = lr_value_net
        self.discount_factor = discount_factor
        self.max_steps = max_steps
        self.weight_entropy = weight_entropy
        self.action_std = action_std_init

        # Initialize interal attributes
        self.counter_timesteps = 0

        # Define paths to store data
        self.log_f_path = setup_logging(algorithm="REINFORCE")
        self.save_f_path = setup_model_saving(algorithm="REINFORCE")
        
        
    def choose_action(self, state, policy_net):
        '''
        Sample action in continuous action space modelled with a Multivariate Normal distribution
        '''
        # Predict action mean from Policy Network
        action_mean = policy_net(torch.from_numpy(state).float())

        # Estimate action variance (decaying action std)
        action_var = torch.full(size=(policy_net.fc3.out_features,) , fill_value = self.action_std**2)
        cov_mat = torch.diag(action_var).unsqueeze(dim=0) 

        # Generate Multivariate Normal distribution with estimated mean and variance
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        # Sample action
        action = dist.sample()

        # Compute logprob and entropy
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob , entropy

    def rollout(self, env, policy_net, value_net):
        """
        Runs an episode of experience and collects relevant information
        """
        trajectory = {}
        trajectory["values"] = []
        trajectory["actions"] = []
        trajectory["logprobs"] = []
        trajectory["rewards"] = []
        trajectory["entropies"] = []

        done = False
        env.reset()
        state = env.state
        
        while not done:
            action , action_logprob , entropy = self.choose_action(state, policy_net,)
            next_state , reward , done , _ = env.step(action.detach().numpy().flatten())

            trajectory["values"].append(value_net(torch.from_numpy(state).float()))
            trajectory["logprobs"].append(action_logprob)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

            state = next_state
            self.counter_timesteps += 1

        return trajectory

    def calculate_returns(self, rewards, discount_factor):
        '''
        Computes discounted return G_t.
            G_t = sum_{k=t+1}{T} {gamma^(k-t-1) R_k}
            where R_k is the reward of timestep k
        '''
        discounted_return = 0
        returns = []
        for r in reversed(rewards):
            discounted_return = r + discount_factor * discounted_return
            returns.insert(0, discounted_return)
        return torch.tensor(returns , dtype=torch.float32)
        
    def train(  self,
                env, *, 
                h1_size = 128, 
                h2_size = 64):
        
        # Create log file
        self.log_f = open(self.log_f_path, 'w+')
        self.log_f.write("Timestep,Reward\n")
        
        # Initialize policies and optimizers
        policy_net = PolicyNetwork( input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0],
                                    h1_size = h1_size,
                                    h2_size = h2_size)
        value_net = ValueNetwork(input_size=env.observation_space.shape[0])
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=self.lr_policy_net)
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=self.lr_value_net)

        self.counter_timesteps = 0

        while self.counter_timesteps < self.max_steps:
            # Generate an episode following policy_net
            trajectory = self.rollout(env_train, policy_net, value_net)
            
            logprobs = torch.stack(trajectory["logprobs"]).squeeze() # shape : (episode_length, )
            entropies = torch.stack(trajectory["entropies"]).squeeze() # shape : (episode_length, )
            values = torch.stack(trajectory["values"]).squeeze() # shape : (episode_length, )

            returns = self.calculate_returns(trajectory["rewards"], discount_factor=.99) # shape : (episode_length, )

            advantages = returns - values

            # Update Policy Network
            loss_policy = (-1) * torch.mean(advantages.detach() * logprobs) + self.weight_entropy * ((-1) * torch.mean(entropies))
            optimizer_policy.zero_grad()
            loss_policy.backward()
            # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
            optimizer_policy.step()

            # Update Value Network
            loss_value = torch.nn.functional.mse_loss(values , returns)
            optimizer_value.zero_grad()
            loss_value.backward()
            # torch.nn.utils.clip_grad_norm_(value_net.parameters(), float('inf'))
            optimizer_value.step()

            # Write episode undiscounted return evolution to log file
            log_return = round(np.mean(sum(trajectory["rewards"])), 4)
            self.log_f.write('{:.0f},{:.3f}\n'.format(self.counter_timesteps, log_return))
            self.log_f.flush() 

        # Close log file
        self.log_f.close()
        print(f"Log file saved in: {self.log_f_path}") 

        # Save agent policy
        torch.save(policy_net.state_dict(), self.save_f_path)
        print(f"Policy model weights saved in: {self.save_f_path}") 
    

#endregion


if __name__ == "__main__":

    #region ENVIRONMENT DEFINITION
    n_retailers = 2
    n_DCs = 1
    n_suppliers = 1
    supply_chain_structure = [[n_retailers] , [n_DCs], n_suppliers]

    env_train = MESCEnv(supply_chain_structure, num_periods = 7*4)
    #endregion ENVIRONMENT DEFINITION

    #region HYPERPARAMETER DEFINITION
    hyperparam = {}

    #endregion HYPERPARAMETER DEFINITION

    policy_net = PolicyNetwork(input_size=env_train.observation_space.shape[0], output_size=env_train.action_space.shape[0],)
    agent = REINFORCE(max_steps=28*1000)
    agent.train(env_train)
    timesteps , rewards = read_log_file(agent.log_f_path)
    plot_reward_evolution(timesteps, rewards)
    plt.show()