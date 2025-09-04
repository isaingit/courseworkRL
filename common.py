import numpy as np
import torch

'''
File containing common functions shared by the algorithms that model the policy with a neural network
'''
class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, h1_size = 128, h2_size = 64): 
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.fc3 = torch.nn.Linear(h2_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    
    def sample_action(self, state):
        # Pre-process state
        state = torch.FloatTensor(state)

        # Get action according to the policy network
        action_mean = self(state)
        action_mean = action_mean.detach().numpy()
        action_mean = np.floor(action_mean)

        return action_mean

def reward_fcn(policy_net, env, num_episodes=10, demand=None):
    '''
    Runs a series of episodes and computes the average total return.

    Arguments:
    - policy_net --> Neural network that predicts the optimal action given the state 
    - env --> Instance of MESCEnv environment
    - num_episodes --> Number of runs or episodes to estimate the average return (optional, default: 10)
    - demand --> List of scenario sets, containing realizations of customers' demand for each time step of each episode (optional, default: None)

    Returns:
    - mean_reward --> Average reward across runs
    - std_reward --> Standar deviation across runs
    '''
    # Input checking
    assert num_episodes > 0, "Number of episodes must be greater than 0"
    
    # Fix customer demand (if provided)
    env.demand_dataset = demand

    # Initialize buffer list to store results of each run
    reward_list = []

    # Run each episode and compute total undiscounted reward
    for i in range(num_episodes):
        # Reset environment before each episode
        env.reset()
        state = env.state
        episode_terminated = False
        # Initialize reward counter
        total_reward = 0
        
        while episode_terminated == False:
            # Sample action
            action_mean = policy_net.sample_action(state)
            
            # Interact with the environment to get reward and next state
            state , reward, episode_terminated, _ = env.step(action_mean)
            total_reward += reward
            
        reward_list.append(total_reward)

    # Compute mean and standard deviation
    mean_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)
    
    return mean_reward, std_reward