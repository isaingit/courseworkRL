import numpy as np 
import torch
from torch.distributions import MultivariateNormal

from utils import setup_logging, setup_model_saving, read_log_file
from IMP_CW_env import MESCEnv


#region NEURAL NETWORKS DEFINITION: POLICY AND VALUE NETWORKS
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

class ValueNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#endregion

#region TRAINING LOOP
def train_REINFORCE(env, **kwargs):
    if kwargs.get("logging",True):
        log_f_name = setup_logging()
        print(f"Log file saved in: {log_f_name}")
        log_f = open(log_f_name, 'w+')
        log_f.write("Iteration,Episode,Timestep,Reward,Std\n")

    # STEP 1: INITIALIZATION
    policy_net = PolicyNetwork( input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0],
                                h1_size = kwargs.get("h1_size",128),
                                h2_size = kwargs.get("h2_size",64))
    value_net = ValueNetwork(input_size=env.observation_space.shape[0])

    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=kwargs.get("lr_policy_net", 0.00005))
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=kwargs.get("lr_value_net", 0.0001))

    count_timesteps = 0
    count_episodes = 0
    count_updates = 0

    log_episode_return = []

    action_std = kwargs.get("action_std_init", .5)
    action_var = torch.full(size=(env.action_space.shape[0],) , fill_value = action_std**2)
    cov_mat = torch.diag(action_var).unsqueeze(dim=0) # ASSUMPTION: independent actions, so the covariance matrix is diagonal with variance = action_variance

    while count_timesteps < kwargs.get("max_timesteps", 4e6):

        # STEP 2: GENERATE N TRAJECTORIES
        N = kwargs.get('num_trajectories', 100)
        for _ in range(N):

            reward_buffer = []
            state_buffer = []
            state_value_buffer = []
            logprob_buffer = []
            loss_policy = []
            loss_value = []

            env.reset()
            state = env.state
            done = False

            while not done:
                # ACTION SELECTION
                # 1. Predict action mean
                action_mean = policy_net(torch.FloatTensor(state)) # Returns a tensor where each element is the mean of the action distribution. The sum of all the elements in this tensor adds up to 1.
                # 2. Decay action std (if proceed)
                if count_timesteps % kwargs.get("decay_freq", 1000) == 0:
                    action_std = max(action_std * (1-kwargs.get("decay_rate", 0.03)), kwargs.get("action_std_min", 0.05))
                    action_var = torch.full(size=(env.action_space.shape[0],) , fill_value = action_std**2)
                    cov_mat = torch.diag(action_var).unsqueeze(dim=0) 
                # 3. Sample action from normal distribution
                dist = MultivariateNormal(action_mean, cov_mat) # Creates a multivariate normal distribution
                action = dist.sample() #.detach()
                action_logprob = dist.log_prob(action) #.detach()

                # STORE DATA (PART 1)
                state_value_buffer.append(value_net(torch.FloatTensor(state)))
                logprob_buffer.append(action_logprob)

                # ACTION EXECUTION
                state , reward , done , _ = env.step(action.detach().numpy().flatten())

                # STORE DATA (PART 2)
                reward_buffer.append(reward)
                
                count_timesteps += 1

            # STEP 3: COMPUTE REWARD-TO-GO
            reward2go_buffer = []
            discounted_reward = 0
            for r in reversed(reward_buffer):
                discounted_reward = r + kwargs.get("discount_factor",.9) * discounted_reward
                reward2go_buffer.insert(0, discounted_reward)

            # AUXILIARY STEP: CONVERT LISTS TO TENSORS
            reward2go_buffer = torch.tensor(reward2go_buffer, dtype=torch.float32) 
            state_value_buffer = torch.stack(state_value_buffer)
            logprob_buffer = torch.stack(logprob_buffer)

            # STEP 4: COMPUTE POLICY LOSS
            advantages =  (reward2go_buffer - state_value_buffer).detach()
            advantages = (advantages - advantages.mean() / (advantages.std() + 1e-10))
            loss_policy.append(torch.sum(torch.mul( logprob_buffer , advantages)))

            # STEP 5: COMPUTE VALUE LOSS
            ## Normalize rewards. PPO is very sensitive to the scale of the loss funciton. If rewards are too high or too low, updated can be erratic.
            # reward2go_buffer = (reward2go_buffer - reward2go_buffer.mean()) / (reward2go_buffer.std() + 1e-10)
            loss_value.append(torch.sum((reward2go_buffer - state_value_buffer)**2)) # check if average over timesteps

            count_episodes += 1
            log_episode_return.append(sum(reward_buffer))
            
        # STEP 5: UPDATE NETWORKS
        loss_policy = (-1) * torch.stack(loss_policy).mean()
        optimizer_policy.zero_grad()
        loss_policy.backward()
        optimizer_policy.step()

        loss_value = torch.stack(loss_value).mean()
        optimizer_value.zero_grad()
        loss_value.backward()
        optimizer_value.step()

        count_updates += 1

        # LOG DATA
        if kwargs.get("logging",True): 
            # log average reward till last episode
            log_avg_reward = np.mean(log_episode_return)
            log_avg_reward = round(log_avg_reward, 4)
            log_std_reward = np.std(log_episode_return)
            # Write to log file
            log_f.write('{:.0f},{:.0f},{:.0f},{:.3f},{:.3f}\n'.format(count_updates ,count_episodes, count_timesteps, log_avg_reward, log_std_reward))
            log_f.flush()

    # Save policy
    save_path = setup_model_saving()
    torch.save(policy_net.state_dict(), save_path)
    print(f"Policy model weights saved in: {save_path}")

    # Close log file
    if kwargs.get("logging",True):
        log_f.close()

    return policy_net, save_path
#endregion TRAINING LOOP

if __name__ == "__main__":

    #region ENVIRONMENT DEFINITION
    n_retailers = 2
    n_DCs = 1
    n_suppliers = 1
    supply_chain_structure = [[n_retailers] , [n_DCs], n_suppliers]

    env_train = MESCEnv(supply_chain_structure)
    #endregion ENVIRONMENT DEFINITION

    #region HYPERPARAMETER DEFINITION
    hyperparam = {}
    hyperparam["num_trajectories"]= 3
    hyperparam["max_timesteps"] =  3 * 50 * env_train.n_periods # Stopping criteria: number of environment time steps that PPO algorithm will be executed
    hyperparam["discount_factor"] = 0.99 # Discount factor to evaluate return

    hyperparam["action_std_init"] = .5
    hyperparam["decay_freq"] = 1000
    hyperparam["decay_rate"] = 0.03
    hyperparam["action_std_min"] = 0.05
    #endregion HYPERPARAMETER DEFINITION

    optimal_policy , _ = train_REINFORCE(env_train, **hyperparam)
    log_file_path = r"C:\Users\Isabela\Desktop\GitHubRepos\courseworkRL\REINFORCE_logs\REINFORCE_log_0.csv"
    timesteps, rewards = read_log_file(log_file_path)