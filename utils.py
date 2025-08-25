import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl   
import torch

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
mpl.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize

mpl.rc("figure", figsize=(10, 5))
mpl.rc("savefig", bbox="tight", dpi=600)

plt.ion()  # Enable interactive mode for live plotting

def read_log_file(log_file_path):
    df = pd.read_csv(log_file_path)
    x = df['Timestep']
    y = df['Reward']
    try: df['Std']
    except:
        return x, y
    else:
        return x, y, df['Std']

def plot_reward_evolution(timesteps, rewards, save_dir=None, reward_std=None): 
    
    def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
        '''
        Implementation of the exponential moving average (EMA) method for smoothing a series
        '''
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  
            smoothed.append(smoothed_val)                        
            last = smoothed_val                                  
            
        return smoothed
    
    if reward_std is None:
        rewards_smooth = smooth(rewards, 0.95)
        timesteps_smooth = timesteps[:len(rewards_smooth)]

        # Plotting
        plt.clf()
        plt.plot(timesteps, rewards, label='Average reward', color='dodgerblue', alpha=0.5)
        plt.plot(timesteps_smooth, rewards_smooth, label='Smoothed reward (EMA)', color='orange', linewidth=2)

        # Labels and title
        plt.xlabel("Time steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()
    else:
        plt.clf()
        plt.plot(timesteps, rewards, label='Average reward', color='blue')
        plt.fill_between(timesteps, 
                         np.array(rewards)-np.array(reward_std), 
                         np.array(rewards)+np.array(reward_std) , 
                         color='dodgerblue', alpha=0.2)
        best_reward = np.max(rewards)
        best_reward_idx = np.argmax(rewards)
        plt.plot(timesteps[best_reward_idx], best_reward, 'o', color='red', zorder=5)

        # Labels and title
        plt.xlabel("Iterations")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()

def plot_reward_distribution(rewards, labels=None):
    '''
    Arguments:
        reward_arr : must be a [n_episodes, n_data_sets] array
    '''
    # Colors for each dataset
    colors = ['skyblue', 'salmon', 'lightgreen']

    if  isinstance(rewards, list) or rewards.ndim == 1:
        fig = plt.figure(figsize=(5,5))
        plt.hist(rewards, density = True, edgecolor='black', alpha=0.5);
        plt.xlabel('Empirical expected reward (MU)') # TODO :  update monetary units to actual units
        plt.ylabel('Probability density')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-4,4))
        if labels is not None:
            plt.legend(labels)

    else:
        if np.std(rewards.mean(axis=0)) > 100000:
            # Plot each dataset in a separate subplot:
            _ , axs = plt.subplots(1, rewards.shape[1], figsize=(5*rewards.shape[1], 5))
            for i in range(rewards.shape[1]):
                axs[i].hist(rewards[:, i], density=True, edgecolor='black', alpha=0.5, color=colors[i % len(colors)])
                axs[i].set_xlabel('Empirical expected reward (MU)')
                axs[i].set_ylabel('Probability density')
                axs[i].ticklabel_format(style='sci', axis='x', scilimits=(-4,4))
                if labels is not None:
                    axs[i].set_title(labels[i])
        else:
            for i in range(rewards.shape[1]):
                plt.hist(rewards[:, i], density=True, edgecolor='black', alpha=0.5, color=colors[i % len(colors)])
                plt.xlabel('Empirical expected reward (MU)')
                plt.ylabel('Probability density')
                plt.ticklabel_format(style='sci', axis='x', scilimits=(-4,4))
                if labels is not None:
                    plt.legend(labels)

def setup_logging(algorithm = "REINFORCE"):

    # log files for multiple runs are NOT overwritten
    log_dir = algorithm + '_logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    log_f_name = log_dir + "/" + algorithm + "_log_" + str(run_num) + ".csv"

    return log_f_name

def setup_model_saving(algorithm = "REINFORCE"):

    # Actor nn models for multiple runs are NOT overwritten
    save_dir = algorithm + "_policies" + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(save_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    save_f_name = save_dir + "/"+ algorithm + "_policy_" + str(run_num) + ".pt"

    return save_f_name

def sample_uniform_params(params_prev, param_min, param_max):
    '''
    Sample random point within given parameter bounds. Tailored for EXPLORATORY purposes
    '''
    params = {k: torch.rand(v.shape) * (param_max - param_min) + param_min \
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, param_min, param_max):
    '''
    Sample a random point in the neighborhood of a given point or value or the parameters (v). Tailored for EXPLOITATION purposes

    Explanation:
    sign = (torch.randint(2, (v.shape)) * 2 - 1) # This returns either -1 or 1
    eps = torch.rand(v.shape) * (param_max - param_min) # This returns the width of the step to be taken in the modification of the parameters
    Hence, the total update is: v + sign*eps.
    '''
    params = {k: torch.rand(v.shape) * (param_max - param_min) * (torch.randint(2, (v.shape))*2 - 1) + v \
              for k, v in params_prev.items()}
    return params

