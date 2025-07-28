import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl   

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
    return x, y

def plot_reward_evolution(timesteps, rewards, save_dir=None, ):
    
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
    
    rewards_smooth = smooth(rewards, 0.75)
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

def plot_reward_distribution(reward_arr, labels=None):
    '''
    Arguments:
        reward_arr : must be a [n_episodes, n_data_sets] array
    '''
    if  isinstance(reward_arr, list) or reward_arr.ndim == 1:
        fig = plt.figure(figsize=(5,5))
        plt.hist(reward_arr, density = True);
        plt.xlabel('Empirical expected reward (MU)') # TODO :  update monetary units to actual units
        plt.ylabel('Probability density')
        if labels is not None:
            plt.legend(labels)

    else:
        _ , axs = plt.subplots(1, reward_arr.shape[1], figsize=(5*reward_arr.shape[1], 5))
        for i in range(reward_arr.shape[1]):
            axs[i].hist(reward_arr[:, i], density=True)
            axs[i].set_xlabel('Empirical expected reward (MU)')
            axs[i].set_ylabel('Probability density')
            if labels is not None:
                axs[i].set_title(labels[i])

def setup_logging():

    # log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs" + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    log_f_name = log_dir + "/PPO_log_" + str(run_num) + ".csv"

    return log_f_name

def setup_model_saving():

    # Actor nn models for multiple runs are NOT overwritten
    save_dir = "ActorModels" + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(save_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    save_f_name = save_dir + "/PPO_ActorModel_" + str(run_num) + ".pt"

    return save_f_name

