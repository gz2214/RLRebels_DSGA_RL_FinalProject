import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import numpy as np 
from dqn import DQN_CONV,DQN_MLP,Agent
import matplotlib.pyplot as plt
import time
import sys

def main(pretrained_model_name,atari_game_2,num_episodes):
    """
    Main function for transfer learning with reinforcement learning using Deep Q-Network (DQN).

    Args:
        pretrained_model_name (str): Name of the pre-trained model file (without extension).
        atari_game_2 (str): Name of the Atari game to be used.
        num_episodes (int): Number of episodes for training.

    Returns:
        None

    This function initializes the environment, loads a pre-trained model, adds a transfer learning layer,
    sets up the agent with the combined model, and then proceeds with training over the specified number of episodes.
    It saves the trained model, episode rewards, and episode losses. Additionally, it plots and saves graphs
    of cumulative moving averages of rewards and losses over the training episodes.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'gpu count: {torch.cuda.device_count()}')
    ## set up environment
    env = gym.make(atari_game_2, obs_type="grayscale")
    atari_game_2 = atari_game_2.replace('/', '_')
    num_actions = env.action_space.n
    num_observations = env.observation_space.shape[0] 

    ## freeze other layers parameters so they do not change

    pre_trained_model=torch.load(f'{pretrained_model_name}.pth',map_location=device)
    in_feat=pre_trained_model.fc1.out_features
    for p in pre_trained_model.parameters():
        p.requires_grad=False
    ## add layer to pretrain model 
    transferLearningLayer=nn.Linear(in_features=in_feat, out_features=num_actions)
    for param in transferLearningLayer.parameters():
        param.requires_grad = True
    ## add model to agent
    agent = Agent(env,'DQN_MLP',device)
    agent.model=nn.Sequential(pre_trained_model,transferLearningLayer)

    num_episodes = num_episodes
    batch_size = 1
    target_update_freq = 100

    min_loss = np.inf
    episode_rewards=[]
    episode_loss=[]
    start_time = time.time()

    total_steps = 0
    #train
    for episode in range(num_episodes):
        #print("episode:", episode)
        state = env.reset()[0]
        start = time.time()
        # state = preProcess(state)  # Preprocess the initial state
        done = False
        episode_reward = 0
        loss1=[]
        step_per_ep = 0
        episode_steps=[]

        while not done:
            step_per_ep += 1
            total_steps += 1
            action = agent.select_action(state)
            # print("action:", action)
            next_state, reward, done, truncated, info = env.step(action)
            # next_state = preProcess(next_state)  # Preprocess the next state
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            
            cur_loss, _ = agent.train(batch_size, target_update_freq)

            loss1.append(cur_loss)
            
            if total_steps % 10000 == 0:
                agent.update_target_model
        
        loss1=[l for l in loss1 if l !=float('inf')]
        mean_loss=sum(loss1)/len(loss1)
        
        if episode >= 200 and episode % 50 == 0:
            agent.update_epsilon()
        
        if mean_loss < min_loss:
            torch.save(agent.model, f'transferLearningModel_model_{atari_game_2}_{pretrained_model_name}.pth')
        
        if episode % 100 == 0:
            torch.save(agent.model.state_dict(), f'cache_model_transfer_learning_{atari_game_2}_from_{pretrained_model_name}.pth')
        print(f"Episode {episode + 1}, Reward: {episode_reward}, Mean Loss: {mean_loss}, Number of Steps: {step_per_ep}")
        if episode == 0:
            print(f'each episode takes approx. {time.time()-start} seconds')

        episode_rewards.append(episode_reward)
        episode_loss.append(mean_loss)
        episode_steps.append(step_per_ep)
        
        np.save(f'episode_rewards_{atari_game_2}_transferLearning_from_{pretrained_model_name}.npy', np.array(episode_rewards))
        np.save(f'episode_loss_{atari_game_2}_transferLearning_from_{pretrained_model_name}.npy', np.array(episode_loss))
        np.save(f'episode_steps_{atari_game_2}_transferLearning_from_{pretrained_model_name}.npy', np.array(episode_steps))
        
    print('training process done!')


    
    i = 1
# Initialize an empty list to store cumulative moving averages
    moving_reward_averages = []
# Store cumulative sums of array in cum_sum array
    cum_sum_reward=np.cumsum(episode_rewards)
# Loop through the array elements
    while i <= len(episode_rewards):
# Calculate the cumulative average by dividing cumulative sum by number of elements till that position
        window_average_reward = round(cum_sum_reward[i-1] / i, 2)
# Store the cumulative average of
# current window in moving average list
        moving_reward_averages.append(window_average_reward)
# Shift window to right by one position
        i += 1

    i=1
# Initialize an empty list to store cumulative moving averages
    moving_loss_averages = []
# Store cumulative sums of array in cum_sum array
    cum_sum_loss=np.cumsum(episode_loss)
# Loop through the array elements
    while i <= len(episode_loss):
# Calculate the cumulative average by dividing cumulative sum by number of elements till that position
        window_average_loss = round(cum_sum_loss[i-1] / i, 2)
# Store the cumulative average of
# current window in moving average list
        moving_loss_averages.append(window_average_loss)
# Shift window to right by one position
        i += 1
#graph results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(range(num_episodes),moving_reward_averages)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward (3 Episode Average)')

    axs[1].plot(range(num_episodes),moving_loss_averages)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Average Loss (3 Episode Average)')

    plt.savefig(f'transfer_model_subplots_{atari_game_2}.png')

    plt.close()

    plt.plot(range(num_episodes),episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (3 Episode Average)')
    plt.savefig(f'transfer_model_rewards_{atari_game_2}.png')

    end_time = time.time()

# Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Elapsed time:", elapsed_time, "seconds")

    
if __name__=="__main__":

    # Parse command-line arguments
    pretrained_model=sys.argv[1]
    atari_game_2 = sys.argv[2]
    num_episodes = int(sys.argv[3])
    main(pretrained_model,atari_game_2, num_episodes)

