import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
from dqn import DQN_MLP, DQN_CONV, Agent, prepro
from matplotlib import pyplot as plt
import time
import sys
import warnings
warnings.filterwarnings("ignore")

def main(atari_game, model_name, num_episodes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'which device: {device}')
    print(f'gpu count: {torch.cuda.device_count()}')
    

    # Setup environment
    env = gym.make(atari_game, obs_type="grayscale")
    atari_game = atari_game.replace('/', '_')
    num_actions = env.action_space.n
    if 'Pong' in model_name: #Pong has two useless actions assigned as 5 and 6
        num_actions -= 2

    agent = Agent(env, model_name, device)

    # Training Parameters
    num_episodes = num_episodes
    batch_size = 32
    target_update_freq = 100
    warmup_ep = 10

    min_loss = np.inf
    episode_rewards=[]
    episode_loss=[]
    num_steps = []
    print(torch.cuda.memory_summary())

    total_steps = 0
    
    print('agent warming up!')

    for episode in range(num_episodes+warmup_ep): # warmup with the first ten episodes
        #print("episode:", episode)
        state = env.reset()[0]
        start = time.time()
        # state = preProcess(state)  # Preprocess the initial state
        done = False
        episode_reward = 0
        loss=[]
        step_per_ep = 0
        
        if episode == warmup_ep:
            print('start training!')

        while not done:
            step_per_ep += 1
            total_steps += 1
            # print("while not done:", state.shape)
            action = agent.select_action(state)
            # print("action:", action)
            next_state, reward, done, truncated, info = env.step(action)
            # next_state = preProcess(next_state)  # Preprocess the next state
            agent.store_transition(state, action, reward, next_state, done)
            # print("next state:", next_state.shape)
            episode_reward += reward
            state = next_state
            # print("update to new state", state.shape)
            
            if episode >= warmup_ep:
                cur_loss, _ = agent.train(batch_size, target_update_freq)
                loss.append(cur_loss)
            
                if total_steps % 10000 == 0:
                    agent.update_target_model()
                    torch.cuda.empty_cache()
            
        
        if episode >= warmup_ep:
            loss=[l for l in loss if l !=float('inf')]
            mean_loss=sum(loss)/len(loss)
        
            if episode >= 200 and episode % 100 == 0:
                agent.update_epsilon()

            if mean_loss < min_loss:
                torch.save(agent.model, f'pretrained_model_{atari_game}_{model_name}_{num_episodes}.pth')

            if episode % 100 == 0:
                torch.save(agent.model.state_dict(), f'cache_model_{atari_game}_{model_name}_{num_episodes}.pth')
            print(f"Episode {episode + 1 - warmup_ep}, Reward: {episode_reward}, Mean Loss: {mean_loss}, Number of Steps: {step_per_ep}")
            
            num_steps.append(step_per_ep)
            episode_rewards.append(episode_reward)
            episode_loss.append(mean_loss)
        
        if episode == warmup_ep:
            print(f'each episode takes approx. {time.time()-start} seconds')
        
    print('training process done!')
    
    

    np.save(f'episode_rewards_{atari_game}_{model_name}_{num_episodes}.npy', np.array(episode_rewards))
    np.save(f'episode_loss_{atari_game}_{model_name}_{num_episodes}.npy', np.array(episode_loss))
    np.save(f'steps_per_ep_{atari_game}_{model_name}_{num_episodes}.npy', np.array(episode_loss))

    i=1
    
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




    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(range(num_episodes),moving_reward_averages)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward (3 Episode Average)')

    axs[1].plot(range(num_episodes),moving_loss_averages)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Average Loss (3 Episode Average)')
    plt.savefig(f'subplots_{atari_game}_{model_name}_{num_episodes}.png')

    plt.close()

    plt.plot(range(num_episodes),episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (3 Episode Average)')
    plt.savefig(f'model_reward_{atari_game}_{model_name}_{num_episodes}.png')
    

if __name__=="__main__":
    if len(sys.argv) != 4:
        print("Usage: python train.py atari_game model_name num_episodes")
        sys.exit(1)
    
    if sys.argv[2] not in ["DQN_MLP", "DQN_CONV", "DQN_CONVLSTM"]:
        print("Model not recognized, please use [DQN_MLP, DQN_CONV, DQN_CONVLSTM] as model_name")
        sys.exit(1)

    # Parse command-line arguments
    atari_game = sys.argv[1]
    model_name = sys.argv[2]
    num_episodes = int(sys.argv[3])
    main(atari_game, model_name, num_episodes)