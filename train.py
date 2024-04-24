import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
from dqn import DQN, Agent
from matplotlib import pyplot as plt
import time
import sys
import warnings
warnings.filterwarnings("ignore")

def main(atari_game, num_episodes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'gpu count: {torch.cuda.device_count()}')

    # Setup environment
    env = gym.make(atari_game, obs_type="grayscale")
    atari_game = atari_game.replace('/', '_')
    num_actions = env.action_space.n
    num_observations = env.observation_space.shape[0] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, device)

    num_episodes = num_episodes
    batch_size = 1
    target_update_freq = 100

    min_loss = np.inf
    episode_rewards=[]
    episode_loss=[]
    start = time.time()

    for episode in range(num_episodes):
        #print("episode:", episode)
        state = env.reset()[0]
        # state = preProcess(state)  # Preprocess the initial state
        done = False
        episode_reward = 0
        loss=[]
        while not done:
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
            
            cur_loss, _ = agent.train(batch_size, target_update_freq)

            loss.append(cur_loss)
        
        loss=[l for l in loss if l !=float('inf')]
        mean_loss=sum(loss)/len(loss)
        
        if mean_loss < min_loss:
            torch.save(agent.model, f'pretrained_model_{atari_game}.pth')
        
        if episode % target_update_freq == 0:
            torch.save(agent.model.state_dict(), f'cache_model_{atari_game}.pth')
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        if episode == 0:
            print(f'each episode takes approx. {time.time()-start} seconds')

        episode_rewards.append(episode_reward)
        episode_loss.append(mean_loss)
        
    print('training process done!')
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(range(num_episodes),episode_rewards)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')

    axs[1].plot(range(num_episodes),episode_loss)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Average Loss')
    plt.savefig(f'subplots_{atari_game}.png')

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py atari_game num_episodes")
        sys.exit(1)

    # Parse command-line arguments
    atari_game = sys.argv[1]
    num_episodes = int(sys.argv[2])
    main(atari_game, num_episodes)
