import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
from dqn import DQN, Agent

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup environment
    env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    num_actions = env.action_space.n
    num_observations = env.observation_space.shape[0] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, device)

    num_episodes = 4
    batch_size = 1
    target_update_freq = 2

    min_loss = np.inf
    episode_rewards=[]
    episode_loss=[]

    for episode in range(num_episodes):
        print("episode:", episode)
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
            
            if episode % target_update_freq == 0:
                agent.update_target_model()

            if cur_loss < min_loss:
                torch.save(agent.model, 'cache_model.pth')

            loss.append(cur_loss)
        
        loss=[l for l in loss if l !=float('inf')]
        mean_loss=sum(loss)/len(loss)
        if episode % 2 == 0:
            torch.save(agent.model.state_dict(), 'pretrained_model.pth')
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        episode_rewards.append(episode_reward)
        episode_loss.append(mean_loss)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(range(num_episodes),episode_rewards)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')

    axs[1].plot(range(num_episodes),episode_loss)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Average Loss')
    plt.savefig('subplots.png')

if __name__=="__main__":
    main()
