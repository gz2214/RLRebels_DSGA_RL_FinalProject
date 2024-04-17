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

    for episode in range(num_episodes):
        print("episode:", episode)
        state = env.reset()[0]
        # state = preProcess(state)  # Preprocess the initial state
        done = False
        episode_reward = 0
        
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
                torch.save(agent.model.state_dict(), 'cache_model.pth')
        
        if episode % 2 == 0:
            torch.save(agent.model, 'pretrained_model.pth')



        print(f"Episode {episode + 1}, Reward: {episode_reward}")

if __name__=="__main__":
    main()