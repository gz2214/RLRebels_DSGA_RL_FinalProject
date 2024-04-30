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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = sys.argv[1]
pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))

env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode="human")

state = env.reset()[0]

if 'DQN_CONV' in model_path:
    agent = Agent(env, 'DQN_CONV', device=device, rendering=True)
if 'DQN_MLP' in model_path:
    agent = Agent(env, 'DQN_MLP', device=device, rendering=True)

agent.model = pretrained_model
info = None
steps = 0

for counter in range(500):
    """
    Function to test a pre-trained DQN model on the Breakout and Pong Atari environments.

    This function initializes a pre-trained DQN model (either on Breakout or Atari). It runs the agent using actions determined by
    the pre-trained model for a fixed number of steps (500 in this case) to evaluate its performance.

    Returns:
        None
    """

    env.render()
    action = agent.select_action(state, info)
    next_state, reward, done, truncated, info = env.step(action)
    print(info)
    steps += 1
    print(action)
    # print(info)
    # agent.store_transition(state, action, reward, next_state, done)
    state = next_state

    if done:
        print(f"Test episode done, took {steps} steps.")
        observation, info = env.reset()   
        break
        
env.close()
