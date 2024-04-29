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

pretrained_model = torch.load("presentation/pretrained_model_ALE_Breakout-v5_DQN_CONV_400.pth", map_location=torch.device('cpu'))

env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode="human")

state = env.reset()[0]

agent = Agent(env, "DQN_CONV", device, rendering = False)

agent.model = pretrained_model
agent.epsilon = 0.0
info = None
steps = 0

for counter in range(500):

    env.render()
    action = agent.select_action(state, info)
    next_state, reward, done, truncated, info = env.step(action)
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
