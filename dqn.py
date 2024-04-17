import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=4, padding =1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding =1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(34560, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_actions)


    def forward(self, x):
        x.to(device)
        x = x.unsqueeze(0)
        # print("x shape:", x.shape)

        # block 1
        x = self.conv1(x)
        # print("conv1 shape:", x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        # print("block 1", x.shape)

        # block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print("block 2", x.shape)

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # print("block 3", x.shape)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x
    
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    

class Agent():
    def __init__(self, env, device):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.device = device

        # model setup
        self.model = DQN(self.num_actions).to(device)
        self.target_model = DQN(self.num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # buffer and optimizer setup 
        self.buffer = ReplayBuffer()
        self.alpha = 0.0001
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_minimum = 0.05

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                # print('select action state', state.size())
                state = state.unsqueeze(0)
                q_values = self.model(state)
                return torch.argmax(q_values)  # Action with the highest Q-value
            
    def train(self, batch_size, target_update_freq):
        # print("len_buffer:", len(self.buffer.buffer))
        # print("target_update_freq:", target_update_freq)
        if len(self.buffer.buffer) < target_update_freq:
            return np.inf, 0 
        
        samples = self.buffer.sample(batch_size)
        state, action, reward, next_state, done = [list(item) for item in zip(*samples)]

        # print("state", state)
        # print("State shape:", state[0].shape) 
        # Convert batches to tensors
        # print("state:", state)
        # print("action:", action)
        # print("reward:", reward)
        # print("next_state:", next_state)
        batch_state = torch.tensor(state, dtype=torch.float32, device=self.device)
        # batch_state = torch.tensor(state, dtype=torch.float32, device=self.device)
        batch_action = torch.tensor(action, device=self.device)
        batch_reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        batch_next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        batch_done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # print("model batch_state:", batch_state.size())
        # Calculate current Q-values
        q_values = self.model(batch_state)

        # print("target batch_state:", batch_state.size())
        # Calculate next Q-values from target model
        next_q_values = self.target_model(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)

        return loss.item(), torch.max(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_minimum
        
