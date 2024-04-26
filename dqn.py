import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_MLP(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_actions, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_actions, num_actions, bias=True)


    def forward(self, x):
        x.to(device)
        x = x.view(x.size(0), -1) # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x
        
class DQN_CONV(nn.Module):
    def __init__(self, num_actions):
        super(DQN_CONV, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=12, stride = 7)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(638, num_actions)


    def forward(self, x):
        x.to(device)
        x = x.unsqueeze(1)
        # print("x shape:", x.shape)

        # block 1
        x = self.conv1(x)
        # print("conv1 shape:", x.shape)

        x = x.reshape(x.size(0), -1)
        # print("input to fc1:", x.shape)
        x = self.fc1(x)
        # print("output fc1", x.shape)

        return x

class ReplayBuffer():
    def __init__(self, device, capacity=10000):
        self.buffer_state = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_action = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_reward = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_next_state = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_done = torch.empty((0,), dtype=torch.float32, device=device)
        self.capacity = capacity
        self.device = device

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer_state) >= self.capacity:
            # Remove the oldest data if at capacity
            self.buffer_state.pop(0)
            self.buffer_action.pop(0)
            self.buffer_reward.pop(0)
            self.buffer_next_state.pop(0)
            self.buffer_done.pop(0)

        # Add new elements to the buffers
        self.buffer_state = torch.cat((self.buffer_state, torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)), dim=0)
        self.buffer_action = torch.cat((self.buffer_action, torch.tensor([action], dtype=torch.float32, device=device)), dim=0)
        self.buffer_reward = torch.cat((self.buffer_reward, torch.tensor([reward], dtype=torch.float32, device=device)), dim=0)
        self.buffer_next_state = torch.cat((self.buffer_next_state, torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)), dim=0)
        self.buffer_done = torch.cat((self.buffer_done, torch.tensor([done], dtype=torch.float32, device=device)), dim=0)

    def sample(self, batch_size):
        # Make sure we do not sample more elements than we have
        max_index = self.buffer_state.size()[0]
        indices = torch.randint(0, max_index, (batch_size,), device=self.device)

        # Retrieve samples by indexed selection
        states = torch.stack([self.buffer_state[i] for i in indices])
        actions = torch.stack([self.buffer_action[i] for i in indices])
        rewards = torch.stack([self.buffer_reward[i] for i in indices])
        next_states = torch.stack([self.buffer_next_state[i] for i in indices])
        dones = torch.stack([self.buffer_done[i] for i in indices])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.buffer_state.size()[0]
    

class Agent():
    def __init__(self, env, model_name, device):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.num_observations = self.env.observation_space.shape
        self.device = device

        # model setup
        if model_name == "DQN_MLP":
            self.model = DQN_MLP(self.num_observations[0]*self.num_observations[1], self.num_actions).to(device)
            self.target_model = DQN_MLP(self.num_observations[0]*self.num_observations[1], self.num_actions).to(device)
        elif model_name == 'DQN_CONV':
            self.model = DQN_CONV(self.num_actions).to(device)
            self.target_model = DQN_CONV(self.num_actions).to(device)
        # elif model_name = 'DQN_CONVLSTM':
        #     self.model = DQN_CONVLSTM(self.num_actions).to(device)
        #     self.target_model = DQN_CONVLSTM(self.num_actions).to(device)

        self.target_model.load_state_dict(self.model.state_dict())

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total number of parameters: ", total_params)

        # buffer and optimizer setup 
        self.buffer = ReplayBuffer(device)
        self.alpha = 0.0001
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
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
        if self.buffer.__len__() < target_update_freq:
            return np.inf, 0 
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample(batch_size)
#         print(f'sample device: {batch_state.device} with size {batch_state.size()}')

        # print("batch_state", state)
#         print("State shape:", batch_state.shape) 
#         print("action shape:", batch_action.shape) 
#         print("reward shape:", batch_reward.shape) 

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
        
