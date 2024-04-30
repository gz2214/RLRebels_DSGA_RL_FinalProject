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

def prepro(I, flatten=False):
    """
    Preprocesses a 210x160 grayscale frame into an 80x80 2D array or a 6400-element 1D float vector, based on the `flatten` flag.
    
    Args:
        I (numpy array): The input grayscale frame of size 210x160.
        flatten (bool): If True, the output is flattened into a 1D vector. If False, the output remains a 2D array.
        
    Returns:
        numpy array: The processed frame as a 2D array or 1D vector.
    """
    # Crop the image to remove the top and bottom parts that might not contain useful information
    I = I[:, 35:195, :]  # Crop vertically from 210 to 160, keeping horizontal dimension
    
    # Downsample by a factor of 2
    # Since you want to get from [160, 160] to [80, 80], downsample each dimension by factor of 2
    I = I[:, ::2, ::2]  # downsample to 80x80

    # Erase background by setting specific color values to 0
    I[I == 144] = 0  # erase background type 1
    I[I == 109] = 0  # erase background type 2

    # Set all other non-zero values to 1 (paddles, ball)
    I[I != 0] = 1

    # Convert to float and optionally flatten
    I = I.to(dtype=torch.float32)
    # print("I shape:", I.shape)
    if flatten:
        return I.reshape(I.size(0), -1)  # Flatten each frame in the batch
    else:
        return I  # Return the 2D tensor if flatten is False

class DQN_MLP(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_actions, bias=True)


    def forward(self, x):
        x = prepro(x, flatten=True)
        # print("I shape flatten:", x.shape)
        x.to(device)
        x = x.view(x.size(0), -1) # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x
        
class DQN_CONV(nn.Module):
    def __init__(self, num_actions):
        super(DQN_CONV, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=12, stride = 9)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(64, num_actions)


    def forward(self, x):
        x = prepro(x, flatten=False)
        x.to(device)
        x = x.unsqueeze(1)
#         print("x shape:", x.shape)

        # block 1
        x = self.conv1(x)
        # print("conv1 shape:", x.shape)

        x = x.reshape(x.size(0), -1)
        # print("input to fc1:", x.shape)
        x = self.fc1(x)
        # print("output fc1", x.shape)

        return x

class ReplayBuffer():
    """
    A simple FIFO (first in, first out) buffer for storing experiences.

    Attributes:
        buffer_state (torch.Tensor): Buffer for storing states.
        buffer_action (torch.Tensor): Buffer for storing actions.
        buffer_reward (torch.Tensor): Buffer for storing rewards.
        buffer_next_state (torch.Tensor): Buffer for storing next states.
        buffer_done (torch.Tensor): Buffer for storing done flags.
        capacity (int): Maximum number of transitions the buffer can hold.
        device (torch.device): Device on which the buffer will be stored.

    Args:
        device (torch.device): The computation device, CPU or GPU.
        capacity (int): The maximum size of the buffer.
    """
    def __init__(self, device, capacity=10000):
        self.buffer_state = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_action = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_reward = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_next_state = torch.empty((0,), dtype=torch.float32, device=device)
        self.buffer_done = torch.empty((0,), dtype=torch.float32, device=device)
        self.capacity = capacity
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """
        Adds a single transition to the buffer. If the buffer is full, it removes the oldest transition to make space.

        Args:
            state (array-like or torch.Tensor): The state observed before taking the action.
            action (int or torch.Tensor): The action taken in the state.
            reward (float or torch.Tensor): The reward received after taking the action.
            next_state (array-like or torch.Tensor): The next state reached after taking the action.
            done (bool or torch.Tensor): A boolean indicating whether the episode ended after this transition.

        Note:
            This method does not return anything. It updates the internal state of the buffer to include the new transition.
        """
        if len(self.buffer_state) >= self.capacity:
            # Remove the oldest data if at capacity
            self.buffer_state = self.buffer_state[1:]
            self.buffer_action = self.buffer_action[1:]
            self.buffer_reward = self.buffer_reward[1:]
            self.buffer_next_state = self.buffer_next_state[1:]
            self.buffer_done = self.buffer_done[1:]

        # Add new elements to the buffers
        self.buffer_state = torch.cat((self.buffer_state, torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)), dim=0)
        self.buffer_action = torch.cat((self.buffer_action, torch.tensor([action], dtype=torch.float32, device=device)), dim=0)
        self.buffer_reward = torch.cat((self.buffer_reward, torch.tensor([reward], dtype=torch.float32, device=device)), dim=0)
        self.buffer_next_state = torch.cat((self.buffer_next_state, torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)), dim=0)
        self.buffer_done = torch.cat((self.buffer_done, torch.tensor([done], dtype=torch.float32, device=device)), dim=0)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer uniformly at random.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple of torch.Tensors: A tuple containing batches of states, actions, rewards, next_states, and dones.
            Each element of the tuple is a tensor containing `batch_size` elements corresponding to the sampled transitions.
        """
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
    """
    Implements a Deep Q-Network (DQN) agent for reinforcement learning.

    Attributes:
        env (gym.Env): An instance of an OpenAI Gym environment.
        num_actions (int): Number of possible actions in the environment's action space.
        num_observations (tuple): Shape of the observation space from the environment.
        lives (int): Number of lives the agent has, used for games with lives like Breakout.
        device (torch.device): The device (CPU or GPU) on which tensors will be allocated.
        model (torch.nn.Module): The current DQN model.
        target_model (torch.nn.Module): The target DQN model for stable learning.
        buffer (ReplayBuffer): Buffer to store transitions for experience replay.
        alpha (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        optimizer (torch.optim.Optimizer): Optimizer for learning the model's weights.
        loss_fn (torch.nn.modules.loss): Loss function used for training.
        epsilon (float): Epsilon value for epsilon-greedy action selection.
        epsilon_decay (float): Decay rate for epsilon, reducing as training progresses.
        epsilon_minimum (float): Minimum value that epsilon can reach.

    Args:
        env (gym.Env): The gym environment to interact with.
        model_name (str): Identifier for selecting the model type.
        device (torch.device): The computation device, CPU or GPU.
        rendering (bool): Flag to indicate if the agent is for rendering (evaluation) or training.
    """
    def __init__(self, env, model_name, device, rendering=False):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.num_observations = self.env.observation_space.shape
        self.lives = self.env.unwrapped.ale.lives()
        self.device = device

        # model setup
        if model_name == "DQN_MLP":
            self.model = DQN_MLP(6400, self.num_actions).to(device)
            self.target_model = DQN_MLP(6400, self.num_actions).to(device)
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
        self.epsilon = 1.0 if not rendering else 0.0 # no need for epsilon greedy when rendering game play
        self.epsilon_decay = 0.98
        self.epsilon_minimum = 0.05

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def select_action(self, state, info = None):
        """
        Selects an action based on the current state using an epsilon-greedy policy.
        
        Args:
            state (array-like): The current state representation from the environment.
            info (dict, optional): Additional information about the current state (e.g., remaining lives).
        
        Returns:
            int: The action to be taken.
        """
        print('this is self.lives:',self.lives)
        if info == None:
            return 1 # force fire action
        elif info['lives'] < self.lives:
            self.lives = info['lives']
            return 1 # force fire action
        
        elif random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action

#         if random.random() < self.epsilon:
#             return self.env.action_space.sample()  # Random action
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                state = state.unsqueeze(0)
#                 print('select action state', state.size())
                q_values = self.model(state)
                return torch.argmax(q_values).item()  # Action with the highest Q-value
            
    def train(self, batch_size, target_update_freq):
        """
        Trains the agent using batches of experience from the replay buffer.

        Args:
            batch_size (int): The size of the batch to train on.
            target_update_freq (int): Frequency (in steps) at which the target network is updated.

        Returns:
            tuple: A tuple containing the loss of the training step and the maximum Q value.
        """
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample(batch_size)

        # Calculate current Q-values
#         print('batch_states', batch_state.shape)
        cur_q_values = self.model(batch_state).gather(1, batch_action.unsqueeze(1).long()).squeeze(1)

        # print("target batch_state:", batch_state.size())
        # Calculate next Q-values from target model
        next_q_values = self.target_model(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        loss = self.loss_fn(cur_q_values, expected_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)

        return loss.item(), torch.max(cur_q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.

        Args:
            state (array-like): The state before the action was taken.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (array-like): The state after the action was taken.
            done (bool): A boolean flag indicating if the episode ended after the action.
        """
        self.buffer.add(state, action, reward, next_state, done)
        
    def update_epsilon(self):
        """
        Updates the epsilon value used for the epsilon-greedy policy based on the decay rate.
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_minimum
        
