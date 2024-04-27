import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import numpy as np 
import dqn
import matplotlib.pyplot as plt
import time
import sys

def main(pretrained_model_name,atari_game_2,num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'gpu count: {torch.cuda.device_count()}')
## set up environment
    env = gym.make(atari_game_2, obs_type="grayscale")
    atari_game_2 = atari_game_2.replace('/', '_')
    num_actions = env.action_space.n
    num_observations = env.observation_space.shape[0] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## freeze other layers parameters so they do not change
    pre_trained_model=torch.load(f'{pretrained_model_name}.pth')
    in_feat=pre_trained_model.output_shape[-1]
    for p in pre_trained_model.parameters():
        p.requires_grad=False
## add layer to pretrain model 
    pre_trained_model.breakout_output_layer=nn.Linear(in_features=in_feat, out_features=num_actions)
## add model to agent
    agent = dqn.Agent(env, device)
    agent.model=pre_trained_model

    num_episodes = num_episodes
    batch_size = 1
    target_update_freq = 100

    min_loss = np.inf
    episode_rewards=[]
    episode_loss=[]
    start = time.time()

    total_steps = 0

    for episode in range(num_episodes):
        #print("episode:", episode)
        state = env.reset()[0]
        start = time.time()
        # state = preProcess(state)  # Preprocess the initial state
        done = False
        episode_reward = 0
        loss=[]
        step_per_ep = 0

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
            
            cur_loss, _ = agent.train(batch_size, target_update_freq)

            loss.append(cur_loss)
            
            if total_steps % 10000 == 0:
                agent.update_target_model
        
        loss=[l for l in loss if l !=float('inf')]
        mean_loss=sum(loss)/len(loss)
        
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
        
        np.save(f'episode_rewards_{atari_game_2}_transferLearning_from_{pretrained_model_name}.npy', np.array(episode_rewards))
        np.save(f'episode_loss_{atari_game_2}_transferLearning_from_{pretrained_model_name}.npy', np.array(episode_loss))
        
    print('training process done!')
    
    i = 1
# Initialize an empty list to store cumulative moving averages
    moving_reward_averages = []
# Store cumulative sums of array in cum_sum array
    cum_sum_reward=np.cumsum(episode_rewards)
# Loop through the array elements
    while i <= len(episode_reward):
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

    plt.savefig(f'transfer_model_subplots_{atari_game_2}.png')

    plt.close()

    plt.plot(range(num_episodes),episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (3 Episode Average)')
    plt.savefig(f'transfer_model_rewards_{atari_game_2}.png')

    
if __name__=="__main__":

    # Parse command-line arguments
    pretrained_model=sys.argv[0]
    atari_game_2 = sys.argv[1]
    num_episodes = int(sys.argv[2])
    main(pretrained_model,atari_game_2, num_episodes)

