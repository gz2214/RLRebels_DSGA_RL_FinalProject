import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import numpy as np 
import dqn
import matplotlib.pyplot as plt
import time
import sys

def main(pretrained_model_path,atari_game_2,num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'gpu count: {torch.cuda.device_count()}')
## set up environment
    env = gym.make(atari_game_2, obs_type="grayscale")
    atari_game_2 = atari_game_2.replace('/', '_')
    num_actions = env.action_space.n
    num_observations = env.observation_space.shape[0] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## add layer to pretrain model 
    agent = dqn.Agent(env, device)
    agent.model=torch.load(pretrained_model_path)
    agent.model.breakout_output_layer=nn.Linear(in_features=512, out_features=num_actions)

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
            torch.save(agent.model, f'transfered_pretrained_model_{atari_game_2}.pth')
        
        if episode % target_update_freq == 0:
            torch.save(agent.model.state_dict(), f'transfered_cache_model_{atari_game_2}.pth')
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
    plt.savefig(f'transfer_model_subplots_{atari_game_2}.png')

    
if __name__=="__main__":

    # Parse command-line arguments
    pretrained_model=sys.argv[0]
    atari_game_2 = sys.argv[1]
    num_episodes = int(sys.argv[2])
    main(pretrained_model,atari_game_2, num_episodes)

