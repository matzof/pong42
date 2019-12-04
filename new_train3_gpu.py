import gym
import numpy as np
import random
import torch
from torch import nn
from new_model3 import Agent42
from new_model3 import Policy
from wimblepong import wimblepong
from wimblepong.simple_ai import SimpleAi

env = gym.make("WimblepongVisualMultiplayer-v0")

env.reset()

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SimpleAi(env, opponent_id)
player = Agent42(env, player_id)
policy = Policy()

num_episodes = 1000000
K_epochs = 5
horizon = 200
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

reward_sum_running_avg = None

for it in range(num_episodes):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for ep in range(horizon):
        (ob1, ob2), prev_obs = env.reset(), None
        for t in range(190000):
            #env.render()

            d_obs = player.pre_process_cnn(ob1, prev_obs)
            with torch.no_grad():
                action, action_prob = player.get_action(d_obs)
            
            prev_obs = ob1
            action1 = action
            action2 = opponent.get_action()
            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
            
            d_obs_history.append(d_obs)
            action_history.append(action)
            action_prob_history.append(action_prob)
            reward_history.append(rew1)

            if done:
                reward_sum = sum(reward_history[-t:])
                reward_sum_running_avg = 0.99*reward_sum_running_avg + 0.01*reward_sum if reward_sum_running_avg else reward_sum
                print('Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                #print(action_history[-5:])
                break
    
    # compute advantage
    R = 0
    discounted_rewards = []

    for r in reward_history[::-1]:
        if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum
        R = r + player.gamma * R
        discounted_rewards.insert(0, R)

    #print(discounted_rewards[:5])

    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
    
    # update policy
    for _ in range(K_epochs):
        n_batch = round(len(action_history)*0.7)
        idxs = random.sample(range(len(action_history)), n_batch)

        d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0).to(player.train_device).detach()    
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs]).to(player.train_device).detach()
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs]).to(player.train_device).detach()
        rewards_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs]).to(player.train_device).detach()

        opt.zero_grad()
        loss = player.PPO_update(d_obs_batch, action_batch, action_prob_batch, rewards_batch)
        loss.backward()
        opt.step()

        print('V3200 -- Iteration %d -- Loss: %.3f' % (it, loss))

env.close()

















