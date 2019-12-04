# PPO5: use one single forward fucntion, one less hidden fc than PPO4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random

class Policy(torch.nn.Module):
    def __init__(self, action_space = 2, hidden = 512):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
#        self.conv1 = torch.nn.Conv2d(2, 32, 8, 4)
#        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
#        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
#        self.reshaped_size = 64*9*9
        self.input_size = 20000
        self.fc1 = nn.Linear(self.input_size, hidden)
        self.fc2_action = nn.Linear(hidden, action_space)
        self.fc2_value = nn.Linear(hidden, 1)
        
    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = F.relu(self.conv3(x))
#        x = x.reshape(-1, self.reshaped_size)
        x = F.relu(self.fc1(x))

        action_probs = F.softmax(self.fc2_action(x), -1)
        values = self.fc2_value(x)

        return action_probs, values

    
class Agent42(object):
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id # Set the player id that determines on which side the ai is going to play                        
        self.name = "AI42"
        self.train_device = torch.device("cpu")
        self.policy = Policy().to(self.train_device)
        self.prev_obs = None
        
        self.policy_old = self.policy.to(self.train_device)
        self.policy_old.load_state_dict(self.policy.state_dict()) 
        self.optimizer = torch.optim.Adam(self.policy.parameters(), 
                                          lr=1e-3, betas=(0.9,0.999))
        self.gamma = 0.99
        self.eps_clip = 0.2  # TODO: Clip parameter for PPO
        self.K_epochs = 5 # TODO: Update policy for K epochs
        self.actions = []
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.MseLoss = nn.MSELoss()

    def get_name(self):
        """ Interface function to retrieve the agents name """
        return self.name

    def reset(self):
        self.prev_obs = None

    # Monte Carlo estimate of state rewards
    def discount_rewards(self):
        discounted_rewards = []
        running_add = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                running_add = 0
            running_add = running_add * self.gamma + reward
            discounted_rewards.insert(0, running_add)
        return discounted_rewards

    # Update the actor-critic
    def PPO_update(self):
        # Compute and normalize discounted rewards (use the discount_rewards function)
        rewards = np.asarray(self.discount_rewards())
        rewards = (rewards - np.mean(rewards))/np.var(rewards) + 1e-5
        
        for _ in range(self.K_epochs):
            # sample a random 50% of the data stored in every epoch
            len_history = len(self.actions)
            n_batch = round(len_history*0.7)
            idxs = random.sample(range(len_history), n_batch)
            
            batch_rewards = torch.tensor([rewards[idx] for idx in idxs]).to(self.train_device)
            batch_states = [self.states[idx] for idx in idxs]
            batch_action_probs = [self.action_probs[idx] for idx in idxs]
            batch_actions = [self.actions[idx] for idx in idxs]
    
            # Convert list to tensor
            batch_states = torch.stack(batch_states, dim=0).to(self.train_device).detach()
            batch_action_probs = torch.stack(batch_action_probs, dim=0).to(self.train_device).detach()
            batch_actions = torch.stack(batch_actions, dim=0).to(self.train_device).detach()
            
            # Evaluate batch actions and values: 
            # Pass batch states to actor layers
            action_probs, _ = self.policy.forward(batch_states)
            action_distribution = Categorical(action_probs)
            # Caculate action log probability and entropy given batch actions
            action_probs = action_distribution.log_prob(batch_actions)
            dist_entropy = action_distribution.entropy()
            # Pass batch states to  critic layers
            _, values = self.policy.forward(batch_states)

            # Caculate the loss:
            # Finding the ratio (pi_theta / pi_theta__batch) 
            ratios = torch.exp(action_probs - batch_action_probs)
            
            # Finding Surrogate Loss:
            advantages = batch_rewards - values.detach()
            surr1 =  ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = (-torch.min(surr1, surr2) 
                    + 0.5 * self.MseLoss(values.squeeze(1), batch_rewards) 
                    - 0.01 * dist_entropy)
            
            # Take gradient step to update network parameters 
            self.optimizer.zero_grad()
            loss.mean().backward()
            print('Loss:', loss.mean())
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states, self.action_probs, self.actions, self.rewards, self.dones = [], [], [], [], []

    def get_action(self, observation):
        """ Interface function that returns the action that the agent 
        takes based on the observation """
        observation = self.preprocess_observation(observation)
        stack_ob = self.stack_obs(observation)
        # Pass state x through the actor network 
        action_probs, _ = self.policy.forward(stack_ob)
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()
        action_prob = action_distribution.log_prob(action)
        self.store_transition(stack_ob.squeeze(0), action_prob, action)
        self.prev_obs = observation
        return action + 1

    def preprocess_observation_conv(self, obs):
        obs = obs[::2, ::2].mean(axis=-1) # grayscale and downsample
        obs[obs < 50] = 0 # set background as 0
        obs[obs != 0] = 1 # set paddles and ball as 1
        obs = np.reshape(obs, (1, obs.shape[0], obs.shape[1]))
        return obs
    
    def preprocess_observation(self, obs):
        obs = obs[::2, ::2].mean(axis=-1) # grayscale and downsample
        obs[obs < 50] = 0 # set background as 0
        obs[obs != 0] = 1 # set paddles and ball as 1
        obs = np.reshape(obs.ravel(), (1, len(obs.ravel)))
    
    def stack_obs(self, obs):
        if self.prev_obs is None:
            self.prev_obs = obs
        stack_ob = np.concatenate((self.prev_obs, obs), axis=0)
        stack_ob = torch.from_numpy(stack_ob).float().to(self.train_device)
        return stack_ob
    
    def store_transition(self, state, action_prob, action):
        self.states.append(state)
        self.action_probs.append(action_prob)
        self.actions.append(action)
    
    def store_result(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
        if done == 1:
            self.reset()
    
    def load_model(self):
        weights = torch.load("model.mdl")
        self.policy.load_state_dict(weights, strict=False)





















