# PPO3: use one single forward fucntion instead of distinguish as actor and critic forward as PPO3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Policy(torch.nn.Module):
    def __init__(self, action_space = 3, hidden = 64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(2, 32, 8, 4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.reshaped_size = 64*9*9
        self.fc1 = nn.Linear(self.reshaped_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3_action = nn.Linear(hidden, action_space)
        self.fc3_value = nn.Linear(hidden, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, self.reshaped_size)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))

        action_probs = F.softmax(self.fc3_action(x))
        values = self.fc3_value(x)

        return action_probs, values

    
class Agent42(object):
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id # Set the player id that determines on which side the ai is going to play                        
        self.name = "AI42"
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        rewards = self.discount_rewards()
        rewards = torch.tensor(rewards).to(self.train_device)
        rewards = (torch.sub(rewards, rewards.mean())).div(rewards.var() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(self.states, dim=0).to(self.train_device).detach()
        old_action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).detach()
        old_actions = torch.stack(self.actions, dim=0).to(self.train_device).detach()

        for _ in range(self.K_epochs):
            # Evaluate old actions and values: 
            # Pass old states to actor layers
            action_probs, _ = self.policy.forward(old_states)
            action_distribution = Categorical(action_probs)
            # Caculate action log probability and entropy given old actions
            action_probs = action_distribution.log_prob(old_actions)
            dist_entropy = action_distribution.entropy()
            # Pass old states to  critic layers
            _, values = self.policy.forward(old_states)

            # Caculate the loss:
            # Finding the ratio (pi_theta / pi_theta__old) 
            ratios = torch.exp(action_probs - old_action_probs)
            
            # Finding Surrogate Loss:
            advantages = rewards - values.detach()
            surr1 =  ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values.squeeze(1), rewards) - 0.01 * dist_entropy
            
            # Take gradient step to update network parameters 
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states, self.action_probs, self.actions, self.rewards, self.dones = [], [], [], [], []

    def get_action(self, observation):
        """ Interface function that returns the action that the agent 
        takes based on the observation """
        state = self.preprocess_observation(observation)
        # Pass state x through the actor network 
        action_probs, _ = self.policy.forward(state)
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()
        action_prob = action_distribution.log_prob(action)
        self.store_transition(state.squeeze(0), action_prob, action)
        return action

    def preprocess_observation(self, observation):
        observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = np.concatenate((self.prev_obs, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0).to(self.train_device)
        stack_ob = stack_ob.transpose(1, 3)
        return stack_ob
    
    def store_transition(self, state, action_prob, action):
        self.states.append(state)
        self.action_probs.append(action_prob)
        self.actions.append(action)
    
    def store_result(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def load_model(self):
        weights = torch.load("model.mdl")
        self.policy.load_state_dict(weights, strict=False)




















