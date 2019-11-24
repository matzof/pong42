import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from utils import Transition, ReplayMemory, extract_state, extract_state_cheating
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space = 4, action_space = 3, hidden = 64):
        super().__init__()

        self.actor_layer = nn.Sequential(
            nn.Linear(state_space, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_space),
            nn.Softmax(dim=-1)
        )

        self.critic_layer = nn.Sequential(
            nn.Linear(state_space, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        ) 


class Agent(object):
    def __init__(self, policy):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3, betas=(0.9,0.999))
        self.gamma = 0.98
        self.eps_clip = 0.2  # TODO: Clip parameter for PPO
        self.K_epochs = 4 # TODO: Update policy for K epochs
        self.actions = []
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.MseLoss = nn.MSELoss()

    # Monte Carlo estimate of state rewards
    def discount_rewards(self, dones):
        discounted_rewards = np.zero_like(self.rewards)
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
        rewards = discount_rewards(self)
        rewards = (torch.sub(rewards, rewards.mean())).div(rewards.var() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        old_action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        old_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)

        for _ in range(self.K_epochs):
            # Evaluate old actions and values: 
            # Pass old states to actor layers
            action_probs = Policy.actor_layer(old_states)
            action_distribution = Categorical(action_probs)
            # Caculate action log probability and entropy given old actions
            action_probs = action_distribution.log_prob(old_actions)
            dist_entropy = action_distribution.entropy()
            # Pass old states to  critic layers
            values = Policy.critic_layer(old_states)


            # Caculate the loss:
            # Finding the ratio (pi_theta / pi_theta__old) 
            ratios = torch.exp(action_probs - old_action_probs.detach())
            # Finding Surrogate Loss:
            advantages = rewards - values.detach()
            surr1 =  ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy

            # Take gradient step to update network parameters 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.states, self.action_probs, self.actions, self.rewards = [], [], [], []


    def get_action(self, state, evaluation=False):

        x = torch.from_numpy(state).float().to(self.train_device)

        # Pass state x through the actor network 
        action_probs = Policy.actor_layer(x)
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()
        act_log_prob = action_distribution.log_prob(action)

        return action, act_log_prob


    def store_transition(self, observation, action_prob, action_taken, reward, done, model):
        
        state = extract_state(observation, model)        
        self.states.append(state)
        self.action_probs.append(action_prob)
        self.actions.append(action_taken)
        self.rewards.append(reward)
        self.dones.append(done)


    def store_transition_cheating(self, env, action_prob, action_taken, reward, done, player_id):
        
        state = extract_state_cheating(env, player_id)
        self.states.append(state)
        self.action_probs.append(action_prob)
        self.actions.append(action_taken)
        self.rewards.append(reward)

