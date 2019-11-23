import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = torch.zeros(1) + 5 # TODO: Implement accordingly (T1: 5, T2: update in cartpole.py)
        self.sigma = torch.nn.Parameter(torch.zeros(1) + 10)  # TODO: learn sigma as parameter (T2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = F.softplus(self.sigma)  # TODO use softplus to avoid errors with negative values of sigma

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        return Normal(mu, sigma)
        # TODO: Add a layer for state value calculation (T3)

class Agent(object):
    def __init__(self, policy, baseline=0):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.baseline = baseline

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_r = discount_rewards(rewards, self.gamma) - self.baseline
        discounted_r = (torch.sub(discounted_r, discounted_r.mean())).div(discounted_r.var()) # TASK 1C
        # TODO: Compute critic loss and advantages (T3)

        # TODO: Compute the optimization term (T1, T3)
        loss = (torch.sum(torch.mul(action_probs, Variable(discounted_r)).mul(-1), -1))
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        self.optimizer.zero_grad()
        loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        action_distribution = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = action_distribution.mean()
        else:
            action = action_distribution.sample()
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = action_distribution.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)
        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))



