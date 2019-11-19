import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import Transition, ReplayMemory, extract_state



class Policy(torch.nn.Module):
    def __init__(self, state_space = 4, action_space = 3):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 128 #64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space)
        self.fc3 = torch.nn.Linear(self.hidden, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        policy_dist = Categorical(self.fc2(x))
        value = self.fc3(x)
        return value, policy_dist

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
        self.values = []

    def episode_finished(self, episode_number, observation):
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0) \
            .to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0) \
            .to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        # TODO: Compute critic loss and advantages (T3)
        # Always put the last next_state predicted value as 0 because the episode is over
        next_values = torch.cat((values[1:], torch.tensor([0.0]))).detach()
        advantages = (rewards + self.gamma * next_values) - values
        critic_loss = torch.mean(advantages**2)

        # TODO: Compute the optimization term (T1, T3)
        actor_loss = torch.mean(-action_probs * advantages.detach())

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss = critic_loss + actor_loss
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        value, action_distribution = self.policy.forward(x)
        action = action_distribution.sample()

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = action_distribution.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)
        self.values.append(torch.Tensor(value))
        return action, act_log_prob

    def store_transition(self, state, action_prob, action_taken, reward, model):
        state = extract_state(state, model)

        self.states.append(state)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

