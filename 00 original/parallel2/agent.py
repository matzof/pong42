import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 64)
        self.fc2_mean = torch.nn.Linear(64, action_space)
        self.fc2_value = torch.nn.Linear(64, 1)
        self.sigma = torch.nn.Parameter(torch.ones(1)*10, requires_grad=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)#, -1e-3, 1e-3)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        x_mean = self.fc2_mean(x)
        sigma = F.softplus(self.sigma)
        dist = Normal(x_mean, sigma)

        # Critic part
        value = self.fc2_value(x)

        return dist, value


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-3)
        self.batch_size = 1
        self.gamma = 0.98
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.done = []

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        all_next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        all_done = torch.Tensor(self.done).to(self.train_device)
        self.states, self.next_states, self.actions, self.rewards, self.done = [], [], [], [], []


        # Compute state value estimates
        _, old_state_values = self.policy(all_states)
        _, next_state_values = self.policy(all_next_states)

        # Zero out values of terminal states
        next_state_values = next_state_values.squeeze(-1) * (1-all_done)

        # Detach, squeeze, etc.
        next_state_values = next_state_values.detach()
        old_state_values = old_state_values.squeeze(-1)

        # Estimate of state value and critic loss
        updated_state_values = all_rewards + self.gamma*next_state_values
        critic_loss = F.mse_loss(old_state_values, updated_state_values.detach())

        # Advantage estimates
        advantages = updated_state_values - old_state_values
        #advantages = advantages - torch.mean(advantages)
        #advantages = advantages / torch.std(advantages)

        # Weighted log probs and actor loss
        weighted_probs = all_actions * advantages.detach()
        actor_loss = -torch.mean(weighted_probs)

        # Total loss
        loss = actor_loss + critic_loss

        # Compute gradients
        loss.backward()

        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

        return critic_loss.item()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        dist, _ = self.policy.forward(x)
        if evaluation:
            action = dist.mean
        else:
            action = dist.sample()
        aprob = dist.log_prob(action)
        return action, aprob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.actions.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
