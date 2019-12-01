import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from utils import Transition, ReplayMemory, extract_state, extract_state_cheating


class Policy(torch.nn.Module):
    def __init__(self, state_space = 4, action_space = 3, hidden = 128):
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
    def actor(self, state):
        return self.actor_layer(state)

    def critic(self, state):
        return self.critic_layer(state)


class Agent(object):
    def __init__(self, policy):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        self.policy_old = policy.to(self.train_device)
        self.policy_old.load_state_dict(self.policy.state_dict()) 
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3, betas=(0.9,0.999))
        self.gamma = 0.99
        self.eps_clip = 0.2  # TODO: Clip parameter for PPO
        self.K_epochs = 10 # TODO: Update policy for K epochs
        self.actions = []
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.MseLoss = nn.MSELoss()

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
            action_probs = self.policy.actor(old_states)
            action_distribution = Categorical(action_probs)
            # Caculate action log probability and entropy given old actions
            action_probs = action_distribution.log_prob(old_actions)
            dist_entropy = action_distribution.entropy()
            # Pass old states to  critic layers
            values = self.policy.critic(old_states)


            # Caculate the loss:
            # Finding the ratio (pi_theta / pi_theta__old) 
            ratios = torch.exp(action_probs - old_action_probs)
            
            # Finding Surrogate Loss:
            advantages = rewards - values.detach()
            surr1 =  ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step to update network parameters 
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states, self.action_probs, self.actions, self.rewards = [], [], [], []


    def get_action(self, state, evaluation=False):

        x = torch.from_numpy(state).float().to(self.train_device)
        # Pass state x through the actor network 
        action_probs = self.policy.actor(x)
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()
        act_log_prob = action_distribution.log_prob(action)

        return action, act_log_prob


    def store_transition(self, observation, action_prob, action_taken, reward, done, model):
        
        state = extract_state(observation, model)  
        state = torch.from_numpy(state).float().to(self.train_device)
        self.states.append(state)
        self.action_probs.append(action_prob)
        self.actions.append(action_taken)
        self.rewards.append(reward)
        self.dones.append(done)


    def store_transition_cheating(self, env, action_prob, action_taken, reward, done, player_id):
        
        state = extract_state_cheating(env, player_id)
        state = torch.from_numpy(state).float().to(self.train_device)
        self.states.append(state)
        self.action_probs.append(action_prob)
        self.actions.append(action_taken)
        self.rewards.append(reward)
        self.dones.append(done)

