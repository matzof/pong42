# + 3 CNN layers, no action convert -> action_space = 3

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import random

class Policy(nn.Module):
    def __init__(self, action_space = 3, hidden = 64):
        super().__init__()
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.eps_clip = 0.1

        self.conv1 = nn.Conv2d(2, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.reshaped_size = 64*9*9
        self.fc1 = nn.Linear(self.reshaped_size, hidden)
        self.fc2 = nn.Linear(hidden, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, self.reshaped_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def state_to_tensor_cnn(self, obs):
        obs = obs[::2, ::2].mean(axis=-1) # grayscale and downsample
        obs[obs < 50] = 0 # set background as 0
        obs[obs != 0] = 1 # set paddles and ball as 1  
        obs = np.reshape(obs, (1, obs.shape[0], obs.shape[1]))    
        return obs

    def pre_process_cnn(self, obs, prev_obs):
        if prev_obs is None:
            prev_obs = obs
        obs = self.state_to_tensor_cnn(obs)
        prev_obs = self.state_to_tensor_cnn(prev_obs)  
        stack_ob = np.concatenate((prev_obs, obs), axis=0)
        stack_ob = torch.from_numpy(stack_ob).unsqueeze(0).float()
        return stack_ob

        obs = obs[::2, ::2].mean(axis=-1) # grayscale and downsample
        obs[obs < 50] = 0 # set background as 0
        obs[obs != 0] = 1 # set paddles and ball as 1
        obs = np.reshape(obs, (1, obs.shape[0], obs.shape[1]))
        return obs


    def convert_action(self, action):
        return action + 1

    def get_action(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.forward(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob
        '''
        # policy gradient (REINFORCE)
        logits = self.layers(d_obs)
        loss = F.cross_entropy(logits, action, reduction='none') * advantage
        return loss.mean()
        '''
        
    def PPO_update(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        # PPO
        vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])
        
        logits = self.forward(d_obs)
        ratios = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = ratios * advantage
        loss2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss

















