"""Created by Matzof on Tue Dec  3 16:27:16 2019"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import random

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.eps_clip = 0.1

        self.layers = nn.Sequential(
            nn.Linear(20000, 512), nn.ReLU(),
            nn.Linear(512, 2),
        )
    
    def state_to_tensor(self, obs):
        obs = obs[::2, ::2].mean(axis=-1) # grayscale and downsample
        obs[obs < 50] = 0 # set background as 0
        obs[obs != 0] = 1 # set paddles and ball as 1
        obs = np.expand_dims(obs, axis=-1)        
        return torch.from_numpy(obs.astype(np.float32).ravel()).unsqueeze(0)

    def pre_process(self, obs, prev_obs):
        if prev_obs is None:
            prev_obs = obs
        obs = self.state_to_tensor(obs)
        prev_obs = self.state_to_tensor(prev_obs)
        return torch.cat([obs, prev_obs], dim=1)

    def convert_action(self, action):
        return action + 1

    def get_action(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
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
        vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])
        
        logits = self.layers(d_obs)
        ratios = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = ratios * advantage
        loss2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss

















