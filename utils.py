import numpy as np
from collections import namedtuple
import torch
import matplotlib.pyplot as plt
import random



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) < 100:
        means = rewards_t.unfold(0, len(rewards_t), 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(len(rewards_t)-1), means))
        plt.plot(means.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def extract_state(ob, model):
    ob = np.mean(ob, -1)
    ob = np.reshape(ob, (1, ob.shape[0], ob.shape[1], 1))
    state = model.predict(ob)
    # state = [0, state[0, 0]-state[0, 1], state[0, 0]-state[0, 2], state[0, 3]]
    return np.asarray(state)

def extract_state_cheating(env, player_id):
    # Get the player id from the environmen
    player = env.player1 if player_id == 1 else env.player2
    opponent = env.player2 if player_id == 1 else env.player1
    state = [player.y, env.ball.x, env.ball.y, opponent.y]
    return np.asarray(state)
