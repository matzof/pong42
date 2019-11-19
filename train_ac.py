"""Created by Matzof on Fri Nov 15 16:22:49 2019"""
import gym

import wimblepong
from AI42_ac import AI42
from wimblepong.fast_ai import FastAi
from utils import plot_rewards
import torch
from keras.models import load_model

# %%
env = gym.make("WimblepongVisualMultiplayer-v0")
# %%
# Parameters
render = False
env.unwrapped.scale = 2
env.unwrapped.fps = 10000
episodes = 10000
glie_a = episodes / 20
num_episodes = 1000
TARGET_UPDATE = 20

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = FastAi(env, opponent_id)
player = AI42(env, player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

model = load_model('00_baseline.h5')
(ob1, ob2), (rew1, rew2), done, info = env.step((2, 2))
win1 = 0
length_history = []
for ep in range(episodes):
    done = False
    length_ep = 0
    while not done:
        # Get the actions from both SimpleAIs
        action1, action_probabilities1 = player.get_action(ob1, model)
        action2, _ = opponent.get_action(ob2)
        # Step the environment and get the rewards and new observations
        previous_state1 = ob1
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        # Store action's outcome (so that the agent can improve its policy)
        player.agent.store_transition(previous_state1, action_probabilities1, action1, rew1, model)
        # store total length of each episode
        length_ep += 1
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if render:
            env.render()
        if done:
            observation = env.reset()
            print("episode {} over. Length ep: {}. Broken WR: {:.3f}".format(ep,
                                                                             length_ep, win1 / (ep + 1)))
    length_history.append(length_ep)
    plot_rewards(length_history)

    # Update the target network, copying all weights and biases in DQN
    if ep % TARGET_UPDATE == 0:
        player.agent.update_target_network()
    # Save the policy
    if ep % 1000 == 0:
        torch.save(player.agent.policy_net.state_dict(),
                   "weights_%s_%d.mdl" % ('wimblepong', ep))













