"""Created by Matzof on Fri Nov 15 16:22:49 2019"""
import gym

import wimblepong
from AI42_ac import AI42
from wimblepong.fast_ai import FastAi
from utils import plot_rewards, extract_state_cheating
import torch
from keras.models import load_model
# %%
env = gym.make("WimblepongVisualMultiplayer-v0")
# %%
# Parameters
render = False
episodes = 1000000
glie_a = episodes / 20
TARGET_UPDATE = 20

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = FastAi(env, opponent_id)
player = AI42(env, player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

model = 1# load_model('00_baseline.h5')
(ob1, ob2), (rew1, rew2), done, info = env.step((2, 2))
win1 = 0
length_history = []
for ep in range(episodes):
    done = False
    length_ep = 0
    while not done:
        # Get the actions from both SimpleAIs
        action1, action_probabilities1 = player.get_action_cheating(ob1, model)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        # adjust reward for training purpose
        if rew1 == 10:
            win1 += 1
#        rew1 += round(length_ep/40)
        
        # Store action's outcome (so that the agent can improve its policy)
#        player.agent.store_transition(previous_state1, action_probabilities1, 
#                                      action1, rew1, model)
        player.agent.store_transition_cheating(env, action_probabilities1, 
                                      action1, rew1, player_id)
        # store total length of each episode
        length_ep += 1
        # Count the wins
        if render:
            env.render()
        if done:
            length_history.append(length_ep)
            observation = env.reset()
            print("episode {} over. Length ep: {}. Mean Length: {:.1f}. Winrate: {:.3f}. Reward: {}".format(ep,
                       length_ep, sum(length_history[len(length_history)-1000:])/1000, 
                        win1 / (ep + 1), rew1))
            
            state = extract_state_cheating(env, player_id)
            player.agent.episode_finished(ep, state)
        
            
#    plot_rewards(length_history)









