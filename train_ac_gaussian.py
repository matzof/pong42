"""Created by Matzof on Tue Nov 26 16:37:18 2019"""
import gym

import wimblepong
from AI42_ac_gaussian import AI42
from wimblepong.fast_ai import FastAi
from utils import plot_rewards, extract_state_cheating
import torch
from keras.models import load_model
# %%
env = gym.make("WimblepongVisualMultiplayer-v0")
# %%
# Parameters
render = True
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

model = load_model('00_baseline.h5')
(ob1, ob2), (rew1, rew2), done, info = env.step((2, 2))
win1 = 0
length_history = []

for ep in range(episodes):
    done = False
    length_ep = 0

    # Reset the environment and observe the initial state
    ob1 = env.reset()
    ob2 = env.reset()

    # Loop until the episode is over
    while not done:

        # Get the actions from both SimpleAIs
        action1, action_probabilities1 = player.get_action_cheating(ob1, model)
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        
        # Count the win for print
        if rew1 == 10:
            win1 += 1
        
        # TODO: adjust reward for training purpose
        rew1 += round(length_ep/30)
        
        # Store action's outcome (so that the agent can improve its policy)
        # TODO: Cheating -> comment/uncomment:
#        player.agent.store_transition(previous_state1, action_probabilities1, 
#                                      action1, rew1, model)
        player.agent.store_transition_cheating(env, action_probabilities1, 
                                      action1, rew1, player_id)
        
        # store total length of each episode
        length_ep += 1

        if render:
            env.render()


    # when done:

    # TODO: cheating -> comment/ uncomment:
    # state = extract_state(ob1, model)     
    state = extract_state_cheating(env, player_id) 
    player.agent.episode_finished(ep, state)

    length_history.append(length_ep)
    
    print("episode {} over. Length ep: {}. Mean Length: {:.1f}. Winrate: {:.3f}. Reward: {}".format(ep,
                length_ep, sum(length_history[len(length_history)-2000:])/2000, 
                win1 / (ep + 1), rew1))


    
    #plot_rewards(length_history)


























