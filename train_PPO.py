"""Created by Matzof on Fri Nov 15 16:22:49 2019"""
import gym

#import wimblepong
from AI42_PPO import AI42
from wimblepong.fast_ai import FastAi
from utils import plot_rewards, extract_state_cheating
import torch
from keras.models import load_model
# %%
env = gym.make("WimblepongSimpleAI-v0")
# %%
# Parameters
render = True
episodes = 1000000
num_episodes = 1000
TARGET_UPDATE = 20

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = FastAi(env, opponent_id)
player = AI42(env, player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())


# TODO:not cheating -> uncomment: 
# model = load_model('00_baseline.h5')

observation, reward, done, info = env.step((2))
length_history = []
win_history = []

for ep in range(episodes):
    done = False
    length_ep = 0

    # Reset the environment and observe the initial state
    ob1 = env.reset()

    # Loop until the episode is over
    while not done:

        # Get the actions from both AIs

        # TODO: Cheating -> comment/uncomment:
        # action1, action_probabilities1 = player.get_action(ob1, model)
        action1, action_probabilities1 = player.get_action_cheating(observation, ep)

        # Step the environment and get the rewards and new observations
        previous_state1 = ob1
        observation, rew1, done, info = env.step((action1))
        
        # Store action's outcome (so that the agent can improve its policy)
        # TODO: Cheating -> comment/uncomment:
#       player.agent.store_transition(previous_state1, action_probabilities1, 
#                                      action1, rew1, done, model)
        player.agent.store_transition_cheating(observation, action_probabilities1, 
                                      action1, rew1, done)

        # store total length of each episode
        length_ep += 1

        # Count the wins
        if render:
            env.render()

        # PPO Update    
        if length_ep % 100 == 0 or done:
            # TODO: cheating -> comment/ uncomment:
            # state =  state = extract_state(env, model)
#            state = extract_state_cheating(env, player_id)
            player.agent.PPO_update()            


    if ep > 1000:   
        length_history.pop(0)
        win_history.pop(0)
    length_history.append(length_ep)
    win_history.append(1) if rew1 == 10 else win_history.append(0)
        
        
    print("episode {} over. Length ep: {}. Mean Length: {:.1f}. Winrate: {:.3f}. Reward: {}".format(ep,
                length_ep, sum(length_history)/len(length_history), 
                sum(win_history) / len(win_history), rew1))

    # plot_rewards(length_history)









