"""Created by Matzof on Fri Nov 15 16:22:49 2019"""
import gym

#import wimblepong
from AI42_PPO_old import AI42
from wimblepong.simple_ai import SimpleAi
from utils import plot_rewards, extract_state_cheating
import torch
from keras.models import load_model
# %%
env = gym.make("WimblepongVisualMultiplayer-v0")
# %%
# Parameters
render = False
num_episodes = 1000000
TARGET_UPDATE = 20

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SimpleAi(env, opponent_id)
player = AI42(env, player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())


# TODO:not cheating -> uncomment: 
# model = load_model('00_baseline.h5')

(ob1, ob2), (rew1, rew2), done, info = env.step((2, 2))
win1 = 0
length_history = []
win_history = []
timesteps = 0

for ep in range(num_episodes):
    done = False
    length_ep = 0

    # Reset the environment and observe the initial state
    ob1 = env.reset()

    # Loop until the episode is over
    while not done:

        # Get the actions from both AIs

        # TODO: Cheating -> comment/uncomment:
        # action1, action_probabilities1 = player.get_action(ob1, model)
        action1, action_probabilities1 = player.get_action_cheating(ob1)
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        previous_state1 = ob1
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        # TODO: adjust reward for training purpose
#        rew1 += round(length_ep/30)
        
        # Store action's outcome (so that the agent can improve its policy)
        # TODO: Cheating -> comment/uncomment:
#       player.agent.store_transition(previous_state1, action_probabilities1, 
#                                      action1, rew1, done, model)
        player.agent.store_transition_cheating(env, action_probabilities1, 
                                      action1, rew1, done, player_id)
        
        # store total length of each episode
        length_ep += 1
        timesteps += 1

        # Count the wins
        if render:
            env.render()        
        
        # PPO Update    
        if timesteps % 4000 == 0:
            print("Updating ------------------------------")
            # TODO: cheating -> comment/ uncomment:
            # state =  state = extract_state(env, model)
            state = extract_state_cheating(env, player_id)
            player.agent.PPO_update()    


    # when done:
    win_history.append(1 if rew1==10 else 0)
    length_history.append(length_ep)
    
    if len(win_history) == 1000:
        length_history.pop(0)
        win_history.pop(0)
        
    print("episode {} over. Length ep: {}. Mean Length: {:.1f}. Winrate: {:.3f}. Reward: {}".format(ep,
                length_ep, sum(length_history)/len(length_history), 
                sum(win_history)/len(win_history), rew1))
    length_ep = 0

    # plot_rewards(length_history)












