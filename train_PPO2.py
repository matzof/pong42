import gym

from AI42_PPO2 import Agent42
from wimblepong.simple_ai import SimpleAi

env = gym.make("WimblepongVisualMultiplayer-v0")

# Parameters
render = False
num_episodes = 1000000
horizon = 2000 # TODO: adapt the PPO Update frequency

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SimpleAi(env, opponent_id)
player = Agent42(env, player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

(ob1, ob2), (rew1, rew2), done, info = env.step((2, 2))
timesteps = 0
win1 = 0
culmulative_reward = 0
length_history = []
win_history = []


for ep in range(num_episodes):
    done = False
    length_ep = 0

    # Reset the environment and observe the initial state
    ob1, ob2 = env.reset()

    # Loop until the episode is over
    while not done:

        # Get the actions from both AIs

        # TODO: Cheating -> comment/uncomment:
        # action1 = player.get_action(ob1, model)
        action1 = player.get_action(ob1)
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        # TODO: adjust reward for training purpose
        # rew1 += round(length_ep/30)
        player.store_result(rew1, done)
        culmulative_reward += rew1

        # store total length of each episode
        length_ep += 1
        timesteps += 1

        # Count the wins
        if render:
            env.render()        
        
        # PPO Update
        if timesteps % horizon == 0:
            print("Updating ------------------------------")
            player.PPO_update() 
            timestep = 0
       


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












