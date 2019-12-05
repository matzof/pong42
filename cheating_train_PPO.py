import gym

from working_AI42_PPO import Agent42
from wimblepong.simple_ai import SimpleAi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%%
env = gym.make("WimblepongVisualMultiplayer-v0")
#%%
# Parameters
render = False
num_iterations = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SimpleAi(env, opponent_id)
player = Agent42(env, player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

(ob1, ob2), (rew1, rew2), done, info = env.step((2, 2))
win1 = 0
culmulative_reward = 0
length_history = []
win_history = []
mean_winrate_history = []
mean_length_history = []

for it in range(num_iterations):
    for ep in range(200):
        done = False
        length_ep = 0
    
        # Reset the environment and observe the initial state
        ob1, ob2 = env.reset()
    
        # Loop until the episode is over
        while not done:
    
            # Get the actions from both AIs
            action1 = player.get_action(ob1)
            action2 = opponent.get_action()
    
            # Step the environment and get the rewards and new observations
            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
    
            # Store the results (reward and done) of the step performed
            player.store_result(rew1, done)
    
            # store total length of each episode
            length_ep += 1
    
            # Count the wins
            if render:
                env.render()                   
    
        # when done:
        win_history.append(1 if rew1==10 else 0)
        length_history.append(length_ep)
        
        if len(win_history) == 500:
            length_history.pop(0)
            win_history.pop(0)
        
        print("Iter:", it, "Ep:", ep, "Length ep:", length_ep, 
              "Victory: {:.0f}".format((rew1+10)/20),
              "Mean Length: {:.1f}".format(sum(length_history)/len(length_history)),
              "Winrate: {:.1f}%".format(100*sum(win_history)/len(win_history)))
        length_ep = 0
        
        if ep % 20 == 0:
            # Plot Mean Reward History
            mean_winrate_history.append(100*sum(win_history)/len(win_history))
            mean_length_history.append(0.333*sum(length_history)/len(length_history))
            plt.figure(figsize=(20.0, 10.0))
            plt.xlabel("Number of Iterations*10", fontsize=30)
            plt.plot(mean_winrate_history)
            plt.plot(mean_length_history)
            plt.legend(["Mean Winrate", "Mean Length of Episodes (divided by 3)"], fontsize ='xx-large')
            plt.grid()
            plt.savefig("training_performance_plot.png")
            plt.close()
    
#    # Saving Model
#    if it % 300 == 0:
#        print("Saving -----------------------------------------------")
#        player.store_model(player.policy)
#        player.policy = player.load_model(player.policy)
    
    # PPO Update
    print("Updating ---------------------------------------------")
    player.PPO_update() 
    
    











