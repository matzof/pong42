import gym

from AI42_PPO_6 import Agent42
from AI42_PPO_4 import Agent
from some_other_agent import Agent as SomeOtherAgent
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
opponent = Agent(env, opponent_id)
player = Agent42(env, player_id)
player.load_model("model_6.mdl")
opponent.load_model("model_4.mdl")

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
        
        if len(win_history) == 200:
            length_history.pop(0)
            win_history.pop(0)
        
        print("Iter:", it, "Ep:", ep, "Length ep:", length_ep, 
              "Victory: {:.0f}".format((rew1+10)/20),
              "Mean Length: {:.1f}".format(sum(length_history)/len(length_history)),
              "Winrate: {:.1f}%".format(100*sum(win_history)/len(win_history)))
        length_ep = 0
            
    # Saving Model
    if it % 1 == 0:
        print("Saving -----------------------------------------------")
        player.store_model(it % 20)

    # Plot Mean Reward History
    mean_winrate_history.append(100*sum(win_history)/len(win_history))
    mean_length_history.append(0.2*sum(length_history)/len(length_history))
    plt.figure(figsize=(20.0, 10.0))
    plt.xlabel("Number of Iterations", fontsize=30)
    plt.plot(mean_winrate_history)
    plt.plot(mean_length_history)
    plt.legend(["Mean Winrate", "Mean Length of Episodes (divided by 5)"], fontsize ='xx-large')
    plt.grid()
    plt.savefig("plots/" + str(it) + "_training_performance_plot.png")
    plt.close()
    
    # PPO Update
    print("Updating ---------------------------------------------")
    player.PPO_update() 
    
    











