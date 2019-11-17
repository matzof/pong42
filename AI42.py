"""Created by Matzof on Sat Nov 16 22:23:22 2019"""
from dqn_agent import Agent as DQNAgent
from utils import extract_state
#%%
class AI42(object):
    def __init__(self, env, player_id=1):
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4                
        self.name = "AI42"
        self.agent = DQNAgent(4, 3, replay_buffer_size=50000,
                 batch_size=32, hidden_size=12, gamma=0.98)

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob, epsilon, model):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        state = extract_state(ob, model)
        action = self.agent.get_action(state, epsilon)
        return action

    def reset(self):
        # Nothing to done for now...
        return




















