"""Created by Matzof on Sat Nov 16 22:23:22 2019"""
from agent_PPO import Agent, Policy
from utils import extract_state, extract_state_cheating
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
        self.policy = Policy()
        self.agent = Agent(self.policy)

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob, model):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        state = extract_state(ob, model)
        action, action_probabilities = self.agent.get_action(state)
        return action, action_probabilities
    
    def get_action_cheating(self, ob):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        state = extract_state_cheating(self.env, self.player_id)
        action, action_probabilities = self.agent.get_action(state)
        return action, action_probabilities

    def reset(self):
        # Nothing to done for now...
        return




















