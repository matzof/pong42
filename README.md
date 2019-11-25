# pong42 - Reinforcement Learning Project work

Mattia Zoffoli | Xinyi Tu

The final project work for the course (ELEC-E8125 - Reinforcement learning, 09.09.2019-04.12.2019, Aalto University) is about implementing a reinforcement learning agent that can play the game of Pong from pixels. In this environment, the agent controls one paddle and can take one of three actions: moving up or down, or staying in place.


RL algorithms experiments tried: 

• Deep Q Networks (DQN) - improvements over the ’basic’ Q-learning [3]

• Proximal policy optimization (PPO) — — applying additional constraints to policy gradient policy updates, similar idea to TRPO with simplified mathematical formulation [5]

• Actor-critic with experience replay (ACER) — reuse past experience when doing policy updates [6]


## The Pong game Components:

• wimblepong.py: Contains the implementation of the pong game. It provides the follow- ing public methods: step(actions), reset(), render(), set_names(name player 1, name player2). An example implementation of how these functions are used can be found in test_pong_simple_ai.py, which demonstrates an implementation for 2 simple hard-coded AI players,

• simple_ai.py: A simple hard-coded pong agent,

• test_pong_ai.py: This file contains an example implementation of two simple AI agents playing against each other,

• test_agent.py: This file tests your agent against Simple AI or against another agent (and can also be used to verify that your agent meets the interface requirements),

• mass_test_simple_ai.py: Tests all agents in the given directory against simple AI,

• epic_battle_royale.py: Tests all agents in the given directory against each other.


## Interfacing with the pong game

Your agent should be contained in a separate file called agent.py, and defined as a separate class called Agent. The Agent class must implement the following interface:

• load_model():void - a method that loads the saved model from a file,

• reset():void - a function that takes no arguments and resets the agent’s state after an episode is finished,

• get_action(frame:np.array):int - a function that takes a raw frame from Pong (as a numpy array), and returns an action (integer),

• get_name():str - a function that returns the name of the groups agent (max 16 characters, ASCII only).



## References
[1] D. Ha and J. Schmidhuber, “World models,” CoRR, vol. abs/1803.10122, 2018. [Online]. Available: http://arxiv.org/abs/1803.10122

[2] S.Levine,C.Finn,T.Darrell,andP.Abbeel,“End-to-endtrainingofdeepvisuomotorpolicies,” CoRR, vol. abs/1504.00702, 2015. [Online]. Available: http://arxiv.org/abs/1504.00702

[3] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. A. Riedmiller, “Playing atari with deep reinforcement learning,” CoRR, vol. abs/1312.5602, 2013. [Online]. Available: http://arxiv.org/abs/1312.5602

[4] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel, “Trust region policy optimization,” CoRR, vol. abs/1502.05477, 2015. [Online]. Available: http: //arxiv.org/abs/1502.05477

[5] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” CoRR, vol. abs/1707.06347, 2017. [Online]. Available: http://arxiv.org/abs/1707.06347

[6] Z. Wang, V. Bapst, N. Heess, V. Mnih, R. Munos, K. Kavukcuoglu, and N. de Freitas, “Sample efficient actor-critic with experience replay,” CoRR, vol. abs/1611.01224, 2016.
[Online]. Available: http://arxiv.org/abs/1611.01224
