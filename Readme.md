# Readme

This project teaches an agent to play Pong from Atari receiving as input directly the image pixels. Our approach employs Actor Critic with PPO update.

SimpleAI is a simple bot used for training and testing, which follows the ball with some noise in order to avoid always reflecting the ball with a straight trajectory.

The structure of the agent can be seen inside the file `agent.py`, while the environment and SimpleAI are defined into the folder `\wimblepong`. The file `report.pdf`  explains our design choices and the results obtained.

### Dependencies

Gym

PyTorch

Numpy

### Training

To train, simply run the file `train_PPO_4.py`. 

### Testing

To test the trained `model.mdl` against SimpleAI, simply run the file `test_42pong_simpleai.py`. 

The model named `model.mdl` has been trained against SimpleAI, while the model named `model_adversarial.mdl` has been trained in an adversarial way, against a really similar model, which was also being trained at the same time. Both of the models have been training for approximately a week.