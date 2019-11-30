import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
from parallel_env import ParallelEnvs
import pandas as pd


# Policy training function
def train(env_name="ContinuousCartPole-v0"):
    update_steps = 10

    # Create a Gym environment
    # This creates 64 parallel envs running in 8 processes (8 threads each)
    env = ParallelEnvs(env_name, processes=8, envs_per_process=8)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    total_steps = 0

    # Run actual training
    reward_sum, timesteps = 0, 0
    done = False
    # Reset the environment and observe the initial state
    observation = env.reset()

    # Loop forever
    while True:
        # Get action from the agent
        action, action_probabilities = agent.get_action(observation)
        previous_observation = observation

        # Perform the action on the environment, get new state and reward
        observation, reward, done, info = env.step(action.detach().numpy())

        # Store action's outcome (so that the agent can improve its policy)
        for i in range(len(info["infos"])):
            env_done = False
            # Check if the environment is finished; if so, store cumulative reward
            for envid, envreward in info["finished"]:
                if envid == i:
                    reward_history.append(envreward)
                    average_reward_history.append(np.mean(reward_history[-500:]))
                    env_done = True
                    break
            agent.store_outcome(previous_observation[i], observation[i],
                                action_probabilities[i], reward[i], env_done)

        # Draw the frame, if desired
        #if args.render_training:
        #    env.render()

        # Store total episode reward
        timesteps += 1
        total_steps += 1

        if total_steps % update_steps == update_steps-1:
            agent.episode_finished(0)

        plot_freq = 5000
        if total_steps % plot_freq == plot_freq-1:
            # Training is finished - plot rewards
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "500-episode average"])
            plt.title("Non-episodic AC reward history")
            plt.savefig("rewards_%s.png" % env_name)
            plt.clf()
            torch.save(agent.policy.state_dict(), "model.mdl")
            print("%d: Plot and model saved." % total_steps)
    return data


# Function to test a trained policy
def test(episodes, agent, env):
    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            action = action.detach().numpy()
            observation, reward, done, info = env.step(action)

            if args.render_test:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # Print some stuff
    print("Environment:", args.env)
    #print("Training device:", agent.train_device)
    #print("Observation space dimensions:", observation_space_dim)
    #print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(env_name=args.env)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
        model_file = "%s_params.mdl" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)
    else:
        # Instantiate agent and its policy
        env = gym.make(args.env)

        # Get dimensionalities of actions and observations
        action_space_dim = env.action_space.shape[-1]
        observation_space_dim = env.observation_space.shape[-1]
        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(policy)
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(100, agent, env)

