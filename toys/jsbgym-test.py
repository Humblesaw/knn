"""
A simple example of using the Gymnasium-JSBSim environment with a random agent.
"""

import gymnasium as gym

import gymnasium_jsbsim  # noqa: F401

# Define the maximum number of steps to run
MAX_STEPS = 100

# Create the environment
env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")

# Reset the environment to start
env.reset()

for step in range(MAX_STEPS):
    # Take a random action
    action = env.action_space.sample()

    # Step the environment
    state, reward, terminated, truncated, info = env.step(action)

    # Print the results of the step
    done = terminated or truncated
    print(f"Step: {step}, State: {state}, Reward: {reward}, Done: {done}\n")

    # If the episode is done, exit the loop
    if done:
        break

# Close the environment
env.close()
