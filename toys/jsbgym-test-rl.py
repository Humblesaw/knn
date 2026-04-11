"""
A simple example of using the Gymnasium-JSBSim environment with a deep learning agent.
"""

import gymnasium as gym

import gymnasium_jsbsim  # noqa: F401

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

# create the environment
env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0")

# create a vectorized env for training/rollout (1 copy)
vec_env = DummyVecEnv([lambda: gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0")])

# train model and attach vec_env
model = DDPG("MlpPolicy", env=vec_env, verbose=1)
model.learn(total_timesteps=10_000, log_interval=10)

# wrap the environment
vec_env = model.get_env()

single_env = vec_env.envs[0]

# reset the environment to start
obs, info = single_env.reset()

# define the maximum number of steps to run
MAX_STEPS = 100000
for episode in range(MAX_STEPS):
    
    action, _state = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = single_env.step(action)

    # print the results of the step
    done = terminated or truncated
    print(f"Obs: {obs}, State: {_state}, Reward: {reward}, Done: {done}\n")
    
    single_env.render()

    if done:
        obs, info = single_env.reset()

# Close the environment
env.close()
