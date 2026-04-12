import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
from stable_baselines3 import PPO

# Changed to gym.Wrapper so we can modify 'terminated'
class WingsLevelWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.target_heading = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Capture the heading exactly as we start
        raw = self.env.unwrapped.sim.jsbsim
        self.target_heading = raw.get_property_value('attitude/heading-true-rad')
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped.sim.jsbsim
        
        # Get Current States
        current_roll = raw.get_property_value('attitude/roll-rad')
        current_pitch = raw.get_property_value('attitude/pitch-rad')
        current_heading = raw.get_property_value('attitude/heading-true-rad')
        
        # Calculate Heading Error (Shortest Distance on a Circle)
        heading_error = np.arctan2(
            np.sin(current_heading - self.target_heading), 
            np.cos(current_heading - self.target_heading)
        )
        
        # Precision Rewards
        r_rew = np.exp(-(current_roll**2) / 0.0015)
        p_rew = np.exp(-(current_pitch**2) / 0.0025)
        
        # Heading Reward (Targeting 0 error)
        h_rew = np.exp(-(heading_error**2) / 0.0025)
        
        new_reward = (0.28 * r_rew) + (0.28 * p_rew) + (0.35 * h_rew)
        
        # Smoothness Penalty
        new_reward -= (abs(action[0]) + abs(action[1]) + abs(action[2])) * 0.05
        
        # Strict Kill Switches
        if abs(current_roll) > 0.35 or abs(current_pitch) > 0.35 or abs(heading_error) > 0.26:
            terminated = True
            new_reward = -25.0 
            
        return observation, float(new_reward), terminated, truncated, info

base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
env = WingsLevelWrapper(base_env)

model = PPO.load("level_flight_perfected5because4wasyawratedtrained", env=env)

# 0.00005
model.learning_rate = 0.0001

print("Starting Strict Refinement Training...")
model.learn(total_timesteps=750000, reset_num_timesteps=False)
model.save("end_of_phase_1_test")