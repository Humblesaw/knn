import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
from stable_baselines3 import PPO

# Changed to gym.Wrapper so we can modify 'terminated'
class WingsLevelWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw_jsbsim = self.env.unwrapped.sim.jsbsim
        
        # 1. Get Orientation & Rate Data
        current_roll = raw_jsbsim.get_property_value('attitude/roll-rad')
        current_pitch = raw_jsbsim.get_property_value('attitude/pitch-rad')
        yaw_rate = raw_jsbsim.get_property_value('velocities/r-rad-sec') # Yaw rate
        
        # 2. ULTRA-TIGHT REWARDS
        # Roll: 0.005 is very strict. Pitch: 0.01 is strict.
        roll_reward = np.exp(-(current_roll**2) / 0.0025)
        pitch_reward = np.exp(-(current_pitch**2) / 0.0075)
        
        # 3. YAW STABILITY (The "Ball in the Center")
        # Reward the AI for NOT yawing/spinning.
        yaw_reward = np.exp(-(yaw_rate**2) / 0.001)
        
        # 4. COMBINE (Weighted)
        # 40% Roll, 40% Pitch, 20% Yaw stability
        new_reward = (0.4 * roll_reward) + (0.4 * pitch_reward) + (0.2 * yaw_reward)
        

        # 6. ACTION PENALTY (Smoothness)
        new_reward -= (abs(action[0]) + abs(action[1]) + abs(action[2])) * 0.025
        
        # 7. STRICT KILL SWITCHES
        # Reduced bank limit to 20 degrees (0.35 rad) to force correction sooner
        if abs(current_roll) > 0.35 or abs(current_pitch) > 0.35:
            terminated = True
            new_reward = -20.0 
            
        return observation, float(new_reward), terminated, truncated, info

# --- Setup and Train ---
base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
env = WingsLevelWrapper(base_env)

# Load the brain you just visualized
model = PPO.load("level_flight_perfected3because2wasntgood", env=env)

# Lower the learning rate! 
# Since the model is already "smart," we don't want to blow up its brain.
# 0.00005 (half of before) is good for fine-tuning.
model.learning_rate = 0.00008

print("Starting Strict Refinement Training...")
model.learn(total_timesteps=500000, reset_num_timesteps=False)
model.save("level_flight_perfected4because3wasyawing")