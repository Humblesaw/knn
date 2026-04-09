import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
from stable_baselines3 import PPO
import time

# --- COPY YOUR WRAPPER HERE ---
class LevelFlightWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw_jsbsim = self.env.unwrapped.sim.jsbsim
        
        current_roll = raw_jsbsim.get_property_value('attitude/roll-rad')
        current_pitch = raw_jsbsim.get_property_value('attitude/pitch-rad')
        
        # Match your training Kill Switches exactly
        if abs(current_roll) > 0.52 or abs(current_pitch) > 0.40:
            terminated = True
            print(f"CRITICAL FAILURE: Roll {np.degrees(current_roll):.1f}°, Pitch {np.degrees(current_pitch):.1f}°")
            
        return observation, float(reward), terminated, truncated, info

# 1. Setup the Environment
base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0")

# 2. APPLY THE WRAPPER (This adds your Kill Switches back)
env = LevelFlightWrapper(base_env)

# 3. Load the model
model = PPO.load("level_flight_perfected3because2wasntgood", env=env)

state, info = env.reset()

for step in range(500000):
    action, _states = model.predict(state, deterministic=True)
    
    # This now uses YOUR 'step' logic, which checks for banking/pitching
    state, reward, terminated, truncated, info = env.step(action)
    
    env.unwrapped.render(mode="flightgear")
    time.sleep(0.02)

    if terminated or truncated:
        print("Kill Switch Triggered or Task End. Resetting...")
        state, info = env.reset()

env.close()