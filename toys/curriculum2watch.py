import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
from stable_baselines3 import PPO
import time
import matplotlib.pyplot as plt
import os

os.environ["FG_ARGUMENTS"] = "--timeofday=noon"

# --- WRAPPER (Ensure this matches your training wrapper exactly) ---
class WaypointRefinementWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.target_heading = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped.sim.jsbsim
        
        current_h = raw.get_property_value('attitude/psi-deg')
        
        side = np.random.choice([-1, 1])
        change = np.random.uniform(10, 30) * side
        self.target_heading = (current_h + change) % 360
        
        raw.set_property_value('guidance/target-heading-deg', self.target_heading)
        
        print(f"Randomized Objective: Turn {change:.1f}° to Heading {self.target_heading:.1f}°")
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped.sim.jsbsim
        
        track_error = raw.get_property_value('error/track-error-deg')
        current_roll = raw.get_property_value('attitude/roll-rad')
        current_pitch = raw.get_property_value('attitude/pitch-rad')

        nav_rew = np.exp(-(track_error**2) / 10.0) 
        
        r_rew = np.exp(-(current_roll**2) / 0.15) 
        p_rew = np.exp(-(current_pitch**2) / 0.05)

        base_reward = nav_rew * (0.5 * r_rew + 0.5 * p_rew)
        
        hint_bonus = 0.0
        if (track_error > 5 and current_roll > 0.1) or (track_error < -5 and current_roll < -0.1):
            hint_bonus = 0.2 * nav_rew # Bonus only matters if we are somewhat on track
            
        new_reward = base_reward + hint_bonus
        
        new_reward -= (abs(action[0]) + abs(action[1]) + abs(action[2])) * 0.025
        
        if abs(current_roll) > 0.78 or abs(current_pitch) > 0.45:
            terminated = True
            new_reward = -30.0
            
        return observation, float(new_reward), terminated, truncated, info

base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0")
env = WaypointRefinementWrapper(base_env)
model = PPO.load("can_you_yaw_please", env=env)

# --- MATPLOTLIB DUAL SETUP ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)

history = {'roll': [], 'pitch': [], 'heading': [], 'target': []}
max_points = 150

state, info = env.reset()

try:
    for step in range(500000):
        action, _states = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        
        raw = env.unwrapped.sim.jsbsim
        
        # Data Gathering
        r = np.degrees(raw.get_property_value('attitude/roll-rad'))
        p = np.degrees(raw.get_property_value('attitude/pitch-rad'))
        # Using heading-true-deg directly for simplicity
        h = raw.get_property_value('attitude/psi-deg')
        t = raw.get_property_value('guidance/target-heading-deg')

        if step % 50 == 0:
            print(f"Debug - Raw Heading: {raw.get_property_value('attitude/heading-true-deg')}")

        # Update History
        for key, val in zip(history.keys(), [r, p, h, t]):
            history[key].append(val)
            if len(history[key]) > max_points:
                history[key].pop(0)
    
        # Plotting every 5 steps
        if step % 5 == 0:
            # Top Plot: Attitude
            ax1.clear()
            ax1.plot(history['roll'], label=f"Roll: {r:.1f}°", color='red', linewidth=1.5)
            ax1.plot(history['pitch'], label=f"Pitch: {p:.1f}°", color='green', linewidth=1.5)
            ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax1.set_ylim(-40, 40)
            ax1.set_ylabel("Degrees")
            ax1.set_title("Aircraft Attitude (Stability)")
            ax1.legend(loc='upper right')

            # Bottom Plot: Navigation
            ax2.clear()
            ax2.plot(history['heading'], label=f"Actual Heading: {h:.1f}°", color='blue', linewidth=2)
            ax2.plot(history['target'], label=f"Target: {t:.1f}°", color='orange', linestyle='--', linewidth=2)
            ax2.set_ylabel("Heading (deg)")
            ax2.set_title("Navigation (Heading Control)")
            ax2.legend(loc='upper right')
            
            # Dynamic Y-limit for heading so we can see the detail
            ax2.set_ylim(0, 360) 

            plt.pause(0.001)

        env.unwrapped.render(mode="flightgear")
        
        if terminated or truncated:
            state, info = env.reset()
            for key in history: history[key] = []

except KeyboardInterrupt:
    print("Stopping...")
finally:
    plt.ioff()
    plt.show()
    env.close()