import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
from stable_baselines3 import PPO
import time
import matplotlib.pyplot as plt
import os
os.environ["FG_ARGUMENTS"] = "--timeofday=noon"
# --- WRAPPER ---
class LevelFlightWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw_jsbsim = self.env.unwrapped.sim.jsbsim
        current_roll = raw_jsbsim.get_property_value('attitude/roll-rad')
        current_pitch = raw_jsbsim.get_property_value('attitude/pitch-rad')
        
        if abs(current_roll) > 0.52 or abs(current_pitch) > 0.40:
            terminated = True
        return observation, float(reward), terminated, truncated, info


base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0")
env = LevelFlightWrapper(base_env)
model = PPO.load("end_of_phase_1_test", env=env)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
roll_history, pitch_history, yaw_history = [], [], []
max_points = 100

# --- FLIGHT LOOP ---
state, info = env.reset()
try:
    for step in range(500000):
        action, _states = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        
        raw = env.unwrapped.sim.jsbsim
        r = np.degrees(raw.get_property_value('attitude/roll-rad'))
        p = np.degrees(raw.get_property_value('attitude/pitch-rad'))
        y = 270-np.degrees(raw.get_property_value('attitude/heading-true-rad'))

        roll_history.append(r)
        pitch_history.append(p)
        yaw_history.append(y)
        if len(roll_history) > max_points:
            roll_history.pop(0)
            pitch_history.pop(0)
            yaw_history.pop(0)

        if step % 5 == 0:
            ax.clear()
            ax.plot(roll_history, label=f"Roll: {r:.1f}°", color='red')
            ax.plot(pitch_history, label=f"Pitch: {p:.1f}°", color='green')
            ax.plot(yaw_history, label=f"Yaw: {y:.1f}°", color='blue')
            # Draw Target Line (Zero)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylim(-30, 30)
            ax.legend(loc='upper right')
            ax.set_title("AI Stability Monitor")
            plt.pause(0.001)

        env.unwrapped.render(mode="flightgear")
        
        if terminated or truncated:
            state, info = env.reset()
            roll_history, pitch_history, yaw_history = [], [], []

except KeyboardInterrupt:
    print("Stopping...")
finally:
    plt.ioff()
    plt.show()
    env.close()