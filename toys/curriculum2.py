import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
from stable_baselines3 import PPO
from  stable_baselines3.common.utils import LinearSchedule

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

        nav_rew = np.exp(-(track_error**2) / 5.0) 
        
        r_rew = np.exp(-(current_roll**2) / 0.25) 
        p_rew = np.exp(-(current_pitch**2) / 0.15)

        base_reward = nav_rew * (0.25 * r_rew + 0.25 * p_rew)
        
        hint_bonus = 0.0
        if (track_error > 5 and current_roll > 0.1) or (track_error < -5 and current_roll < -0.1):
            hint_bonus = 0.2 * nav_rew # Bonus only matters if we are somewhat on track
            
        new_reward = base_reward + hint_bonus
        
        new_reward -= (abs(action[0]) + abs(action[1]) + abs(action[2])) * 0.025
        
        if abs(current_roll) > 0.78 or abs(current_pitch) > 0.45:
            terminated = True
            new_reward = -30.0
            
        return observation, float(new_reward), terminated, truncated, info

base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
env = WaypointRefinementWrapper(base_env)

lr_schedule = LinearSchedule(start=3e-3, end=3e-6, end_fraction=0.8)

model = PPO.load("start_of_phase_2_test", env=env, ent_coef=0.01, learning_rate=lr_schedule)


print("Starting Phase 2: Heading Acquisition...")
model.learn(total_timesteps=250000, reset_num_timesteps=False)
model.save("can_you_yaw_please")