import gymnasium as gym
import numpy as np

class BaseFlightWrapper(gym.Wrapper):
    """Common logic for all JSBSim tasks."""
    def __init__(self, env):
        super().__init__(env)
        self.raw = None

    def get_jsbsim_data(self):
        if self.raw is None:
            self.raw = self.env.unwrapped.sim.jsbsim
        return {
            'roll': self.raw.get_property_value('attitude/roll-rad'),
            'pitch': self.raw.get_property_value('attitude/pitch-rad'),
            'yaw_rate': self.raw.get_property_value('velocities/yaw-rad'),
            'heading': self.raw.get_property_value('attitude/psi-deg'),
            'track_error': self.raw.get_property_value('error/track-error-deg')
        }

class Stage1LevelHeading(BaseFlightWrapper):
    """Stage 1: maintain level flight and keep a fixed heading."""

    def __init__(self, env):
        super().__init__(env)
        self.target_heading = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw = self.env.unwrapped.sim.jsbsim
        self.target_heading = self.raw.get_property_value('attitude/psi-deg')
        self.raw.set_property_value('guidance/target-heading-deg', self.target_heading)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        d = self.get_jsbsim_data()

        roll_rew = np.exp(-(d['roll']**2) / 0.025)
        pitch_rew = np.exp(-(d['pitch']**2) / 0.020)
        heading_error = abs((self.target_heading - d['heading'] + 180) % 360 - 180)
        heading_rew = np.exp(-(heading_error**2) / 100.0)

        new_reward = 0.5 * roll_rew + 0.3 * pitch_rew + 0.2 * heading_rew

        if abs(d['roll']) > 0.5 or abs(d['pitch']) > 0.5:
            terminated = True
            new_reward = -25.0

        return obs, float(new_reward), terminated, truncated, info

class Stage2RandomHeading(BaseFlightWrapper):
    """Stage 2: level flight and navigate to a randomly selected heading."""
    def __init__(self, env, window=40):
        super().__init__(env)
        self.window = window
        self.target_heading = 0.0

        obs_space = env.observation_space
        if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
            low = np.concatenate((obs_space.low, np.array([-180.0], dtype=np.float64)))
            high = np.concatenate((obs_space.high, np.array([180.0], dtype=np.float64)))
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def augment_obs(self, obs):
        if self.raw is None:
            self.raw = self.env.unwrapped.sim.jsbsim
        track_error = self.raw.get_property_value('error/track-error-deg')
        if isinstance(obs, np.ndarray):
            return np.append(obs, np.array([track_error], dtype=np.float64))
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw = self.env.unwrapped.sim.jsbsim
        current_heading = self.raw.get_property_value('attitude/psi-deg')

        change = np.random.uniform(20.0, self.window) * np.random.choice([-1, 1])
        self.target_heading = (current_heading + change) % 360
        self.raw.set_property_value('guidance/target-heading-deg', self.target_heading)

        return self.augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        d = self.get_jsbsim_data()

        heading_error = abs((self.target_heading - d['heading'] + 180) % 360 - 180)
        roll = d['roll']
        pitch = d['pitch']

        heading_rew = np.exp(-(heading_error**2) / 10.0)
        roll_rew = np.exp(-(roll**2) / 0.025)
        pitch_rew = np.exp(-(pitch**2) / 0.020)

        new_reward = 0.6 * heading_rew + 0.2 * roll_rew + 0.2 * pitch_rew

        if abs(roll) > 0.6 or abs(pitch) > 0.5:
            terminated = True
            new_reward = -20.0

        return self.augment_obs(obs), float(new_reward), terminated, truncated, info

class Stage3AltitudeHeading(gym.Wrapper):
    """Stage 3: maintain altitude and navigate to random heading (2D control)."""

    def __init__(self, env):
        super().__init__(env)
        self.target_heading = 0.0
        self.target_altitude = 1000.0  # feet
        self.altitude_tolerance = 50.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped.sim.jsbsim
        current_heading = raw.get_property_value('attitude/psi-deg')
        current_alt = raw.get_property_value('position/h-agl-ft')

        change_heading = np.random.uniform(30.0, 120.0) * np.random.choice([-1, 1])
        self.target_heading = (current_heading + change_heading) % 360
        self.target_altitude = np.clip(current_alt + np.random.uniform(-200, 200), 500, 3000)

        raw.set_property_value('guidance/target-heading-deg', self.target_heading)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped.sim.jsbsim

        current_heading = raw.get_property_value('attitude/psi-deg')
        current_alt = raw.get_property_value('position/h-agl-ft')
        roll = raw.get_property_value('attitude/roll-rad')
        pitch = raw.get_property_value('attitude/pitch-rad')

        heading_error = abs((self.target_heading - current_heading + 180) % 360 - 180)
        altitude_error = abs(current_alt - self.target_altitude)

        heading_rew = np.exp(-(heading_error**2) / 150.0)
        altitude_rew = np.exp(-(altitude_error**2) / 5000.0)
        stability_rew = np.exp(-(roll**2) / 0.03) * np.exp(-(pitch**2) / 0.025)

        new_reward = 0.4 * heading_rew + 0.4 * altitude_rew + 0.2 * stability_rew

        if abs(roll) > 0.7 or abs(pitch) > 0.6:
            terminated = True
            new_reward = -30.0

        return obs, float(new_reward), terminated, truncated, info


class Stage4Waypoints(gym.Wrapper):
    """Stage 4: sequential waypoint following (full mission)."""

    def __init__(self, env, num_waypoints=3):
        super().__init__(env)
        self.num_waypoints = num_waypoints
        self.waypoints = []  # List of (heading, altitude) tuples
        self.current_waypoint = 0
        self.waypoint_distance_threshold = 100.0  # steps
        self.steps_at_waypoint = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped.sim.jsbsim
        current_heading = raw.get_property_value('attitude/psi-deg')
        current_alt = raw.get_property_value('position/h-agl-ft')

        self.waypoints = []
        for i in range(self.num_waypoints):
            heading = (current_heading + np.random.uniform(30, 120) * (1 if i % 2 == 0 else -1)) % 360
            altitude = np.clip(current_alt + np.random.uniform(-300, 300), 500, 3000)
            self.waypoints.append((heading, altitude))

        self.current_waypoint = 0
        self.steps_at_waypoint = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped.sim.jsbsim

        current_heading = raw.get_property_value('attitude/psi-deg')
        current_alt = raw.get_property_value('position/h-agl-ft')
        roll = raw.get_property_value('attitude/roll-rad')
        pitch = raw.get_property_value('attitude/pitch-rad')

        target_heading, target_altitude = self.waypoints[self.current_waypoint]

        heading_error = abs((target_heading - current_heading + 180) % 360 - 180)
        altitude_error = abs(current_alt - target_altitude)

        heading_rew = np.exp(-(heading_error**2) / 150.0)
        altitude_rew = np.exp(-(altitude_error**2) / 5000.0)
        stability_rew = np.exp(-(roll**2) / 0.03) * np.exp(-(pitch**2) / 0.025)

        waypoint_bonus = 0.0
        if heading_error < 5.0 and altitude_error < 50.0:
            self.steps_at_waypoint += 1
            waypoint_bonus = 0.5
            if self.steps_at_waypoint > self.waypoint_distance_threshold:
                if self.current_waypoint < self.num_waypoints - 1:
                    self.current_waypoint += 1
                    self.steps_at_waypoint = 0
                else:
                    waypoint_bonus = 5.0  # Mission complete
        else:
            self.steps_at_waypoint = 0

        new_reward = 0.35 * heading_rew + 0.35 * altitude_rew + 0.15 * stability_rew + waypoint_bonus

        if abs(roll) > 0.8 or abs(pitch) > 0.7:
            terminated = True
            new_reward = -40.0

        return obs, float(new_reward), terminated, truncated, info