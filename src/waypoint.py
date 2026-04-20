import enum
import math
import random
import types
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, List

import gymnasium as gym
import numpy as np

import gymnasium_jsbsim.properties as prp
from gymnasium_jsbsim import assessors, constants, rewards, utils
from gymnasium_jsbsim.assessors import Assessor
from gymnasium_jsbsim.aircraft import Aircraft
from gymnasium_jsbsim.properties import BoundedProperty, Property
from gymnasium_jsbsim.rewards import RewardStub
from gymnasium_jsbsim.simulation import Simulation
from gymnasium_jsbsim.tasks import FlightTask

class MinMaxObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # Normalize to [0, 1]
        obs_range = (self.observation_space.high - self.observation_space.low)
        obs_range[obs_range == 0] = 1e-10

        norm = (obs - self.observation_space.low) / obs_range
        
        # Clip to ensure no out-of-bounds values (FlightGear can sometimes spit out anomalies)
        return np.clip(norm, 0, 1)

class Reward:
    """Simple wrapper for rewards expected by FlightTask"""
    def __init__(self, agent_reward: float, assessment_reward: float):
        self._agent_reward = agent_reward
        self._assessment_reward = assessment_reward

    def agent_reward(self): return self._agent_reward
    def assessment_reward(self): return self._assessment_reward

class WaypointAssessor(Assessor):
    """Calculates rewards based on distance to the active waypoint."""
    
    def __init__(self, hit_bonus: float = 100.0, shaping_multiplier: float = 0.1):
        self.hit_bonus = hit_bonus
        self.shaping_multiplier = shaping_multiplier

    def assess(self, state: Tuple, last_state: Tuple, is_terminal: bool) -> Reward:
        if last_state is None:
            return Reward(0.0, 0.0)

        # get custom_wp_dist which is right after base_state_variables
        current_dist = state[len(FlightTask.base_state_variables)]
        last_dist = last_state[len(FlightTask.base_state_variables)]

        # Dense Reward: Positive if we moved closer, negative if we moved away
        delta_dist = last_dist - current_dist
        step_reward = delta_dist * self.shaping_multiplier
        
        # We handle the big hit_bonus inside the Task itself when the list pops,
        # but the shaping (breadcrumb) reward is calculated here every step.
        return Reward(step_reward, step_reward)

class WaypointTask(FlightTask):
    """
    A 3D Waypoint following task. The agent must fly through sequential 3D coordinates.
    """

    custom_wp_dist = BoundedProperty("custom/waypoint/dist_ft",
        "Distance to active waypoint",
        0.0, 500000.0)
    custom_wp_heading_err = BoundedProperty("custom/waypoint/heading_error_rad",
        "Heading error to waypoint",
        -math.pi, math.pi)
    custom_wp_elev_err = BoundedProperty("custom/waypoint/elevation_error_rad",
        "Elevation error to waypoint",
        -math.pi, math.pi)

    action_variables = (
        prp.aileron_cmd,
        prp.elevator_cmd,
        prp.rudder_cmd,
    )

    def __init__(self,
                aircraft: Aircraft,
                assessor: WaypointAssessor,
                waypoints: List[Tuple[float, float, float]],
                hit_radius_ft: float = 500.0,
                max_steps: int = 1000,
                debug: bool = False):
    
        self.aircraft = aircraft
        # (latitude_deg, longitude_deg, altitude_ft)
        self.original_waypoints = waypoints.copy()
        self.active_waypoints = []
        self.hit_radius_ft = hit_radius_ft
        self.waypoints_cleared = 0
        self.just_hit_waypoint = False

        self.state_variables = (
            FlightTask.base_state_variables + (
                self.custom_wp_dist,
                self.custom_wp_heading_err,
                self.custom_wp_elev_err,
            )
        )

        # early stopping
        self.max_steps = max_steps
        self.current_step = 0
    
        super().__init__(assessor, debug)

    def get_initial_conditions(self) -> Dict[Property, float]:
        """Returns starting conditions. Uses base conditions but can be overridden."""
        extra_conditions = {
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: constants.INITIAL_HEADING_DEG,
        }
        return {**self.base_initial_conditions, **extra_conditions}

    def _new_episode_init(self, sim: 'Simulation') -> None:
        """Reset waypoint list on new episode."""
        print("NEW EPISODE BEGINS")
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(constants.THROTTLE_CMD, constants.MIXTURE_CMD)
        self.active_waypoints = self.original_waypoints.copy()
        self.waypoints_cleared = 0
        self.just_hit_waypoint = False
        # reset step counter
        self.current_step = 0

    def _update_custom_properties(self, sim: 'Simulation') -> None:
        """
        Calculates distance and relative angles to the waypoint.
        Pops the waypoint if the aircraft is within the hit radius.
        """
        self.just_hit_waypoint = False

        if not self.active_waypoints:
            # No waypoints left, lock properties to 0
            sim[self.custom_wp_dist] = 0.0
            sim[self.custom_wp_heading_err] = 0.0
            sim[self.custom_wp_elev_err] = 0.0
            return

        # 1. Get current aircraft position
        lat = sim[prp.lat_geod_deg]
        lon = sim[prp.lng_geoc_deg]
        alt = sim[prp.altitude_sl_ft]
        heading_deg = sim[prp.heading_deg] 
        heading = math.radians(heading_deg)
        pitch = sim[prp.pitch_rad]

        # 2. Get active waypoint
        target_lat, target_lon, target_alt = self.active_waypoints[0]

        # 3. Calculate Distance (Using simple Equirectangular approximation for speed)
        # 1 degree of latitude is roughly 364,000 feet.
        lat_dist_ft = (target_lat - lat) * 364000.0
        lon_dist_ft = (target_lon - lon) * (math.cos(math.radians(lat)) * 364000.0)
        alt_dist_ft = target_alt - alt
        
        # 3D Distance
        dist_ft = math.sqrt(lat_dist_ft**2 + lon_dist_ft**2 + alt_dist_ft**2)
        
        # 4. Check for Waypoint Hit
        if dist_ft < self.hit_radius_ft:
            self.active_waypoints.pop(0)
            self.waypoints_cleared += 1
            self.just_hit_waypoint = True
            
            # Recalculate immediately if there is a next waypoint
            if self.active_waypoints:
                return self._update_custom_properties(sim)
            else:
                dist_ft = 0.0

        # 5. Calculate Relative Heading (Yaw Error)
        # math.atan2(y, x) -> y is longitude difference, x is latitude difference
        target_bearing_rad = math.atan2(lon_dist_ft + 1e-10, lat_dist_ft + 1e-10)
        heading_error = target_bearing_rad - heading
        
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # 6. Calculate Relative Elevation (Pitch Error)
        # Ground distance
        ground_dist_ft = math.sqrt(lat_dist_ft**2 + lon_dist_ft**2)
        target_pitch_rad = math.atan2(alt_dist_ft, ground_dist_ft)
        pitch_error = target_pitch_rad - pitch

        # 7. Write to Simulation
        sim[self.custom_wp_dist] = dist_ft
        sim[self.custom_wp_heading_err] = heading_error
        sim[self.custom_wp_elev_err] = pitch_error

    def _is_terminal(self, sim: 'Simulation') -> bool:
        """Episode ends if all waypoints are hit or plane crashes."""
        if not self.active_waypoints:
            return True # Success!
            
        alt = sim[prp.altitude_sl_ft]
        if alt <= 0.0:
            return True # Crashed

        roll_rad = abs(sim[prp.roll_rad])
        pitch_rad = abs(sim[prp.pitch_rad])
        
        # 1.5 radians is about 85 degrees. 
        # If the Cessna is banking or pitching this hard, it's out of control.
        if roll_rad > 1.5 or pitch_rad > 1.5:
            return True
            
        return False

    def _reward_terminal_override(self, reward: Reward, sim: 'Simulation') -> Reward:
        """Apply sparse rewards for hits, completion, or failure."""
        agent_rew = reward.agent_reward()
        
        # Large penalty for crashing
        if sim[prp.altitude_sl_ft] <= 0.0:
            agent_rew -= 1000.0
            
        # Massive bonus for clearing the whole course
        elif not self.active_waypoints:
            agent_rew += 2000.0

        return Reward(agent_rew, reward.assessment_reward())
        
    def task_step(self, sim: 'Simulation', action: Sequence[float], sim_steps: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Override to inject the waypoint hit bonus into the step reward."""
        state, reward, terminated, truncated, info = super().task_step(sim, action, sim_steps)
        
        # If we just popped a waypoint during _update_custom_properties, add the sparse bonus
        if self.just_hit_waypoint:
            reward += self.assessor.hit_bonus
            info["waypoint_hit"] = True
            
        info["waypoints_remaining"] = len(self.active_waypoints)

        # Early Stopping Logic
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            # Optional: You can subtract a penalty from 'reward' here 
            # if you want to actively punish the agent for being too slow.

        state_values = [sim[prop] for prop in self.state_variables]

        # --- DEBUG START ---
        if any(math.isnan(v) or math.isinf(v) for v in state_values):
            print("!!! NaN/Inf detected in State !!!")
            for name, val in zip(self.State._fields, state_values):
                print(f"{name}: {val}")

            # Severe penalty for "disintegrating" the aircraft
            reward = -2000.0 
            terminated = True
            info["physics_crash"] = True
            
        # --- DEBUG END ---
        
        return state, reward, terminated, truncated, info

    def get_props_to_output(self) -> Tuple:
        """Get properties to output for visualization and monitoring."""
        return (
            self.last_agent_reward,
            self.last_assessment_reward,
        )
