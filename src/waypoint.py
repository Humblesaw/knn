import enum
import math
import random
import types
import warnings
import socket
import subprocess
import time
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
import waypointsgen

class MinMaxObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dist_idx = len(FlightTask.base_state_variables)
        self.head_idx = len(FlightTask.base_state_variables) + 1
        self.elev_idx = len(FlightTask.base_state_variables) + 2

    def observation(self, obs):
        # normalize to [0, 1]
        obs_range = (self.observation_space.high - self.observation_space.low)
        obs_range[obs_range == 0] = 1e-10
        norm = (obs - self.observation_space.low) / obs_range

        # clip to ensure no out-of-bounds values (FlightGear can sometimes spit out anomalies)
        norm = np.clip(norm, 0, 1)

        # keep the errors though
        norm[self.dist_idx] = obs[self.dist_idx]
        norm[self.head_idx] = obs[self.head_idx]
        norm[self.elev_idx] = obs[self.elev_idx]

        return norm

class Reward:
    """Simple wrapper for rewards expected by FlightTask"""
    def __init__(self, agent_reward: float, assessment_reward: float):
        self._agent_reward = agent_reward
        self._assessment_reward = assessment_reward

    def agent_reward(self): return self._agent_reward
    def assessment_reward(self): return self._assessment_reward

class WaypointAssessor(Assessor):
    """Calculates rewards based on a Bell Curve (Gaussian) distance distribution."""
    
    def __init__(self, hit_bonus: float = 1000.0):
        self.hit_bonus = hit_bonus
        
        # --- BELL CURVE PARAMETERS ---
        self.base_multiplier = 0.01  # tiny reward when far away
        self.peak_multiplier = 0.20  # massive reward spike when close
        self.spread_ft = 2000.0      # how wide the bell curve is (starts pulling hard at 2000ft)

    def assess(self, state: Tuple, last_state: Tuple, is_terminal: bool) -> Reward:
        if last_state is None:
            return Reward(0.0, 0.0)

        # grab the custom variables from the state array
        base_idx = len(FlightTask.base_state_variables)
        current_dist = state[base_idx]
        last_dist = last_state[base_idx]
        
        current_heading_err = state[base_idx + 1]
        current_elev_err = state[base_idx + 2]

        # --- THE BELL CURVE MATH ---
        # calculate where we are on the curve: e^(-(distance^2) / (2 * spread^2))
        # this outputs 1.0 if distance is 0, and approaches 0.0 as distance grows.
        bell_curve_value = math.exp(-(current_dist**2) / (2 * (self.spread_ft**2)))
        
        # calculate the dynamic multiplier for this exact frame
        dynamic_multiplier = self.base_multiplier + (self.peak_multiplier * bell_curve_value)

        # apply the dynamic multiplier to the distance covered
        delta_dist = last_dist - current_dist
        step_reward = delta_dist * dynamic_multiplier
        
        # --- PENALTIES ---
        # TIME PENALTY: lose points every step to stop infinite circling
        step_reward -= 0.5
        
        # ALIGNMENT PENALTY: lose points if the nose isn't pointing at the waypoint
        alignment_penalty = (abs(current_heading_err) + abs(current_elev_err)) * 0.5
        step_reward -= alignment_penalty

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
                hit_radius_ft: float = 32.8,
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
        super()._new_episode_init(sim)
        
        # set standard engine controls
        sim.set_throttle_mixture_controls(constants.THROTTLE_CMD, constants.MIXTURE_CMD)
        
        # --- THE FIX: Pre-trim the nose down ---
        # this counteracts the massive lift generated at cruise speed, 
        # allowing the agent's neutral (0.0) stick output to fly perfectly level!
        sim.jsbsim["fcs/pitch-trim-cmd-norm"] = -0.15
        
        # generate dynamic route based on current position
        generator = waypointsgen.RouteGenerator(
            start_lat=sim[prp.lat_geod_deg], 
            start_lon=sim[prp.lng_geoc_deg], 
            start_alt=sim[prp.altitude_sl_ft],
            start_heading_deg=sim[prp.heading_deg]
        )
        
        current_difficulty = getattr(self, 'curriculum_difficulty', 0.0) 
        
        self.original_waypoints = generator.generate_route(num_waypoints=10, difficulty=current_difficulty)
        self.active_waypoints = self.original_waypoints.copy()
        
        self.waypoints_cleared = 0
        self.just_hit_waypoint = False
        self.current_step = 0

    def _update_custom_properties(self, sim: 'Simulation') -> None:
        """
        Calculates distance and relative angles to the waypoint.
        Pops the waypoint if the aircraft is within the hit radius.
        """
        self.just_hit_waypoint = False

        if not self.active_waypoints:
            # no waypoints left, lock properties to 0
            sim[self.custom_wp_dist] = 0.0
            sim[self.custom_wp_heading_err] = 0.0
            sim[self.custom_wp_elev_err] = 0.0
            return

        # get current aircraft position
        lat = sim[prp.lat_geod_deg]
        lon = sim[prp.lng_geoc_deg]
        alt = sim[prp.altitude_sl_ft]
        heading_deg = sim[prp.heading_deg] 
        heading = math.radians(heading_deg)
        pitch = sim[prp.pitch_rad]

        # get active waypoint
        target_lat, target_lon, target_alt = self.active_waypoints[0]

        # calculate distance (using simple Equirectangular approximation for speed)
        # 1 degree of latitude is roughly 364,000 feet.
        lat_dist_ft = (target_lat - lat) * 364000.0
        lon_dist_ft = (target_lon - lon) * (math.cos(math.radians(lat)) * 364000.0)
        alt_dist_ft = target_alt - alt
        
        # 3D Distance
        dist_ft = math.sqrt(lat_dist_ft**2 + lon_dist_ft**2 + alt_dist_ft**2)
        
        # check for waypoint hit
        if dist_ft < self.hit_radius_ft:
            self.active_waypoints.pop(0)
            self.waypoints_cleared += 1
            self.just_hit_waypoint = True
            
            # recalculate immediately if there is a next waypoint
            if self.active_waypoints:
                return self._update_custom_properties(sim)
            else:
                dist_ft = 0.0

        # calculate relative heading (yaw error)
        # math.atan2(y, x) -> y is longitude difference, x is latitude difference
        target_bearing_rad = math.atan2(lon_dist_ft + 1e-10, lat_dist_ft + 1e-10)
        heading_error = target_bearing_rad - heading
        
        # normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # calculate relative elevation (pitch error)
        # ground distance
        ground_dist_ft = math.sqrt(lat_dist_ft**2 + lon_dist_ft**2)
        target_pitch_rad = math.atan2(alt_dist_ft, ground_dist_ft)
        pitch_error = target_pitch_rad - pitch

        # write to sim
        sim[self.custom_wp_dist] = dist_ft
        sim[self.custom_wp_heading_err] = heading_error
        sim[self.custom_wp_elev_err] = pitch_error

    def _is_terminal(self, sim: 'Simulation') -> bool:
        """Episode ends if all waypoints are hit, plane crashes, or it flies out of bounds."""
        if not self.active_waypoints:
            return True
            
        alt = sim[prp.altitude_sl_ft]
        if alt <= 0.0:
            return True

        roll_rad = abs(sim[prp.roll_rad])
        pitch_rad = abs(sim[prp.pitch_rad])
        
        # spinning out of control
        if roll_rad > 0.5 or pitch_rad > 0.5:
            return True
            
        # NEW: out of bounds check
        # waypoints spawn ~10,000 ft away, if it gets 25,000ft away, it is flying in the wrong direction
        dist_ft = sim[self.custom_wp_dist]
        if dist_ft > 25000.0:
            return True
            
        return False

    def _reward_terminal_override(self, reward: Reward, sim: 'Simulation') -> Reward:
        """Apply sparse rewards for hits, completion, failure, or flying away."""
        agent_rew = reward.agent_reward()
        
        # penalty for hitting the ground OR losing control
        if sim[prp.altitude_sl_ft] <= 0.0 or abs(sim[prp.roll_rad]) > 0.5 or abs(sim[prp.pitch_rad]) > 0.5:
            agent_rew -= 1000.0
            
        # NEW: penalty for flying out of bounds
        elif sim[self.custom_wp_dist] > 25000.0:
            agent_rew -= 1000.0
            
        # massive bonus for clearing the whole course
        elif not self.active_waypoints:
            agent_rew += 2000.0

        return Reward(agent_rew, reward.assessment_reward())
        
    def task_step(self, sim: 'Simulation', action: Sequence[float], sim_steps: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Override to inject the waypoint hit bonus, and manually trim the elevator."""
        
        # --- THE FOOLPROOF TRIM FIX ---
        # action[0] = aileron, action[1] = elevator, action[2] = rudder
        modified_action = list(action)
        
        # push the stick forward by 15% to counteract cruise lift
        modified_action[1] -= 0.15 
        
        # clamp it to make sure we don't accidentally send a value outside [-1.0, 1.0]
        modified_action[1] = max(-1.0, min(1.0, modified_action[1]))
        
        # pass our modified, level-flying action to the physics engine instead
        state, reward, terminated, truncated, info = super().task_step(sim, modified_action, sim_steps)
        
        # if we just popped a waypoint during _update_custom_properties, add the sparse bonus
        if self.just_hit_waypoint:
            reward += self.assessor.hit_bonus
            info["waypoint_hit"] = True
            
        info["waypoints_remaining"] = len(self.active_waypoints)

        # early stopping logic
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        state_values = [sim[prop] for prop in self.state_variables]

        # --- DEBUG START ---
        if any(math.isnan(v) or math.isinf(v) for v in state_values):
            print("!!! NaN/Inf detected in State !!!")
            # severe penalty for "disintegrating" the aircraft
            reward = -20000.0
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

class WaypointVisualiser:
    """
    Class for visualising aircraft using the FlightGear simulator.

    This visualiser launches FlightGear and (by default) waits for it to
    launch. A Figure is also displayed (by creating its own FigureVisualiser)
    which is used to display the agent's actions.
    """

    def __init__(
        self, sim: Simulation, model_path, block_until_loaded=True
    ):
        """
        Launches FlightGear in subprocess and starts figure for plotting actions.

        :param sim: Simulation that will be visualised
        :param aircraft: Aircraft to be loaded in FlightGear for visualisation
        :param print_props: collection of Propertys to be printed to Figure
        :param block_until_loaded: visualiser will block until it detects that
            FlightGear has loaded if True.
        """
        self.configure_simulation_output(sim)
        # self.print_props = print_props
        # Note: subprocess is managed manually (not with context manager)
        # because it needs to stay alive for the visualiser's lifetime
        # and is explicitly closed in close() method
        self.flightgear_process = self._launch_flightgear(sim.get_aircraft())
        # self.figure = FigureVisualiser(sim, print_props)
        if block_until_loaded:
            time.sleep(20)
            # self._block_until_flightgear_loaded()

        self.host = "127.0.0.1"
        self.port = 5555
        self.model_path = model_path

    def send_command(self, cmd: str):
        """Sends a raw command to the FlightGear telnet server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.host, self.port))
                s.sendall((cmd + '\r\n').encode('utf-8'))
                response = s.recv(1024).decode('utf-8').strip()
                print(f"Sent: {cmd} | FG Response: {response}")
        except Exception as e:
            # flightgear might not be running yet, which is fine during training
            print(f"Error: {e}")

    def draw_waypoints(self, waypoints):
        """Injects 3D models into FlightGear at the waypoint coordinates."""
        # we start at index 100 to avoid overwriting default FG models
        base_idx = 100 
        
        for i, (lat, lon, alt) in enumerate(waypoints):
            model_idx = base_idx + i
            prefix = f"/models/model[{model_idx}]"
            
            # send the properties to flightgear to spawn the object
            commands = [
                f"set {prefix}/path {self.model_path}",
                f"set {prefix}/latitude-deg {lat}",
                f"set {prefix}/longitude-deg {lon}",
                f"set {prefix}/elevation-ft {alt}",
                f"set {prefix}/load 1" # render
            ]
            
            for cmd in commands:
                self.send_command(cmd)

    def clear_waypoints(self, waypoint_idxs):
        """Removes the models from the sky."""
        base_idx = 100
        for i in waypoint_idxs:
            # unloading the model removes it from the screen
            self.send_command(f"set /models/model[{base_idx + i}]/load 0")

    def plot(self, sim: Simulation) -> None:
        """
        Updates a 3D plot of agent actions.
        """
        # plot nothing for the time being
        # self.figure.plot(sim)

    @staticmethod
    def _launch_flightgear(aircraft: Aircraft):
        """Launch FlightGear subprocess for visualization."""
        cmd_line_args = WaypointVisualiser._create_cmd_line_args(
            aircraft.flightgear_id
        )
        # subprocess is not used with context manager because it needs to persist
        # and is managed manually via close() method
        import os

        # copy the current environment variables
        clean_env = os.environ.copy()

        # remove the Qt and library paths that might confuse the AppImage
        clean_env.pop("QT_PLUGIN_PATH", None)
        clean_env.pop("LD_LIBRARY_PATH", None)
        clean_env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        flightgear_process = subprocess.Popen(
            cmd_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=clean_env,
        )
        return flightgear_process

    def configure_simulation_output(self, sim: Simulation):
        """Configure simulation for FlightGear output."""
        sim.enable_flightgear_output()
        sim.set_simulation_time_factor(constants.FG_TIME_FACTOR)

    @staticmethod
    def _create_cmd_line_args(aircraft_id: str):
        # flightgear doesn't have a 172X model, use the P instead
        if aircraft_id == "c172x":
            aircraft_id = "c172p"

        flightgear_cmd = "fgfs"
        aircraft_arg = f"--aircraft={aircraft_id}"
        flight_model_arg = (
            "--native-fdm=" + f"{constants.FG_TYPE},"
            f"{constants.FG_DIRECTION},"
            f"{constants.FG_RATE},"
            f"{constants.FG_SERVER},"
            f"{constants.FG_PORT},"
            f"{constants.FG_PROTOCOL}"
        )
        flight_model_type_arg = "--fdm=" + "external"
        disable_ai_arg = "--disable-ai-traffic"
        disable_live_weather_arg = "--disable-real-weather-fetch"
        time_of_day_arg = "--timeofday=noon"
        return (
            flightgear_cmd,
            aircraft_arg,
            flight_model_arg,
            flight_model_type_arg,
            disable_ai_arg,
            disable_live_weather_arg,
            time_of_day_arg,
            "--disable-clouds",
            "--disable-clouds3d",
            "--fog-disable",
            "--telnet=socket,in,10,,5555,tcp",
        )

    def _block_until_flightgear_loaded(self):
        """Wait until FlightGear has finished loading."""
        while True:
            msg_out = self.flightgear_process.stdout.readline().decode()
            if constants.FG_LOADED_MESSAGE in msg_out:
                break
            time.sleep(0.001)

    def close(self):
        """Close FlightGear and figure visualiser."""
        if self.flightgear_process:
            self.flightgear_process.kill()
            timeout_secs = 1
            self.flightgear_process.wait(timeout=timeout_secs)
