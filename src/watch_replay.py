import argparse
import gymnasium as gym
import gymnasium_jsbsim  # noqa: F401
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import SAC
import numpy as np

import gymnasium_jsbsim.properties as prp
from waypoint import WaypointVisualiser, MinMaxObservationWrapper
from gymnasium.envs.registration import register
from gymnasium_jsbsim.aircraft import cessna172P
from gymnasium_jsbsim.tasks import Shaping

# Ensure the FG version of the environment is registered
from train_waypoint import WaypointTaskFactory 
register(
    id="JSBSim-WaypointTask-Cessna172P-Shaping.STANDARD-FG-v0",
    entry_point="gymnasium_jsbsim.environment:JsbSimEnv", 
    kwargs={"task_type": WaypointTaskFactory, "aircraft": cessna172P, "shaping": Shaping.STANDARD}
)

def setup_live_plot(start_lat, start_lon, start_alt, waypoints, difficulty):
    """Sets up the Matplotlib window for live 3D rendering."""
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Extract Target Waypoints
    wp_lats = [wp[0] for wp in waypoints]
    wp_lons = [wp[1] for wp in waypoints]
    wp_alts = [wp[2] for wp in waypoints]
    
    # 2. Draw Target Waypoints
    ax.scatter(wp_lons, wp_lats, wp_alts, color='red', s=100, label='Target Waypoints')
    ax.plot(wp_lons, wp_lats, wp_alts, color='red', linestyle='dashed', alpha=0.5)
    ax.scatter([start_lon], [start_lat], [start_alt], color='green', s=100, label='Start')

    # 3. Create the empty flight line that we will update live
    flight_line, = ax.plot([], [], [], label='Live Flight Path', color='blue', linewidth=2)

    # 4. Lock the Axis Limits
    all_lats = wp_lats + [start_lat]
    all_lons = wp_lons + [start_lon]
    all_alts = wp_alts + [start_alt]
    
    lat_pad = (max(all_lats) - min(all_lats)) * 0.2 + 0.01
    lon_pad = (max(all_lons) - min(all_lons)) * 0.2 + 0.01
    alt_pad = 1000 
    
    ax.set_xlim([min(all_lons) - lon_pad, max(all_lons) + lon_pad])
    ax.set_ylim([min(all_lats) - lat_pad, max(all_lats) + lat_pad])
    ax.set_zlim([min(all_alts) - alt_pad, max(all_alts) + alt_pad])

    # Show the current difficulty on the graph title!
    ax.set_title(f'Live Trajectory Tracking (Difficulty: {difficulty})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (ft)')
    ax.legend()
    
    plt.show()
    return fig, ax, flight_line

if __name__ == "__main__":
    # --- COMMAND LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Watch a trained agent fly in FlightGear.")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to the checkpoint .zip file you want to watch.")
    parser.add_argument("--diff", type=float, default=0.0,
                        help="Difficulty of the generated route (0.0 to 1.0). Default: 0.0")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to watch. Default: 3")
    args = parser.parse_args()

    # --- INITIALIZATION ---
    env = MinMaxObservationWrapper(gym.make("JSBSim-WaypointTask-Cessna172P-Shaping.STANDARD-FG-v0"))

    print(f"Loading model from {args.resume}...")
    model = SAC.load(args.resume, env=env)

    raw_env = env.unwrapped
    
    # Set the difficulty based on your CLI input!
    raw_env.task.curriculum_difficulty = args.diff
    
    # Initialize JSBSim before starting the visualizer
    obs, info = env.reset()
    vis = WaypointVisualiser(raw_env.sim)

    # --- MAIN REPLAY LOOP ---
    for episode in range(args.episodes): 
        target_waypoints = raw_env.task.original_waypoints.copy()
        vis.draw_waypoints(target_waypoints)

        # Get Starting Position
        start_lat = raw_env.sim[prp.lat_geod_deg]
        start_lon = raw_env.sim[prp.lng_geoc_deg]
        start_alt = raw_env.sim[prp.altitude_sl_ft]

        # Setup the Live Plot (passing the difficulty)
        fig, ax, flight_line = setup_live_plot(start_lat, start_lon, start_alt, target_waypoints, args.diff)

        flight_lats, flight_lons, flight_alts = [], [], []

        print(f"\n--- Starting Live Replay Episode {episode + 1}/{args.episodes} ---")
        done = False
        step_counter = 0
        
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record coordinates
            flight_lats.append(raw_env.sim[prp.lat_geod_deg])
            flight_lons.append(raw_env.sim[prp.lng_geoc_deg])
            flight_alts.append(raw_env.sim[prp.altitude_sl_ft])

            # LIVE UPDATE LOGIC
            if step_counter % 10 == 0:
                flight_line.set_data(flight_lons, flight_lats)
                flight_line.set_3d_properties(flight_alts)
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            step_counter += 1

        print("Episode finished! Waiting 3 seconds before next flight...")
        plt.pause(3.0) 
        plt.close(fig) 
        vis.clear_waypoints(len(target_waypoints))
        
        # Reset the environment for the next episode
        obs, info = env.reset()

    env.close()
    vis.close()