import gymnasium as gym
import gymnasium_jsbsim  # noqa: F401
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from gymnasium.envs.registration import register
from gymnasium_jsbsim.aircraft import cessna172P
from gymnasium_jsbsim.tasks import Shaping

from functools import partial
from waypoint import WaypointTask, WaypointAssessor, MinMaxObservationWrapper
from waypointsgen import RouteGenerator

dummy_waypoints = [(0.0, 0.0, 5000)]

def WaypointTaskFactory(*args, **kwargs):
    my_assessor = WaypointAssessor()
    return WaypointTask(
        cessna172P,
        my_assessor,
        dummy_waypoints,
    )

register(
    id="JSBSim-WaypointTask-Cessna172P-Shaping.STANDARD-NoFG-v0",
    entry_point="gymnasium_jsbsim.environment:NoFGJsbSimEnv", 
    kwargs={"task_type": WaypointTaskFactory, "aircraft": cessna172P, "shaping": Shaping.STANDARD}
)

class SuccessBasedCurriculumCallback(BaseCallback):
    """Increases difficulty if the agent succeeds, lowers it if it crashes."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_difficulty = 0.0

    def _on_step(self) -> bool:
        # we only want to evaluate and change difficulty at the END of an episode
        if "dones" in self.locals and self.locals["dones"][0]:
            if len(self.model.ep_info_buffer) > 20:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                
                # # lowered from 700 to 350
                # if mean_reward > 350.0:
                #     self.current_difficulty = min(1.0, self.current_difficulty + 0.01)
                # # raised the failure floor slightly
                # elif mean_reward < 100.0:
                #     self.current_difficulty = max(0.0, self.current_difficulty - 0.02)

            for env in self.training_env.envs:
                env.unwrapped.task.curriculum_difficulty = self.current_difficulty
                
            self.logger.record("curriculum/difficulty", self.current_difficulty)
            
        return True

import argparse

if __name__ == "__main__":
    # set up the argument parser
    parser = argparse.ArgumentParser(description="Train a PPO agent for JSBSim Waypoint Navigation.")
    
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .zip file to resume training from. If not provided, starts fresh.")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total number of timesteps to run. Default: 1,000,000")
    parser.add_argument("--diff", type=float, default=0.0,
                        help="Initial curriculum difficulty (0.0 to 1.0). Default: 0.0")
    
    args = parser.parse_args()

    # initialize environment
    env = MinMaxObservationWrapper(gym.make("JSBSim-WaypointTask-Cessna172P-Shaping.STANDARD-NoFG-v0"))
    
    curriculum_callback = SuccessBasedCurriculumCallback()
    curriculum_callback.current_difficulty = args.diff

    # branching logic based on CLI arguments
    if args.resume:
        print(f"--- RESUMING TRAINING FROM: {args.resume} ---")
        model = PPO.load(
            args.resume, 
            env=env, 
            tensorboard_log="../logs/ppo",
            custom_objects={"learning_rate": 1e-4} 
        )
        reset_timesteps = False
        save_prefix = "resumed_flight_model"
    else:
        print("--- STARTING FRESH TRAINING RUN ---")
        model = PPO("MlpPolicy", env, tensorboard_log="../logs/ppo", learning_rate=1e-4, device='cpu')
        reset_timesteps = True
        save_prefix = "flight_model"

    # callbacks & execution
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="../models/checkpoints/",
        name_prefix=save_prefix
    )

    all_callbacks = CallbackList([curriculum_callback, checkpoint_callback])

    print(f"Target Timesteps: {args.steps} | Starting Difficulty: {args.diff}")
    
    model.learn(total_timesteps=args.steps, callback=all_callbacks, progress_bar=True, reset_num_timesteps=reset_timesteps)
    
    model.save("../models/ppo_final_model")
