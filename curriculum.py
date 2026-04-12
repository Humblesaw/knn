import os
import gymnasium as gym
import gymnasium_jsbsim
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from curriculum_tasks import Stage1LevelHeading, Stage2RandomHeading, Stage3AltitudeHeading, Stage4Waypoints

MODEL_DIR = "models"
LOG_DIR = "logs/curriculum"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODE = "train"  # Options: "train" or "vis"
STAGE = 1       # 1 = level heading, 2 = random heading, 3 = altitude+heading, 4 = waypoints
ALGORITHM = "PPO"  # PPO for stages 1-3, consider SAC for stage 4 (off-policy, better exploration)
MODEL_NAME = f"stage_{STAGE}_{ALGORITHM.lower()}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
LOAD_PREV_STAGE = True
PREV_STAGE = STAGE - 1 if STAGE > 1 else None
PREV_MODEL_NAME = f"stage_{PREV_STAGE}_{ALGORITHM.lower()}" if PREV_STAGE else None
PREV_MODEL_PATH = os.path.join(MODEL_DIR, PREV_MODEL_NAME) if PREV_MODEL_NAME else None

TIMESTEPS_PER_STAGE = {
    1: 800_000,
    2: 1_000_000,
    3: 1_500_000,
    4: 2_000_000,  # Waypoint navigation requires more exploration
}

STAGE_LEARNING_RATE = {
    1: 3e-4,
    2: 2.5e-4,
    3: 2e-4,
    4: 1.5e-4,
}

STAGE_ENT_COEF = {
    1: 0.015,
    2: 0.01,
    3: 0.007,
    4: 0.005,
}


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def copy_compatible_policy_weights(source_model: PPO, target_model: PPO):
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    compatible_keys = [
        key for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    ]
    if compatible_keys:
        for key in compatible_keys:
            target_state[key] = source_state[key]
        target_model.policy.load_state_dict(target_state)
        print(f"Transferred {len(compatible_keys)} compatible policy parameters from saved model.")
    else:
        print("No compatible policy parameters found to transfer. Starting from scratch.")


def build_model_with_transferred_brain(env, saved_model_path: str, learning_rate: float, ent_coef: float):
    print(f"Attempting compatible brain transfer from {saved_model_path}.zip")
    source_model = PPO.load(saved_model_path, device="auto")
    target_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=linear_schedule(learning_rate),
        ent_coef=ent_coef,
        gamma=0.99,
        batch_size=256,
        n_steps=2048,
        clip_range=0.2,
    )

    copy_compatible_policy_weights(source_model, target_model)
    return target_model


def make_env(gui: bool = False):
    env_name = (
        "JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0"
        if gui
        else "JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0"
    )
    base_env = gym.make(env_name)
    if STAGE == 1:
        return Stage1LevelHeading(base_env)
    if STAGE == 2:
        return Stage2RandomHeading(base_env)
    if STAGE == 3:
        return Stage3AltitudeHeading(base_env)
    if STAGE == 4:
        return Stage4Waypoints(base_env)
    raise ValueError(f"Unsupported STAGE: {STAGE}. Choose 1, 2, 3, or 4.")

def train():
    set_random_seed(42)
    env = make_env(gui=False)

    learning_rate = STAGE_LEARNING_RATE.get(STAGE, 2e-4)
    ent_coef = STAGE_ENT_COEF.get(STAGE, 0.005)
    total_timesteps = TIMESTEPS_PER_STAGE.get(STAGE, 1_000_000)

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH}.zip")
        try:
            model = PPO.load(MODEL_PATH, env=env, device="auto", tensorboard_log=LOG_DIR)
        except Exception as e:
            print(f"Failed to load existing model directly: {e}")
            model = build_model_with_transferred_brain(env, MODEL_PATH, learning_rate, ent_coef)
    elif LOAD_PREV_STAGE and PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH + ".zip"):
        print(f"Loading previous stage model from {PREV_MODEL_PATH}.zip as the brain for stage {STAGE}")
        try:
            model = PPO.load(PREV_MODEL_PATH, env=env, device="auto", tensorboard_log=LOG_DIR)
        except Exception as e:
            print(f"Failed to load previous stage model directly: {e}")
            model = build_model_with_transferred_brain(env, PREV_MODEL_PATH, learning_rate, ent_coef)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=linear_schedule(learning_rate),
            ent_coef=ent_coef,
            gamma=0.99,
            batch_size=256,
            n_steps=2048,
            clip_range=0.2,
        )

    print(f"Training Stage {STAGE} using {ALGORITHM} for {total_timesteps} timesteps")
    print(f"View progress with: tensorboard --logdir={LOG_DIR}")
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"stage_{STAGE}")
    model.save(MODEL_PATH)
    env.close()


def visualize():
    env = make_env(gui=True)
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}.zip for visualization.")

    print(f"Visualizing model: {MODEL_PATH}.zip")
    model = PPO.load(MODEL_PATH, env=env)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    history = {'roll': [], 'pitch': [], 'heading': [], 'target': []}

    state, info = env.reset()
    try:
        for step in range(100000):
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)

            raw = env.unwrapped.sim.jsbsim
            r = np.degrees(raw.get_property_value('attitude/roll-rad'))
            p = np.degrees(raw.get_property_value('attitude/pitch-rad'))
            h = raw.get_property_value('attitude/psi-deg')
            t = raw.get_property_value('guidance/target-heading-deg') if STAGE in {2, 3, 4} else 0.0

            history['roll'].append(r)
            history['pitch'].append(p)
            history['heading'].append(h)
            history['target'].append(t)
            for key in history:
                if len(history[key]) > 150:
                    history[key].pop(0)

            if step % 5 == 0:
                ax1.clear()
                ax1.plot(history['roll'], color='red', label=f"Roll: {r:.1f}°")
                ax1.plot(history['pitch'], color='green', label=f"Pitch: {p:.1f}°")
                ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
                ax1.set_ylim(-40, 40)
                ax1.set_title("Attitude Stability")
                ax1.legend(loc='upper right')

                ax2.clear()
                ax2.plot(history['heading'], color='blue', label=f"Heading: {h:.1f}°")
                if STAGE in {2, 3, 4}:
                    ax2.plot(history['target'], color='orange', linestyle='--', label=f"Target: {t:.1f}°")
                ax2.set_ylim(0, 360)
                ax2.set_title("Navigation")
                ax2.legend(loc='upper right')
                plt.pause(0.001)

            env.unwrapped.render(mode="flightgear")

            if terminated or truncated:
                state, info = env.reset()
                for key in history:
                    history[key].clear()

    except KeyboardInterrupt:
        print("Closing Visualizer...")
    finally:
        env.close()


if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "vis":
        visualize()
    else:
        raise ValueError(f"Unsupported MODE: {MODE}. Use 'train' or 'vis'.")
