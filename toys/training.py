import gymnasium as gym
import gymnasium_jsbsim
from stable_baselines3 import PPO
import os

# 1. Setup the Environment
# Recommendation: Use NoFG for training (faster), then switch to FG to watch
env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")

# 2. Path to your existing brain
model_path = "cessna_ppo_model_night_largo.zip"

if os.path.exists(model_path):
    print(f"Found existing model: {model_path}. Loading for further training...")
    # Load the model and link it to the current environment
    model = PPO.load(model_path, env=env)
else:
    print("Model not found! Starting training from scratch.")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.000025)

# 3. Continue Training
# reset_num_timesteps=False ensures the total step count doesn't restart at 0
print("Continuing training... Refining the flight skills.")
model.learn(total_timesteps=5000000, reset_num_timesteps=False)

# 4. Save the "Updated Brain"
model.save("cessna_ppo_model_refined")
print("Model saved as 'cessna_ppo_model_refined'!")