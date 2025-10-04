# train_simple_agent.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from satellite_env import SimpleDelegationEnv # <-- Import the new simple environment

# --- 1. Create and check the custom environment ---
env = SimpleDelegationEnv(n_workers=3, n_tasks=3)
# It's good practice to check that your environment follows the gym API
check_env(env)

# --- 2. Define the PPO model ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.1
)

# --- 3. Train the model ---
print("Starting agent training...")
model.learn(total_timesteps=30000)
print("Training complete.")

# --- 4. Use the trained model to make decisions ---
print("\n--- Using the trained agent ---")
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    print(f"Agent action: Assign Task {action[0]} to Worker {action[1]}")
    obs, reward, done, _, info = env.step(action)
    total_reward += reward

print(f"\nEpisode finished. Total reward: {total_reward}")