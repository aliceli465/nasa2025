from stable_baselines3 import PPO
from bleh_env import SatelliteTaskEnv

env = SatelliteTaskEnv(num_tasks=10, num_sats=3)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Test the trained agent
obs = env.reset()
done = False
total_reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print("Total reward:", total_reward)
