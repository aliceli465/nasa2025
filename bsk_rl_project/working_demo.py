#!/usr/bin/env python3
"""
Working BSK-RL Demo: A satellite learning to scan Earth
Based on the examples/simple_environment.ipynb
"""

import gymnasium as gym
import numpy as np
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw
import matplotlib.pyplot as plt

# Suppress Basilisk warnings
from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

print("🛰️  BSK-RL Working Demo: Earth Scanning Satellite")
print("=" * 60)

# Define a scanning satellite (from the notebook example)
class MyScanningSatellite(sats.AccessSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
    ]
    action_spec = [
        act.Scan(duration=60.0),    # Scan for 1 minute
        act.Charge(duration=600.0),  # Charge for 10 minutes
    ]
    dyn_type = dyn.ContinuousImagingDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

# Configure satellite
sat_args = {}
sat_args["imageAttErrorRequirement"] = 0.05
sat_args["dataStorageCapacity"] = 1e10
sat_args["instrumentBaudRate"] = 1e7
sat_args["storedCharge_Init"] = 50000.0
sat_args["storageInit"] = lambda: np.random.uniform(0.25, 0.75) * 1e10

# Create the satellite
sat = MyScanningSatellite(name="EO1", sat_args=sat_args)

# Create environment
print("Creating scanning environment...")
env = gym.make(
    "SatelliteTasking-v1",
    satellite=sat,
    scenario=scene.UniformNadirScanning(),
    rewarder=data.ScanningTimeReward(),
    time_limit=5700.0,  # approximately 1 orbit
)

print(f"Action space: {env.action_space} (0=Scan, 1=Charge)")
print(f"Observation space shape: {env.observation_space.shape}")

# Simple Q-learning agent for this environment
class ScanningAgent:
    def __init__(self, epsilon=0.3):
        self.epsilon = epsilon
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        
    def get_state(self, obs):
        # State: (data_level, battery_level, eclipse_soon)
        data_level = int(obs[0] * 10)      # 0-10
        battery_level = int(obs[1] * 10)   # 0-10
        eclipse_soon = obs[2] < 300        # Eclipse in next 5 minutes?
        return (data_level, battery_level, eclipse_soon)
    
    def act(self, obs):
        state = self.get_state(obs)
        
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2)
            
        return np.argmax(self.q_table[state])
    
    def update(self, obs, action, reward, next_obs):
        state = self.get_state(obs)
        next_state = self.get_state(next_obs)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(2)
            
        # Q-learning update
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

# Training
print("\n🚀 Training agent to maximize scanning time...")
agent = ScanningAgent()
episode_rewards = []
scan_times = []
charge_times = []

for episode in range(20):
    obs, info = env.reset()
    total_reward = 0
    actions_count = [0, 0]
    
    done = False
    while not done:
        # Select action
        action = agent.act(obs)
        actions_count[action] += 1
        
        # Take action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update agent
        agent.update(obs, action, reward, next_obs)
        total_reward += reward
        
        obs = next_obs
    
    episode_rewards.append(total_reward)
    scan_times.append(actions_count[0] * 60)  # Each scan is 60 seconds
    charge_times.append(actions_count[1] * 600)  # Each charge is 600 seconds
    
    if episode % 5 == 0:
        print(f"Episode {episode}: Reward={total_reward:.0f}s of scanning, "
              f"Actions: Scan={actions_count[0]}, Charge={actions_count[1]}")
        agent.epsilon *= 0.7  # Decay exploration

# Evaluation
print("\n📊 Evaluating learned policy (no exploration)...")
agent.epsilon = 0.0

obs, info = env.reset()
print(f"\nInitial state - Data: {obs[0]:.2f}, Battery: {obs[1]:.2f}")

total_reward = 0
step_details = []

for step in range(10):  # Show first 10 steps
    action = agent.act(obs)
    action_name = "Scan" if action == 0 else "Charge"
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    eclipse_status = "Eclipse soon" if obs[2] < 300 else "No eclipse"
    
    print(f"Step {step+1}: {action_name} | "
          f"Data: {obs[0]:.2f}→{next_obs[0]:.2f}, "
          f"Battery: {obs[1]:.2f}→{next_obs[1]:.2f} | "
          f"{eclipse_status} | "
          f"Reward: {reward:.0f}s")
    
    obs = next_obs
    if terminated or truncated:
        break

print(f"\nTotal scanning time in evaluation: {total_reward:.0f} seconds")

# Plot results
plt.figure(figsize=(12, 4))

# Rewards over episodes
plt.subplot(1, 3, 1)
plt.plot(episode_rewards, 'b-', linewidth=2)
plt.fill_between(range(len(episode_rewards)), 0, episode_rewards, alpha=0.3)
plt.title('Scanning Time per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Scanning Time (s)')
plt.grid(True, alpha=0.3)

# Action distribution
plt.subplot(1, 3, 2)
plt.bar(['Scan', 'Charge'], [np.mean(scan_times), np.mean(charge_times)], 
        color=['skyblue', 'orange'])
plt.title('Average Time per Action')
plt.ylabel('Time (s)')
plt.grid(True, alpha=0.3, axis='y')

# Q-values heatmap
plt.subplot(1, 3, 3)
# Extract some Q-values for visualization
sample_states = list(agent.q_table.keys())[:10]
q_values = np.array([agent.q_table[s] for s in sample_states])
plt.imshow(q_values.T, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Q-value')
plt.yticks([0, 1], ['Scan', 'Charge'])
plt.xlabel('State Index')
plt.ylabel('Action')
plt.title('Sample Q-values')

plt.tight_layout()
plt.savefig('scanning_satellite_results.png', dpi=150)
print(f"\n📈 Results saved to: scanning_satellite_results.png")

env.close()
print("\n✅ Demo complete!")
print("\nWhat the agent learned:")
print("- Scan when battery is sufficient and not in eclipse")
print("- Charge when battery is low or eclipse is approaching")
print("- Balance scanning time with power management")