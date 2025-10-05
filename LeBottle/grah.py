import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class SatelliteSchedulerEnv(gym.Env):
    """
    Satellite Task Scheduling Environment
    
    The agent assigns tasks to satellites to maximize reward.
    Reward = priority - (step_count * 0.01) when task completes
    """
    
    def __init__(self, num_satellites=3, num_tasks=15):
        super(SatelliteSchedulerEnv, self).__init__()
        
        self.num_satellites = num_satellites
        self.initial_num_tasks = num_tasks
        
        # Fixed task features
        self.task_priority = 100
        self.task_size = 10
        
        # Simplified action space: 
        # 0 = do nothing
        # 1 to num_satellites = assign next task to satellite i
        self.action_space = spaces.Discrete(1 + num_satellites)
        
        # Observation space (normalized):
        # - Step count / 100
        # - Number of tasks remaining / initial_num_tasks
        # - For each satellite: availability (0/1), time_remaining / task_size
        obs_size = 2 + (num_satellites * 2)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.total_reward = 0
        self.completed_tasks = 0
        
        # Initialize satellites: [available (0/1), time_remaining, assigned_task_wait_time]
        self.satellites = [[1, 0, 0] for _ in range(self.num_satellites)]
        
        # Initialize tasks: [priority, size, wait_time (steps)]
        self.tasks = [[self.task_priority, self.task_size, 0] 
                      for _ in range(self.initial_num_tasks)]
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Convert current state to observation vector (normalized)"""
        obs = [
            self.step_count / 100.0,  # Normalized step count
            len(self.tasks) / self.initial_num_tasks  # Fraction of tasks remaining
        ]
        
        # Satellite states (normalized)
        for sat in self.satellites:
            obs.append(float(sat[0]))  # available (0 or 1)
            obs.append(sat[1] / self.task_size)  # normalized time_remaining
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        self.step_count += 1
        step_reward = 0
        invalid_action = False
        
        # Increment wait time for all tasks
        for task in self.tasks:
            task[2] += 1
        
        # Parse and execute action
        if action > 0 and len(self.tasks) > 0:
            # Action 1 to num_satellites: assign first task to satellite (action-1)
            sat_idx = action - 1
            
            # Check if action is valid
            if sat_idx < self.num_satellites and self.satellites[sat_idx][0] == 1:
                # Assign first task to satellite
                task = self.tasks[0]
                self.satellites[sat_idx][0] = 0  # make unavailable
                self.satellites[sat_idx][1] = task[1]  # set time_remaining to task size
                self.satellites[sat_idx][2] = task[2]  # store wait_time for reward calculation
                
                # Remove task from list
                self.tasks.pop(0)
                
                # Small reward for taking a valid action
                step_reward += 0.5
            elif sat_idx < self.num_satellites and self.satellites[sat_idx][0] == 0:
                # Penalty for trying to assign to a busy satellite
                step_reward -= 0.5
                invalid_action = True
        
        # Small positive reward for each satellite being used
        num_busy_satellites = sum(1 for sat in self.satellites if sat[0] == 0)
        step_reward += num_busy_satellites * 0.2
        
        # Small negative reward for each task waiting (encourage urgency)
        step_reward -= len(self.tasks) * 0.1
        
        # Update satellites and calculate rewards for completed tasks
        for sat in self.satellites:
            if sat[0] == 0:  # satellite is busy
                sat[1] -= 1  # decrement time_remaining
                
                if sat[1] == 0:  # task completed
                    # Calculate reward: priority - (wait_time * 0.01)
                    wait_time = sat[2]
                    reward = self.task_priority - (wait_time * 0.01)
                    step_reward += reward
                    self.completed_tasks += 1
                    
                    # Make satellite available again
                    sat[0] = 1
                    sat[2] = 0
        
        self.total_reward += step_reward
        
        # Episode ends when all tasks are assigned and completed
        all_tasks_assigned = len(self.tasks) == 0
        all_satellites_idle = all(sat[0] == 1 for sat in self.satellites)
        terminated = all_tasks_assigned and all_satellites_idle
        
        # Also end if episode is too long
        truncated = self.step_count > 200
        
        # Add bonus for completing episode efficiently
        if terminated:
            efficiency_bonus = max(0, (200 - self.step_count) * 0.5)
            step_reward += efficiency_bonus
        
        return self._get_obs(), step_reward, terminated, truncated, {}


class RewardCallback(BaseCallback):
    """Callback to track and plot training progress"""
    
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_count += 1
            # Get the episode info
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                # Print progress every 50 episodes
                if self.episode_count % 50 == 0:
                    recent_rewards = self.episode_rewards[-50:]
                    avg_reward = np.mean(recent_rewards)
                    print(f"Episode {self.episode_count}: Avg Reward (last 50) = {avg_reward:.2f}")
        return True


# Create environment
print("Creating Satellite Scheduler Environment...")
env = SatelliteSchedulerEnv(num_satellites=3, num_tasks=15)

# Test the environment
print("\nTesting environment...")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Number of tasks: {len(env.tasks)}")
print(f"Number of satellites: {env.num_satellites}")

# Create and train the PPO agent
print("\n" + "="*50)
print("Training PPO Agent...")
print("="*50)

callback = RewardCallback()

# Improved hyperparameters
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,  # Increased for more exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[128, 128])  # Larger network
)

# Train the model for longer
print("\nStarting training for 200,000 steps...")
model.learn(total_timesteps=200000, callback=callback, progress_bar=True)

print("\n" + "="*50)
print("Training Complete!")
print("="*50)

# Plot training progress
if callback.episode_rewards:
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(callback.episode_rewards, alpha=0.6, linewidth=0.5)
    # Add moving average
    window = 50
    if len(callback.episode_rewards) >= window:
        moving_avg = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(callback.episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (50)')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress: Episode Rewards')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(callback.episode_lengths, alpha=0.6, linewidth=0.5)
    if len(callback.episode_lengths) >= window:
        moving_avg = np.convolve(callback.episode_lengths, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(callback.episode_lengths)), moving_avg, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length (steps)')
    plt.title('Training Progress: Episode Length')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if len(callback.episode_rewards) >= window:
        moving_avg = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (50 episodes)')
        plt.title('Learning Curve')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("\nTraining progress plot saved as 'training_progress.png'")

# Test the trained agent multiple times
print("\n" + "="*50)
print("Testing Trained Agent (5 episodes)...")
print("="*50)

trained_rewards = []
trained_steps = []

for episode in range(5):
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        
        if terminated or truncated:
            trained_rewards.append(episode_reward)
            trained_steps.append(step)
            print(f"Episode {episode+1}: Reward: {episode_reward:.2f}, Steps: {step}")
            break

avg_trained_reward = np.mean(trained_rewards)
avg_trained_steps = np.mean(trained_steps)

# Compare with random policy (multiple episodes)
print("\n" + "="*50)
print("Testing Random Policy (5 episodes)...")
print("="*50)

random_rewards = []
random_steps = []

for episode in range(5):
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        
        if terminated or truncated:
            random_rewards.append(episode_reward)
            random_steps.append(step)
            print(f"Episode {episode+1}: Reward: {episode_reward:.2f}, Steps: {step}")
            break

avg_random_reward = np.mean(random_rewards)
avg_random_steps = np.mean(random_steps)

print("\n" + "="*50)
print("Performance Comparison")
print("="*50)
print(f"Trained Policy - Avg Reward: {avg_trained_reward:.2f}, Avg Steps: {avg_trained_steps:.1f}")
print(f"Random Policy  - Avg Reward: {avg_random_reward:.2f}, Avg Steps: {avg_random_steps:.1f}")

if avg_trained_reward > avg_random_reward:
    improvement = ((avg_trained_reward - avg_random_reward) / abs(avg_random_reward) * 100)
    print(f"\n✓ Trained policy is {improvement:.1f}% better than random!")
else:
    print(f"\n✗ Trained policy needs improvement. Continue training or adjust hyperparameters.")

print("\nModel saved as 'ppo_satellite_scheduler'")
model.save("ppo_satellite_scheduler")