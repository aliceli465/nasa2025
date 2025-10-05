# Author: Nathan Hu 
# Date: 10/5/2025

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Constants
RADIUS = 4000  # km
MIN_ALTITUDE = 500   # km
MAX_ALTITUDE = 5000  # km
NUM_SATELLITES = 5
BATCH_SIZE = 10 #number of tasks to be processed at a time
MAX_QUEUE_SIZE = 30
TIME_STEP = 1  # Time per step in simulation units

class SatelliteTaskEnv(gym.Env):
    """Custom Environment for Satellite Task Assignment"""
    
    def __init__(self):
        super().__init__()
        
        self.num_satellites = NUM_SATELLITES
        self.batch_size = BATCH_SIZE
        self.max_queue_size = MAX_QUEUE_SIZE
        
        self.satellites = []

        # Agent location (fixed on Earth's surface)
        self.agent_pos = np.array([RADIUS, 0, 0]) 
        
        # Action space: Choose task (0 to queue_size-1 or -1 for no-op) and satellite (0 to num_satellites-1)
        self.action_space = spaces.Discrete((MAX_QUEUE_SIZE + 1) * NUM_SATELLITES)
        
        # Observation space: satellite positions (vector3), satellite availability (0 or 1)
        # task queue (priority + size x max_queue), task wait times
        obs_dim = (NUM_SATELLITES * 4 +  # x,y,z position of satellite + availability
                   MAX_QUEUE_SIZE * 3)    # priority + size + wait_time for each task
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(obs_dim,), dtype=np.float32)
        
        # Initialize state
        self.reset()
    
    def _init_satellites(self):
        """Initialize satellite positions and orbital parameters"""
        satellites = []
        for i in range(self.num_satellites):
            altitude = np.random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
            orbital_radius = RADIUS + altitude
            
            # Random orbital inclination and phase
            inclination = np.random.uniform(0, np.pi) 
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Angular velocity 
            angular_velocity = 0.1 / (orbital_radius / RADIUS)
            
            satellites.append({
                'orbital_radius': orbital_radius,
                'inclination': inclination,
                'phase': phase,
                'angular_velocity': angular_velocity,
                'current_angle': np.random.uniform(0, 2 * np.pi), #random angle
                'available': True,
                'task_remaining': 0.0,
                'position': np.zeros(3) 
            })
        
        self._update_satellite_positions()
        return satellites
    
    def _update_satellite_positions(self):
        """Update satellite positions based on orbital mechanics"""
        for sat in self.satellites:
            angle = sat['current_angle']
            inc = sat['inclination']
            r = sat['orbital_radius']
            
            # Position in orbital plane
            x = r * np.cos(angle)
            y = r * np.sin(angle) * np.cos(inc)
            z = r * np.sin(angle) * np.sin(inc)
            
            sat['position'] = np.array([x, y, z]) - self.agent_pos #RELATIVE POSITION to the agent
            sat['current_angle'] += sat['angular_velocity'] * TIME_STEP
            sat['current_angle'] %= (2 * np.pi)
    
    def _generate_tasks(self, num_tasks):
        """Generate new tasks with priority and size"""
        for _ in range(num_tasks):
            if len(self.task_queue) < self.max_queue_size:
                task = {
                    'priority': np.random.uniform(1, 100),
                    'size': np.random.uniform(0.5, 5.0),  # Size of task affects duration
                    'wait_time': 0.0
                }
                self.task_queue.append(task)
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return float(np.linalg.norm(pos1 - pos2))
    
    def _calculate_task_duration(self, task_size, distance):
        """Calculate how long a task takes based on size and distance"""
        # Duration increases with size and distance
        base_duration = float(task_size)
        distance_factor = 1 + (float(distance) / 10000)  # Normalize distance impact
        return base_duration * distance_factor
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.satellites = self._init_satellites()
        self.task_queue = []
        self._generate_tasks(self.batch_size)
        
        self.current_step = 0
        self.total_reward = 0
        self.completed_tasks = 0
        
        return self._get_observation(), {}

    #Essentially prevents agent from assigning a task to an invalid satellite 
    def get_action_mask(self): 
        """Return a binary mask of valid actions (1 = valid, 0 = invalid)."""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
    
        for task_idx in range(len(self.task_queue)):
            for sat_idx in range(self.num_satellites):
                # Skip satellites that are currently unavailable
                if not self.satellites[sat_idx]['available']:
                    continue

                # Flatten 
                action_idx = task_idx * self.num_satellites + sat_idx
                mask[action_idx] = 1.0

        #if np.sum(mask) == 0:
            #print("# Valid actions: {}".format(np.sum(mask)))

        return mask
        
    def _get_observation(self):
        """Generate observation vector"""
        obs = []
        
        # Satellite data: position (3) + availability (1)
        for sat in self.satellites:
            obs.extend(sat['position'])
            obs.append(1.0 if sat['available'] else 0.0)
        
        # Task queue data: priority + size + wait_time
        for i in range(self.max_queue_size):
            if i < len(self.task_queue):
                task = self.task_queue[i]
                obs.append(task['priority'])  
                obs.append(task['size'])       
                obs.append(task['wait_time'])  
            else:
                obs.extend([0.0, 0.0, 0.0])  # Padding
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):

        # Decode action
        task_idx = action // self.num_satellites
        sat_idx = action % self.num_satellites
        
        reward = 0
        task_assigned = False
        
        # Check if valid action
        if task_idx < len(self.task_queue) and self.satellites[sat_idx]['available']:
            task = self.task_queue[task_idx]
            sat = self.satellites[sat_idx]

            if sat['available']:
                # Calculate distance and duration
                distance = float(np.linalg.norm(sat['position'])) #Relative distance from 
                duration = self._calculate_task_duration(task['size'], distance)
                
                # Assign task
                sat['available'] = False
                sat['task_remaining'] = duration
                
                # Calculate reward: priority - wait_time_penalty
                wait_penalty = task['wait_time'] * 0.01
                reward = max(0, task['priority'] - wait_penalty)
                
                # Remove task from queue
                self.task_queue.pop(task_idx)
                self.completed_tasks += 1
                task_assigned = True
            
    
        # Update simulation
        self._update_satellite_positions()
        
        # Update satellite tasks and availability
        for sat in self.satellites:
            if not sat['available']:
                sat['task_remaining'] -= TIME_STEP
                if sat['task_remaining'] <= 0:
                    sat['available'] = True
                    sat['task_remaining'] = 0
        
        # Update task wait times
        for task in self.task_queue:
            task['wait_time'] += TIME_STEP
        
        # Add new tasks when batch is complete
        if self.completed_tasks > 0 and self.completed_tasks % self.batch_size == 0:
            self._generate_tasks(self.batch_size)
        
        self.current_step += 1
        self.total_reward += reward
        
        # Episode ends after many steps or if queue gets too large
        terminated = self.current_step >= 1000
        truncated = len(self.task_queue) >= self.max_queue_size
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def get_state_for_visualization(self):
        """Return current state for visualization"""
        return {
            'satellites': [{
                'position': sat['position'].copy(),
                'available': sat['available']
            } for sat in self.satellites],
            'agent_pos': self.agent_pos.copy(),
            'queue_size': len(self.task_queue),
            'completed': self.completed_tasks
        }


class PPONetwork(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)


class PPOAgent:
    """PPO Agent for training"""
    
    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy = PPONetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': []
        }
    
    def select_action(self, state, mask=None, training=True):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, _ = self.policy(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1).squeeze(0)
        
        # Apply mask -
        #This is necessary because the agent should not be able to assign a task to an unavailable satellite!! 
        if mask is not None:
            mask_tensor = torch.FloatTensor(mask)
            action_probs *= mask_tensor

            total = action_probs.sum()
            if total.item() == 0 or torch.isnan(total):
                #if all actions are invalid, choose randomly from valid ones
                valid_actions = torch.nonzero(mask_tensor).squeeze()
                if valid_actions.ndim == 0:
                    # convert tensor(5) → tensor([5])
                    valid_actions = valid_actions.unsqueeze(0)

                if valid_actions.numel() == 0:
                    # No valid actions at all so pick random action
                    valid_actions = torch.arange(len(mask_tensor))
                random_action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                return random_action.item(), 0.0

            # Renormalize
            action_probs /= total + 1e-8

        if training:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()
        else:
            return torch.argmax(action_probs).item(), 0
    
    def store_transition(self, state, action, reward, done, log_prob):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['log_probs'].append(log_prob)
    
    def update(self):
        """Update policy using PPO"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.FloatTensor(self.memory['log_probs'])
        
        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            action_logits, values = self.policy(states)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Ratio and surrogate loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = returns - values.squeeze()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * advantages.pow(2).mean()
            entropy_loss = -0.01 * entropy.mean()
            
            loss = actor_loss + critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        for key in self.memory:
            self.memory[key] = []


def train_agent(num_episodes=1000, update_freq=2048):
    """Train the PPO agent"""
    env = SatelliteTaskEnv()
    agent = PPOAgent(env, lr=1e-4)
    
    episode_rewards = []
    steps = 0
    
    print("Training PPO agent...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            mask = env.get_action_mask()
            action, log_prob = agent.select_action(state, mask=mask, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, done, log_prob)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if steps % update_freq == 0:
                agent.update()
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}, Avg Reward (last 50): {avg_reward:.2f}")
    
    print("\nTraining complete!")
    return agent, env, episode_rewards


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate trained agent"""
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            mask = env.get_action_mask()
            action, _ = agent.select_action(state, mask=mask, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)


def evaluate_random_policy(env, num_episodes=10):
    """Evaluate random policy baseline"""
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    # Train agent
    agent, env, training_rewards = train_agent(num_episodes=500)
    
    # Evaluate trained agent
    trained_mean, trained_std = evaluate_agent(agent, env)
    print(f"\nTrained Agent - Mean Reward: {trained_mean:.2f} ± {trained_std:.2f}")
    
    # Evaluate random policy
    random_mean, random_std = evaluate_random_policy(env)
    print(f"Random Policy - Mean Reward: {random_mean:.2f} ± {random_std:.2f}")
    
    print(f"\nImprovement: {((trained_mean - random_mean) / random_mean * 100):.1f}%")
    
    # Save the trained agent
    torch.save(agent.policy.state_dict(), 'satellite_agent.pth')
    print("\nAgent saved to 'satellite_agent.pth'")