# simple_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SimpleDelegationEnv(gym.Env):
    """
    A simplified, robust environment for task delegation.

    - State: [task_features..., worker_features...]
    - Action: Assign a specific task to a specific worker.
    - Reward: Based on completing high-priority tasks on time.
    """
    def __init__(self, n_workers=3, n_tasks=5):
        super().__init__()
        self.n_workers = n_workers
        self.n_tasks = n_tasks
        self.max_steps = n_tasks * 5 # Max episode length

        # Action: Choose one task and one worker
        self.action_space = spaces.MultiDiscrete([self.n_tasks, self.n_workers])

        # State: Flattened array of all task and worker features
        # Task features: [priority, time_needed, deadline_remaining]
        # Worker features: [is_available (1 or 0), time_until_free]
        task_features = 3
        worker_features = 2
        obs_shape = (self.n_tasks * task_features) + (self.n_workers * worker_features)
        
        # We define low and high bounds for our state values
        # This is crucial for normalization and for the learning algorithm
        low_bounds = np.zeros(obs_shape, dtype=np.float32)
        high_bounds = np.full(obs_shape, 100.0, dtype=np.float32) # A high value

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

    def _get_obs(self):
        """Constructs the observation vector from the environment's state."""
        # Create a view of tasks for modification
        tasks_view = self.tasks.copy()
        
        # Convert absolute deadline to relative "time remaining"
        # This is much more useful for the agent
        tasks_view[:, 2] = tasks_view[:, 2] - self.current_step
        
        tasks_flat = tasks_view[:, :3].flatten() # Use only the first 3 columns for the obs
        workers_flat = self.workers.flatten()
        
        return np.concatenate([tasks_flat, workers_flat])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # --- Initialize Tasks ---
        # [priority, time_needed, deadline (absolute), status (0:pending, 1:done)]
        self.tasks = np.zeros((self.n_tasks, 4), dtype=np.float32)
        for i in range(self.n_tasks):
            self.tasks[i] = [
                self.np_random.integers(1, 6),       # priority
                self.np_random.integers(5, 16),      # time_needed
                self.np_random.integers(40, 100),     # deadline
                0                                    # status
            ]

        # --- Initialize Workers ---
        # [is_available (1 or 0), time_until_free]
        self.workers = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.workers[:, 0] = 1.0 # All workers start available

        return self._get_obs(), {}

    # In simple_env.py, replace the entire step function with this:

    def step(self, action):
        task_idx, worker_idx = action
        reward = -0.1 # A tiny penalty per step to encourage finishing faster

        task = self.tasks[task_idx]
        worker = self.workers[worker_idx]

        # --- Apply Penalties for Invalid Actions ---
        is_task_pending = (task[3] == 0)
        is_worker_available = (worker[0] == 1)

        if not is_task_pending:
            reward -= 10 # MODIFIED: Reduced penalty
        elif not is_worker_available:
            reward -= 10 # MODIFIED: Reduced penalty
        else:
            # --- Valid Action: Assign Task ---
            # No immediate reward, the reward comes from completion.
            task_time_needed = task[1]

            # Update worker state
            self.workers[worker_idx, 0] = 0 # Becomes unavailable
            self.workers[worker_idx, 1] = task_time_needed # Set time until free
            
            # Store the absolute step when the task will be completed
            self.tasks[task_idx, 3] = self.current_step + task_time_needed

        # --- Update Environment State Over Time ---
        self.current_step += 1

        for i in range(self.n_workers):
            if self.workers[i, 0] == 0: # If worker is busy
                self.workers[i, 1] -= 1 # Decrement time
                if self.workers[i, 1] <= 0:
                    self.workers[i, 0] = 1 # Worker becomes free

        # --- Calculate Rewards for Task Status ---
        all_tasks_now_handled = True
        for i in range(self.n_tasks):
            task_status = self.tasks[i, 3]
            
            if task_status == 0: # If task is still pending
                all_tasks_now_handled = False
                # Check if a pending task has missed its deadline
                if self.current_step > self.tasks[i, 2]:
                    reward -= 50 # Penalty for a deadline miss
                    self.tasks[i, 3] = -1 # Mark as failed
            
            elif task_status > 0 and self.current_step >= task_status:
                # This task was just completed in this step
                priority = self.tasks[i, 0]
                reward += 100 + priority * 20 # BIG reward for on-time completion
                self.tasks[i, 3] = -2 # Mark as completed successfully
        
        # --- Check for Termination ---
        terminated = all_tasks_now_handled or (self.current_step >= self.max_steps)

        return self._get_obs(), reward, terminated, False, {}