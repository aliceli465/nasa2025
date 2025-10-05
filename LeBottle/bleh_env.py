import gym
from gym import spaces
import numpy as np

class SatelliteTaskEnv(gym.Env):
    """Custom Environment for Task-Satellite assignment."""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_tasks=1000, num_sats=3):
        super(SatelliteTaskEnv, self).__init__()

        # Environment configuration
        self.num_tasks = num_tasks
        self.num_sats = num_sats
        self.task_size = 1     # fixed size
        self.task_priority = 1000 # fixed priority
        self.step_count = 0

        # Observation: tasks + satellites
        # Tasks: for each task -> [size, priority, waiting_time]
        # Satellites: for each satellite -> availability (0 or 1)
        self.max_tasks = num_tasks
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.max_tasks, 3 + self.num_sats),  # 3 task features + 1 satellite availability per satellite (flattened)
            dtype=np.float32
        )

        # Action: either do nothing (0), or assign task i to satellite j
        # We flatten task-satellite assignments into 1D discrete space
        self.num_assignments = self.num_tasks * self.num_sats
        self.action_space = spaces.Discrete(self.num_assignments + 1)  # +1 for "do nothing"

        self.reset()

    def reset(self):
        # Reset tasks: [size, priority, waiting_time]
        self.tasks = np.array([[self.task_size, self.task_priority, 0] for _ in range(self.num_tasks)], dtype=np.float32)

        # Satellites: [availability, remaining_task_time]
        self.sats = np.array([[1, 0] for _ in range(self.num_sats)], dtype=np.float32)

        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        # Tasks padded to max_tasks
        tasks_padded = np.zeros((self.max_tasks, 3), dtype=np.float32)
        tasks_padded[:len(self.tasks), :] = self.tasks

        # Satellites: just availability
        sats_avail = np.array([sat[0] for sat in self.sats], dtype=np.float32).reshape(1, -1)
        sats_avail_padded = np.repeat(sats_avail, self.max_tasks, axis=0)  # repeat for each task row
        obs = np.concatenate([tasks_padded, sats_avail_padded], axis=1)
        return obs

    def step(self, action):
        self.step_count += 1
        reward = 0

        # --- Update satellites: reduce remaining task time ---
        for i, sat in enumerate(self.sats):
            if sat[0] == 0:  # unavailable
                sat[1] -= 1
                if sat[1] <= 0:
                    sat[0] = 1  # becomes available

        # --- Handle action ---
        if action > 0:  # assign a task
            action -= 1  # because 0 = do nothing
            task_idx = action // self.num_sats
            sat_idx = action % self.num_sats

            if task_idx < len(self.tasks) and self.sats[sat_idx][0] == 1:  # valid assignment
                task = self.tasks[task_idx]
                # Assign task to satellite
                self.sats[sat_idx][0] = 0
                self.sats[sat_idx][1] = task[0]  # remaining task time = task size
                # Calculate reward
                reward = task[1] - (self.step_count * 0.01)
                # Remove task from task list
                self.tasks = np.delete(self.tasks, task_idx, axis=0)

        # --- Update waiting times for remaining tasks ---
        self.tasks[:, 2] += 1

        done = len(self.tasks) == 0
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.step_count}")
        print(f"Tasks: {self.tasks}")
        print(f"Satellites: {self.sats}")
