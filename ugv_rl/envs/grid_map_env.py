import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import math
from typing import Optional, Dict

from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.mock_robot import MockRobot

# Action constants
ACTION_NORTH = 0  # -Y (up on screen)
ACTION_SOUTH = 1  # +Y (down on screen)
ACTION_EAST = 2   # +X (right on screen)
ACTION_WEST = 3   # -X (left on screen)

ACTION_NAMES = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}

# Grid deltas (dx, dy) for each action
ACTION_DELTAS = {
    ACTION_NORTH: (0, -1),
    ACTION_SOUTH: (0, 1),
    ACTION_EAST: (1, 0),
    ACTION_WEST: (-1, 0),
}

# Heading angle for each action (screen coords: +X right, +Y down)
ACTION_THETAS = {
    ACTION_NORTH: -math.pi / 2,
    ACTION_SOUTH: math.pi / 2,
    ACTION_EAST: 0.0,
    ACTION_WEST: math.pi,
}


class GridMapEnv(gym.Env):
    """
    Discrete grid navigation environment.
    The agent moves one cell at a time (N/S/E/W).
    Position is tracked as integer grid indices — this is authoritative.
    The physical robot executes movements but the grid state drives the RL.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config_path: str = 'config.yaml', robot: RobotInterface = None, map_layout: Optional[np.ndarray] = None):
        super().__init__()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.grid_size = self.config['env'].get('grid_size', 10)
        self.cell_size = self.config['env'].get('cell_size', 1.0)
        self.max_steps = self.config['env'].get('max_steps', 200)

        # Discrete actions: 0=North, 1=South, 2=East, 3=West
        self.action_space = spaces.Discrete(4)

        # Observation: [rel_goal_x, rel_goal_y, sin(theta), cos(theta), ray_0..ray_7]
        self.num_rays = 8
        low = np.array([-np.inf] * 4 + [0.0] * self.num_rays)
        high = np.array([np.inf] * 4 + [self.grid_size * self.cell_size] * self.num_rays)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.robot = robot if robot else MockRobot()

        # Map layout: 0=free, 1=obstacle. Indexed as map[gx, gy].
        if map_layout is not None:
            self.map = map_layout
        else:
            self.map = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Grid positions (authoritative integer indices)
        self.start_gx, self.start_gy = 0, 0
        self.goal_gx = self.grid_size - 1
        self.goal_gy = self.grid_size - 1
        self.agent_gx = 0
        self.agent_gy = 0
        self.agent_theta = 0.0

        # For compatibility with set_map / train.py
        self.start_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([(self.grid_size - 1) * self.cell_size,
                                  (self.grid_size - 1) * self.cell_size])

        self.steps = 0
        self.last_action = None

    def set_map(self, layout: np.ndarray, start: np.ndarray, goal: np.ndarray):
        self.map = layout
        self.start_pos = start
        self.goal_pos = goal
        self.grid_size = layout.shape[0]

        self.start_gx = int(round(start[0] / self.cell_size))
        self.start_gy = int(round(start[1] / self.cell_size))
        self.goal_gx = int(round(goal[0] / self.cell_size))
        self.goal_gy = int(round(goal[1] / self.cell_size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = None

        self.agent_gx = self.start_gx
        self.agent_gy = self.start_gy
        self.agent_theta = 0.0

        x = self.agent_gx * self.cell_size
        y = self.agent_gy * self.cell_size
        self.robot.reset(x, y, 0.0)

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)
        self.last_action = action

        dx, dy = ACTION_DELTAS[action]
        target_gx = self.agent_gx + dx
        target_gy = self.agent_gy + dy

        # Check bounds and collision
        valid = (0 <= target_gx < self.grid_size and
                 0 <= target_gy < self.grid_size and
                 self.map[target_gx, target_gy] != 1)

        if not valid:
            reward = -5.0
        else:
            # Update authoritative grid position
            self.agent_gx = target_gx
            self.agent_gy = target_gy
            self.agent_theta = ACTION_THETAS[action]

            target_x = target_gx * self.cell_size
            target_y = target_gy * self.cell_size

            if isinstance(self.robot, MockRobot):
                # Snap mock robot to exact grid cell center
                self.robot.reset(target_x, target_y, self.agent_theta)
            else:
                # Execute physical turn + drive on real robot
                self._execute_cardinal_move(self.agent_theta, self.cell_size)
                # Sync dead reckoning back to intended grid position
                self.robot.reset(target_x, target_y, self.agent_theta)

            reward = -0.1

        # Check goal
        done = False
        if self.agent_gx == self.goal_gx and self.agent_gy == self.goal_gy:
            reward += 100.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        info = {
            'agent_pos': (self.agent_gx, self.agent_gy),
            'action': ACTION_NAMES.get(action, '?'),
        }

        return self._get_obs(), reward, done, False, info

    def _execute_cardinal_move(self, target_theta, dist):
        """Turn robot to target heading then drive forward one cell. For real robots only."""
        current_theta = self.robot.theta if hasattr(self.robot, 'theta') else 0.0
        diff = target_theta - current_theta
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        w_speed = self.config['robot'].get('turn_speed', 1.0)
        turn_duration = abs(diff) / w_speed

        if abs(diff) > 0.1:
            w_cmd = w_speed if diff > 0 else -w_speed
            self.robot.move(0.0, w_cmd)
            import time
            time.sleep(turn_duration)
            self.robot.stop()

        v_speed = self.config['robot'].get('max_speed', 0.5)
        move_duration = dist / v_speed

        self.robot.move(v_speed, 0.0)
        import time
        time.sleep(move_duration)
        self.robot.stop()

    def _check_collision(self, x, y):
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return self.map[gx, gy] == 1
        return False

    def _raycast(self, x, y, angle):
        step_pc = 0.1
        max_dist = 5.0
        cx, cy = x, y
        dist = 0.0
        while dist < max_dist:
            cx += math.cos(angle) * step_pc
            cy += math.sin(angle) * step_pc
            dist += step_pc
            if self._check_collision(cx, cy):
                return dist
            if not (0 <= cx <= self.grid_size * self.cell_size and 0 <= cy <= self.grid_size * self.cell_size):
                return dist
        return max_dist

    def _get_obs(self):
        # Use authoritative grid position (converted to meters)
        x = self.agent_gx * self.cell_size
        y = self.agent_gy * self.cell_size
        theta = self.agent_theta

        rel_x = self.goal_pos[0] - x
        rel_y = self.goal_pos[1] - y

        rays = []
        for i in range(self.num_rays):
            angle = theta + (i * 2 * math.pi / self.num_rays)
            dist = self._raycast(x, y, angle)
            rays.append(dist)

        obs = [rel_x, rel_y, math.sin(theta), math.cos(theta)] + rays
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        pass
