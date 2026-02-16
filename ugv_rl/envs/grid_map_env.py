import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import math
from typing import Optional, Dict

from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.mock_robot import MockRobot

class GridMapEnv(gym.Env):
    """
    Gym environment for grid navigation with static obstacles.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config_path: str = 'config.yaml', robot: RobotInterface = None, map_layout: Optional[np.ndarray] = None):
        super(GridMapEnv, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.grid_size = self.config['env'].get('grid_size', 10)
        self.cell_size = self.config['env'].get('cell_size', 1.0)
        self.max_steps = self.config['env'].get('max_steps', 200)
        
        # Action space: 0=Forward, 1=Left, 2=Right
        self.action_space = spaces.Discrete(3)
        
        # Observation space:
        # [rel_goal_x, rel_goal_y, sin(theta), cos(theta), lidar_1 ... lidar_8]
        # Adding 8 "raycasts" for obstacle detection in 45 deg increments
        self.num_rays = 8
        low = np.array([-np.inf] * 4 + [0.0] * self.num_rays)
        high = np.array([np.inf] * 4 + [self.grid_size * self.cell_size] * self.num_rays)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        if robot:
            self.robot = robot
        else:
            self.robot = MockRobot()
            
        # Map Layout: 0 = Free, 1 = Obstacle
        if map_layout is not None:
            self.map = map_layout
        else:
            self.map = np.zeros((self.grid_size, self.grid_size), dtype=int)
            
        self.start_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([(self.grid_size-1)*self.cell_size, (self.grid_size-1)*self.cell_size])
        self.steps = 0

    def set_map(self, layout: np.ndarray, start: np.ndarray, goal: np.ndarray):
        """Update the map layout and endpoints."""
        self.map = layout
        self.start_pos = start
        self.goal_pos = goal
        self.grid_size = layout.shape[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Reset robot to start position
        # Add random noise to start if training? For now exact.
        self.robot.reset(self.start_pos[0], self.start_pos[1], 0.0) # Start facing 0 radians
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # Convert action to robot command
        linear_v = 0.0
        angular_v = 0.0
        
        speed = self.config['robot'].get('max_speed', 0.5)
        
        if action == 0: # Forward
            linear_v = speed
        elif action == 1: # Left
            angular_v = 1.0 
        elif action == 2: # Right
            angular_v = -1.0
            
        self.robot.move(linear_v, angular_v)
        
        # Simulation Step
        dt = 0.1
        if isinstance(self.robot, MockRobot):
            original_x, original_y = self.robot.x, self.robot.y
            self.robot.step_simulation(dt)
            # Collision Check for Mock Robot
            # If hit obstacle, assume revert position or stop?
            # Simple check: point-in-obstacle
            if self._check_collision(self.robot.x, self.robot.y):
                # Hit obstacle!
                # Revert to previous position (simple physics)
                self.robot.x = original_x
                self.robot.y = original_y
                reward = -5.0 # Large penalty for collision
                done = True
                return self._get_obs(), reward, done, False, {"collision": True}
        else:
            # Real robot waits
            import time
            time.sleep(dt)
            
        # Get new state
        state = self.robot.get_state()
        x, y = state['x'], state['y']
        
        # Check map bounds
        if not (0 <= x <= self.grid_size * self.cell_size and 0 <= y <= self.grid_size * self.cell_size):
            reward = -5.0 # Out of bounds
            done = True
            return self._get_obs(), reward, done, False, {"out_of_bounds": True}
            
        # Distance to goal
        dist = np.sqrt((self.goal_pos[0] - x)**2 + (self.goal_pos[1] - y)**2)
        
        # Reward
        reward = -0.05 # Smaller step penalty
        done = False
        
        # Reached goal?
        if dist < 0.5: 
            reward += 100.0
            done = True
            
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, {}

    def _check_collision(self, x, y):
        """Check if position (x,y) is inside an obstacle cell."""
        # Convert to grid indices
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return self.map[gx, gy] == 1
        return False # Out of bounds handled elsewhere or assumed free here
        
    def _raycast(self, x, y, angle):
        """Simulate a single lidar ray."""
        step_pc = 0.1 # Ray step size
        max_dist = 5.0 # Max ray distance
        
        cx, cy = x, y
        dist = 0.0
        
        while dist < max_dist:
            cx += math.cos(angle) * step_pc
            cy += math.sin(angle) * step_pc
            dist += step_pc
            
            if self._check_collision(cx, cy):
                return dist
            
            # Also check bounds
            if not (0 <= cx <= self.grid_size * self.cell_size and 0 <= cy <= self.grid_size * self.cell_size):
                return dist
                
        return max_dist

    def _get_obs(self):
        state = self.robot.get_state()
        x, y = state['x'], state['y']
        theta = state['theta']
        
        rel_x = self.goal_pos[0] - x
        rel_y = self.goal_pos[1] - y
        
        # Raycasts in 8 directions relative to robot heading
        rays = []
        for i in range(self.num_rays):
            angle = theta + (i * 2 * math.pi / self.num_rays)
            dist = self._raycast(x, y, angle)
            rays.append(dist)
            
        obs = [rel_x, rel_y, math.sin(theta), math.cos(theta)] + rays
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        pass # Will be handled by Pygame external visualizer
