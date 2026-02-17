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
        
        # Action space: 0=North, 1=South, 2=East, 3=West
        self.action_space = spaces.Discrete(4)
        
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
        
        # 1. Determine Target Position based on Action
        # Current rounded grid position
        cx = round(self.robot.x / self.cell_size) * self.cell_size
        cy = round(self.robot.y / self.cell_size) * self.cell_size
        
        target_x, target_y = cx, cy
        target_theta = 0.0
        
        if action == 0: # North (-Y for grid image, but let's assume standard cartesian +Y is North? 
                        # Wait, in image coords usually +Y is Down. 
                        # Let's stick to standard math: +Y is Up (North), +X is Right (East)
                        # The map editor might use matrix indexing which is (row, col) -> (y, x).
                        # Let's check map editor... it uses [gx, gy]. pygame draws x=gx*cell, y=gy*cell.
                        # So +X is Right, +Y is Down (Screen coords).
                        # So North should be -Y?
                        # Let's define: 0=North(-Y), 1=South(+Y), 2=East(+X), 3=West(-X)
            target_y = cy - self.cell_size
            target_theta = -math.pi / 2 # -90 deg
        elif action == 1: # South (+Y)
            target_y = cy + self.cell_size
            target_theta = math.pi / 2 # +90 deg
        elif action == 2: # East (+X)
            target_x = cx + self.cell_size
            target_theta = 0.0 # 0 deg
        elif action == 3: # West (-X)
            target_x = cx - self.cell_size
            target_theta = math.pi # 180 deg
            
        # 2. Check Validity (Collision / OOB)
        if self._check_collision(target_x, target_y) or self._check_oob(target_x, target_y):
            # Invalid move
            reward = -5.0
            # Don't move robot
            # done = True ? No, let it learn.
        else:
            # 3. Execute Move (Turn then Drive)
            self._execute_cardinal_move(target_theta, self.cell_size)
            reward = -0.1
            
        # Get new state
        state = self.robot.get_state()
        x, y = state['x'], state['y']
        
        # Distance to goal
        dist = np.sqrt((self.goal_pos[0] - x)**2 + (self.goal_pos[1] - y)**2)
        
        if dist < self.cell_size / 2:
            reward += 100.0
            done = True
        else:
            done = False
            
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, {}

    def _check_oob(self, x, y):
        # Slightly permissive bounds since x,y are floats
        return not (0 <= x < self.grid_size * self.cell_size and 0 <= y < self.grid_size * self.cell_size)

    def _execute_cardinal_move(self, target_theta, dist):
        """Blocking function to turn robot then move forward."""
        # A. Turn
        current_theta = self.robot.theta
        diff = target_theta - current_theta
        # Normalize to [-pi, pi]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        
        # Simple turning params
        w_speed = 1.0 # rad/s
        turn_duration = abs(diff) / w_speed
        
        if abs(diff) > 0.1: # Only turn if needed
            w_cmd = w_speed if diff > 0 else -w_speed
            self.robot.move(0.0, w_cmd)
            
            if isinstance(self.robot, MockRobot):
                self.robot.step_simulation(turn_duration)
            else:
                import time
                time.sleep(turn_duration)
                
            self.robot.stop()
            
        # B. Move Forward
        v_speed = self.config['robot'].get('max_speed', 0.5)
        move_duration = dist / v_speed
        
        self.robot.move(v_speed, 0.0)
        
        if isinstance(self.robot, MockRobot):
            self.robot.step_simulation(move_duration)
        else:
            import time
            time.sleep(move_duration)
            
        self.robot.stop()

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
