import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.mock_robot import MockRobot
# RealRobot import will be conditional or injected

class GridNavEnv(gym.Env):
    """
    Gym environment for grid navigation.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path: str = 'config.yaml', robot: RobotInterface = None):
        super(GridNavEnv, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.grid_size = self.config['env']['grid_size']
        self.cell_size = self.config['env']['cell_size']
        self.max_steps = self.config['env']['max_steps']
        
        # Action space: 0=Forward, 1=Left, 2=Right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # [x, y, theta, goal_x, goal_y] relative to start or absolute?
        # Let's use relative to goal for simpler learning: [rel_x, rel_y, sin(theta), cos(theta)]
        # Limits: -grid_size*cell_size to +grid_size*cell_size
        low = np.array([-self.grid_size*self.cell_size, -self.grid_size*self.cell_size, -1.0, -1.0])
        high = np.array([self.grid_size*self.cell_size, self.grid_size*self.cell_size, 1.0, 1.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        if robot:
            self.robot = robot
        else:
            # Default to mock
            self.robot = MockRobot()
            
        self.goal = np.array([5.0, 5.0]) # Fixed goal for now, can be randomized
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Reset robot to (0,0) with random orientation or fixed
        self.robot.reset(0.0, 0.0, 0.0)
        
        # Randomize goal if needed, for now fixed at (5,5)
        self.goal = np.array([5.0, 5.0])
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 0: Forward, 1: Turn Left, 2: Turn Right
        # Speeds should be configurable
        linear_v = 0.0
        angular_v = 0.0
        
        if action == 0: # Forward
            linear_v = self.config['robot']['max_speed']
        elif action == 1: # Left
            angular_v = 1.0 # rad/s
        elif action == 2: # Right
            angular_v = -1.0 # rad/s
            
        self.robot.move(linear_v, angular_v)
        
        # Step simulation or wait
        dt = 0.1 # Simulation time step
        if isinstance(self.robot, MockRobot):
            self.robot.step_simulation(dt)
        else:
            # Real robot: wait for action duration
            time.sleep(dt)
            
        # Get new state
        state = self.robot.get_state()
        x, y = state['x'], state['y']
        
        # Distance to goal
        dist = np.sqrt((self.goal[0] - x)**2 + (self.goal[1] - y)**2)
        
        # Reward
        reward = -0.1 # Step penalty
        done = False
        
        if dist < 0.5: # Reached goal
            reward += 10.0
            done = True
            
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        state = self.robot.get_state()
        x, y = state['x'], state['y']
        theta = state['theta']
        
        rel_x = self.goal[0] - x
        rel_y = self.goal[1] - y
        
        # Rotate to robot frame? Or just global relative?
        # Global relative for now.
        return np.array([rel_x, rel_y, np.sin(theta), np.cos(theta)], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Robot at {self.robot.get_state()['x']:.1f}, {self.robot.get_state()['y']:.1f}")
