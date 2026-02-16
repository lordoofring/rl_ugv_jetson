import math
import time
from typing import Dict, Any
from ugv_rl.core.robot_interface import RobotInterface

class MockRobot(RobotInterface):
    """
    Simulated robot with simple unicycle kinematics.
    """
    
    def __init__(self, dt: float = 0.1):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.dt = dt  # Time step for simulation updates

    def move(self, linear_vel: float, angular_vel: float) -> None:
        self.v = linear_vel
        self.omega = angular_vel
        self._update_state()

    def stop(self) -> None:
        self.v = 0.0
        self.omega = 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'v': self.v,
            'omega': self.omega
        }

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.omega = 0.0

    def _update_state(self):
        """
        Update position based on current velocity and time step.
        Unicycle model:
        x_new = x + v * cos(theta) * dt
        y_new = y + v * sin(theta) * dt
        theta_new = theta + omega * dt
        """
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta += self.omega * self.dt
        
        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def step_simulation(self, dt: float) -> None:
        self.dt = dt
        self._update_state()
