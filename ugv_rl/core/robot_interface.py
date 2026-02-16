from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class RobotInterface(ABC):
    """
    Abstract interface for robot control (Real or Mock).
    """

    @abstractmethod
    def move(self, linear_vel: float, angular_vel: float) -> None:
        """
        Send movement commands to the robot.
        
        Args:
            linear_vel: Forward velocity (m/s)
            angular_vel: Rotational velocity (rad/s)
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the robot immediately.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the robot.
        
        Returns:
            Dictionary containing:
            - 'x': x position (meters)
            - 'y': y position (meters)
            - 'theta': orientation (radians)
            - 'v': current linear velocity (m/s)
            - 'omega': current angular velocity (rad/s)
        """
        pass
        
    @abstractmethod
    def reset(self, x: float, y: float, theta: float) -> None:
        """
        Reset the robot pose (mainly for simulation).
        For real robot, this might just reset internal odometry.
        """
        pass

    @abstractmethod
    def step_simulation(self, dt: float) -> None:
        """
        Advance the simulation by dt seconds.
        For real robots, this should be a no-op or just sleep.
        """
        pass
