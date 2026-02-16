from typing import Dict, Any
from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.waveshare_controller import WaveshareController

class RealRobot(RobotInterface):
    """
    Real robot implementation using WaveshareController.
    """
    # On Jetson Nano, the UART on the 40-pin header is usually /dev/ttyTHS1
    def __init__(self, serial_port: str = '/dev/ttyTHS1', baud_rate: int = 115200, wheel_base: float = 0.15, max_speed: float = 0.5):
        self.controller = WaveshareController(serial_port, baud_rate)
        self.wheel_base = wheel_base
        self.max_speed = max_speed 
        
        self.current_v = 0.0
        self.current_omega = 0.0
        
        # Dead Reckoning State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def move(self, linear_vel: float, angular_vel: float) -> None:
        """
        Convert twist (v, omega) to differential drive wheel speeds (m/s).
        """
        self.current_v = linear_vel
        self.current_omega = angular_vel

        # Differential drive kinematics
        left_vel = linear_vel - (angular_vel * self.wheel_base / 2.0)
        right_vel = linear_vel + (angular_vel * self.wheel_base / 2.0)

        # Clamp values to max speed
        left_vel = max(min(left_vel, self.max_speed), -self.max_speed)
        right_vel = max(min(right_vel, self.max_speed), -self.max_speed)
        
        # Send direct velocity commands (floats)
        self.controller.base_speed_ctrl(left_vel, right_vel)

    def stop(self) -> None:
        self.controller.base_speed_ctrl(0, 0)
        self.current_v = 0.0
        self.current_omega = 0.0

    def get_state(self) -> Dict[str, Any]:
        data = self.controller.on_data_received()
        return {
            'x': self.x, 
            'y': self.y, 
            'theta': self.theta, 
            'v': self.current_v,
            'omega': self.current_omega,
            'raw_sensor_data': data
        }

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.stop()
        self.x = x
        self.y = y
        self.theta = theta

    def close(self):
        self.controller.close()

    def step_simulation(self, dt: float) -> None:
        """
        Update estimated position based on last command.
        """
        import math
        self.x += self.current_v * math.cos(self.theta) * dt
        self.y += self.current_v * math.sin(self.theta) * dt
        self.theta += self.current_omega * dt
        
        # Normalize theta
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
