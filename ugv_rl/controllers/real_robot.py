import logging
from typing import Dict, Any, Optional
from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.waveshare_controller import WaveshareController

logger = logging.getLogger(__name__)


class RealRobot(RobotInterface):
    """
    Real robot implementation using WaveshareController.
    Optionally uses ArUco vision localizer to correct dead-reckoning drift.
    """
    # On Jetson Nano, the UART on the 40-pin header is usually /dev/ttyTHS1
    def __init__(self, serial_port: str = '/dev/ttyTHS1', baud_rate: int = 115200,
                 wheel_base: float = 0.15, max_speed: float = 0.5,
                 use_vision: bool = False, vision_kwargs: Optional[dict] = None):
        self.controller = WaveshareController(serial_port, baud_rate)
        self.wheel_base = wheel_base
        self.max_speed = max_speed

        self.current_v = 0.0
        self.current_omega = 0.0

        # Dead Reckoning State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Vision localizer (optional)
        self.vision = None
        if use_vision:
            try:
                from ugv_rl.vision.localizer import VisionLocalizer
                self.vision = VisionLocalizer(**(vision_kwargs or {}))
                logger.info("Vision localizer enabled.")
            except Exception as e:
                logger.warning("Could not initialise vision localizer: %s", e)

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
            'raw_sensor_data': data,
        }

    def localize(self):
        """Use vision to get corrected (grid_x, grid_y, theta), or None."""
        if self.vision is None:
            return None
        return self.vision.localize()

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.stop()
        self.x = x
        self.y = y
        self.theta = theta

    def close(self):
        if self.vision is not None:
            self.vision.release()
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
