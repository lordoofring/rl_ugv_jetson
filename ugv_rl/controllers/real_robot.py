from typing import Dict, Any
from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.waveshare_controller import WaveshareController

class RealRobot(RobotInterface):
    """
    Real robot implementation using WaveshareController.
    """
    #Uart must be /dev/ttyTHS1
    def __init__(self, serial_port: str = '/dev/ttyTHS1', baud_rate: int = 115200, wheel_base: float = 0.15, max_pwm: int = 255):
        self.controller = WaveshareController(serial_port, baud_rate)
        self.wheel_base = wheel_base
        self.max_pwm = max_pwm # Updated to 255 based on standard Waveshare UGV code
        
        self.current_v = 0.0
        self.current_omega = 0.0

    def move(self, linear_vel: float, angular_vel: float) -> None:
        """
        Convert twist (v, omega) to differential drive wheel speeds (PWM).
        """
        self.current_v = linear_vel
        self.current_omega = angular_vel

        # Differential drive kinematics
        left_vel = linear_vel - (angular_vel * self.wheel_base / 2.0)
        right_vel = linear_vel + (angular_vel * self.wheel_base / 2.0)

        # Scale to PWM
        # Assumption: 0.5 m/s is approximately max speed at max PWM
        velocity_to_pwm_scale = self.max_pwm / 0.5 
        
        left_pwm = int(left_vel * velocity_to_pwm_scale)
        right_pwm = int(right_vel * velocity_to_pwm_scale)
        
        # Clamp values
        left_pwm = max(min(left_pwm, self.max_pwm), -self.max_pwm)
        right_pwm = max(min(right_pwm, self.max_pwm), -self.max_pwm)
        
        # print(f"DEBUG: Move v={linear_vel:.2f}, w={angular_vel:.2f} -> PWM L={left_pwm}, R={right_pwm}")
        self.controller.base_speed_ctrl(left_pwm, right_pwm)

    def stop(self) -> None:
        self.controller.base_speed_ctrl(0, 0)
        self.current_v = 0.0
        self.current_omega = 0.0

    def get_state(self) -> Dict[str, Any]:
        data = self.controller.on_data_received()
        return {
            'x': 0.0, 
            'y': 0.0, 
            'theta': 0.0, 
            'v': self.current_v,
            'omega': self.current_omega,
            'raw_sensor_data': data
        }

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.stop()
        pass

    def close(self):
        self.controller.close()

    def step_simulation(self, dt: float) -> None:
        pass
