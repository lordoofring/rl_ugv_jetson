from typing import Dict, Any
from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.waveshare_controller import WaveshareController

class RealRobot(RobotInterface):
    """
    Real robot implementation using WaveshareController.
    # On Jetson Nano, the UART on the 40-pin header is usually /dev/ttyTHS1
    def __init__(self, serial_port: str = '/dev/ttyTHS1', baud_rate: int = 115200, wheel_base: float = 0.15, max_pwm: int = 100):
        self.controller = WaveshareController(serial_port, baud_rate)
        self.wheel_base = wheel_base
        self.max_pwm = max_pwm # Assuming 0-100 or 0-255 scaling depending on firmware
        # Simple state estimation (dead reckoning not implemented here, 
        # relying on external verification or visual feedback for now)
        self.current_v = 0.0
        self.current_omega = 0.0

    def move(self, linear_vel: float, angular_vel: float) -> None:
        """
        Convert twist (v, omega) to differential drive wheel speeds (PWM).
        """
        self.current_v = linear_vel
        self.current_omega = angular_vel

        # Differential drive kinematics
        # v = (v_r + v_l) / 2
        # omega = (v_r - v_l) / L
        # => v_l = v - (omega * L / 2)
        # => v_r = v + (omega * L / 2)
        
        left_vel = linear_vel - (angular_vel * self.wheel_base / 2.0)
        right_vel = linear_vel + (angular_vel * self.wheel_base / 2.0)

        # Scale to PWM (This requires calibration!)
        # For now, let's assume 1.0 m/s = max_pwm
        # This is a PLACEHOLDER scalar.
        velocity_to_pwm_scale = self.max_pwm / 0.5 # Example: 0.5 m/s is full speed
        
        left_pwm = int(left_vel * velocity_to_pwm_scale)
        right_pwm = int(right_vel * velocity_to_pwm_scale)
        
        # Clamp values
        left_pwm = max(min(left_pwm, self.max_pwm), -self.max_pwm)
        right_pwm = max(min(right_pwm, self.max_pwm), -self.max_pwm)

        self.controller.base_speed_ctrl(left_pwm, right_pwm)

    def stop(self) -> None:
        self.controller.base_speed_ctrl(0, 0)
        self.current_v = 0.0
        self.current_omega = 0.0

    def get_state(self) -> Dict[str, Any]:
        # NOTE: Without odometry feedback from the MCU, we return 0 or last command.
        # The provided base_ctrl.py does receive data: self.rl.readline().
        # If the robot sends back odometry, we should parse it.
        # For this iteration, we return placeholders or what we can read.
        data = self.controller.on_data_received()
        # Parse data if available, otherwise return last known command state (not true state)
        # Assuming data might contain IMU or Encoder values if the firmware supports it.
        return {
            'x': 0.0, # Placeholder
            'y': 0.0, # Placeholder
            'theta': 0.0, # Placeholder
            'v': self.current_v,
            'omega': self.current_omega,
            'raw_sensor_data': data
        }

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        # Cannot reset real robot physical position
        self.stop()
        pass

    def close(self):
        self.controller.close()

    def step_simulation(self, dt: float) -> None:
        pass
