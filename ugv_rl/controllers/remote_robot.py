import socket
import json
import time
from typing import Dict, Any
from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.network.protocol import NetworkProtocol

class RemoteRobot(RobotInterface):
    """
    Client-side interface that sends commands to RobotServer.
    """
    def __init__(self, ip: str, port: int = 5000):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        
        # State Cache
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self._connect()

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, self.port))
            self.connected = True
            print(f"Connected to RobotServer at {self.ip}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to {self.ip}:{self.port}: {e}")
            self.connected = False

    def move(self, linear_vel: float, angular_vel: float) -> None:
        if not self.connected: 
            return
            
        cmd = {
            "cmd": "move",
            "v": linear_vel,
            "w": angular_vel
        }
        try:
            NetworkProtocol.send_msg(self.sock, cmd)
            # Wait for ACK
            NetworkProtocol.recv_msg(self.sock)
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False

    def stop(self) -> None:
        if not self.connected: return
        try:
            NetworkProtocol.send_msg(self.sock, {"cmd": "stop"})
        except:
            pass

    def get_state(self) -> Dict[str, Any]:
        if not self.connected:
            return {'x':0, 'y':0, 'theta':0, 'v':0, 'omega':0}
            
        try:
            NetworkProtocol.send_msg(self.sock, {"cmd": "get_state"})
            resp = NetworkProtocol.recv_msg(self.sock)
            if resp and 'state' in resp:
                s = resp['state']
                # Update local cache
                self.x = s.get('x', 0.0)
                self.y = s.get('y', 0.0)
                self.theta = s.get('theta', 0.0)
                return s
        except Exception as e:
            print(f"Recv error: {e}")
            self.connected = False
            
        return {'x':0, 'y':0, 'theta':0, 'v':0, 'omega':0}

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        if not self.connected: return
        
        # Update local cache immediately
        self.x = x
        self.y = y
        self.theta = theta
        
        cmd = {
            "cmd": "reset",
            "x": x,
            "y": y,
            "theta": theta
        }
        try:
            NetworkProtocol.send_msg(self.sock, cmd)
            NetworkProtocol.recv_msg(self.sock) # ACK
        except:
            pass

    def step_simulation(self, dt: float) -> None:
        pass
        
    def close(self):
        if self.sock:
            self.sock.close()
