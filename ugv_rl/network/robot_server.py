import socket
import threading
import time
import sys
from typing import Optional

from ugv_rl.network.protocol import NetworkProtocol
from ugv_rl.controllers.real_robot import RealRobot

class RobotServer:
    def __init__(self, host='0.0.0.0', port=5000, robot=None):
        self.host = host
        self.port = port
        self.robot = robot
        self.running = False
        self.sock = None
        self.client_sock = None

    def start(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Robot Server listening on {self.host}:{self.port}")
        
        # Start State Update Thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        try:
            while self.running:
                print("Waiting for connection...")
                client, addr = self.sock.accept()
                print(f"Connected by {addr}")
                self.handle_client(client)
        except KeyboardInterrupt:
            print("Server stopping...")
        finally:
            self.stop()

    def _update_loop(self):
        """Periodically update robot state (Dead Reckoning)."""
        dt = 0.1
        while self.running:
            if self.robot:
                self.robot.step_simulation(dt)
            time.sleep(dt)

    def handle_client(self, client):
        self.client_sock = client
        try:
            while self.running:
                req = NetworkProtocol.recv_msg(client)
                if req is None:
                    break
                    
                cmd = req.get('cmd')
                resp = {'status': 'ok'}
                
                if cmd == 'move':
                    v = float(req.get('v', 0.0))
                    w = float(req.get('w', 0.0))
                    if self.robot:
                        self.robot.move(v, w)
                        
                elif cmd == 'stop':
                    if self.robot:
                        self.robot.stop()
                        
                elif cmd == 'get_state':
                    if self.robot:
                        resp['state'] = self.robot.get_state()
                    else:
                        resp['state'] = {'x': 0, 'y': 0, 'theta': 0, 'v': 0, 'omega': 0}
                
                elif cmd == 'reset':
                    x = float(req.get('x', 0.0))
                    y = float(req.get('y', 0.0))
                    theta = float(req.get('theta', 0.0))
                    if self.robot:
                        self.robot.reset(x, y, theta)
                        
                else:
                    resp['status'] = 'error'
                    resp['msg'] = f"Unknown command: {cmd}"
                    
                NetworkProtocol.send_msg(client, resp)
                
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            print("Client disconnected")
            client.close()
            # Stop robot on disconnect for safety
            if self.robot:
                self.robot.stop()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        if self.robot:
            self.robot.close()

if __name__ == '__main__':
    # Standalone execution
    try:
        robot = RealRobot() # Will use default /dev/ttyTHS1
        server = RobotServer(robot=robot)
        server.start()
    except Exception as e:
        print(f"Failed to start server: {e}")
