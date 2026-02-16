import threading
import time
import sys
import os

from ugv_rl.network.robot_server import RobotServer
from ugv_rl.controllers.remote_robot import RemoteRobot
from ugv_rl.controllers.mock_robot import MockRobot

def run_server_threadback():
    print("Starting Mock Server...")
    # Use MockRobot for this test so we don't need hardware
    robot = MockRobot()
    server = RobotServer(port=5000, robot=robot)
    server.start()

def main():
    # Start server in background thread
    t = threading.Thread(target=run_server_threadback, daemon=True)
    t.start()
    
    time.sleep(1.0) # Wait for server to bind
    
    print("Connecting Client...")
    client = RemoteRobot(ip='192.168.137.51', port=5000)
    
    if not client.connected:
        print("Failed to connect!")
        sys.exit(1)
        
    print("Sending Move Command...")
    client.move(0.5, 0.2)
    time.sleep(0.5)
    
    print("Getting State...")
    state = client.get_state()
    print(f"State: {state}")
    
    if state['v'] == 0.5 and state['omega'] == 0.2:
        print("VERIFICATION PASSED: State matches command.")
    else:
        print("VERIFICATION FAILED: State mismatch.")
        
    print("Stopping...")
    client.stop()
    client.close()
    
    # Server thread will die when main dies (daemon)

if __name__ == '__main__':
    main()
