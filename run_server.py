import argparse
import sys
import os

# Ensure local imports work
sys.path.append(os.getcwd())

from ugv_rl.network.robot_server import RobotServer
from ugv_rl.controllers.real_robot import RealRobot

def main():
    parser = argparse.ArgumentParser(description="Run UGV Robot Server on Jetson")
    parser.add_argument('--port', type=int, default=5000, help="Port to listen on")
    parser.add_argument('--uart', type=str, default='/dev/ttyTHS1', help="Serial port")
    args = parser.parse_args()

    print(f"Initializing Real Robot on {args.uart}...")
    try:
        robot = RealRobot(serial_port=args.uart)
    except Exception as e:
        print(f"Error initializing robot: {e}")
        return

    print(f"Starting Server on port {args.port}...")
    server = RobotServer(port=args.port, robot=robot)
    server.start()

if __name__ == '__main__':
    main()
