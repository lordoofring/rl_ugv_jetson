import argparse
import sys
import os
import yaml

# Ensure local imports work
sys.path.append(os.getcwd())

from ugv_rl.network.robot_server import RobotServer
from ugv_rl.controllers.real_robot import RealRobot

def main():
    parser = argparse.ArgumentParser(description="Run UGV Robot Server on Jetson")
    parser.add_argument('--port', type=int, default=5000, help="Port to listen on")
    parser.add_argument('--uart', type=str, default='/dev/ttyTHS1', help="Serial port")
    parser.add_argument('--vision', action='store_true', help="Enable ArUco vision localizer")
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    vision_cfg = config.get('vision', {})
    use_vision = args.vision or vision_cfg.get('enabled', False)
    vision_kwargs = {}
    if use_vision:
        vision_kwargs = {
            'camera_index': vision_cfg.get('camera_index', 0),
            'marker_map_path': vision_cfg.get('marker_map_path', 'marker_map.json'),
            'cell_size': config['env'].get('cell_size', 0.5),
            'marker_size': vision_cfg.get('marker_size', 0.05),
            'aruco_dict_name': vision_cfg.get('aruco_dict', 'DICT_4X4_50'),
        }

    print(f"Initializing Real Robot on {args.uart}...")
    try:
        robot = RealRobot(serial_port=args.uart, use_vision=use_vision, vision_kwargs=vision_kwargs)
    except Exception as e:
        print(f"Error initializing robot: {e}")
        return

    if use_vision:
        print("Vision localizer enabled on server.")

    print(f"Starting Server on port {args.port}...")
    server = RobotServer(port=args.port, robot=robot)
    server.start()

if __name__ == '__main__':
    main()
