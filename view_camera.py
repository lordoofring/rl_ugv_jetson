"""Stream the robot's camera feed to your PC over the network.

Usage:
    python view_camera.py --ip <JETSON_IP>
    python view_camera.py --ip 192.168.1.10 --port 5000

Press Q to quit.
"""

import argparse
import cv2
from ugv_rl.controllers.remote_robot import RemoteRobot


def main():
    parser = argparse.ArgumentParser(description="View robot camera feed remotely")
    parser.add_argument("--ip", type=str, required=True, help="Jetson IP address")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    robot = RemoteRobot(ip=args.ip, port=args.port)

    print("Streaming camera feed. Press Q to quit.")
    while True:
        frame = robot.get_frame()
        if frame is not None:
            cv2.imshow("Robot Camera", frame)
        else:
            print("No frame received.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    robot.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
