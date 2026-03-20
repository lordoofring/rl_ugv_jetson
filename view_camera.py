"""Stream the robot's camera feed to your PC over the network.

Usage:
    python view_camera.py --ip <JETSON_IP>
    python view_camera.py --ip 192.168.1.10 --size 320x240

Press Q to quit.
"""

import argparse
import cv2
from ugv_rl.controllers.remote_robot import RemoteRobot


def main():
    parser = argparse.ArgumentParser(description="View robot camera feed remotely")
    parser.add_argument("--ip", type=str, required=True, help="Jetson IP address")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--size", type=str, default="320x240",
                        help="Display size WxH (default: 320x240)")
    args = parser.parse_args()

    w, h = [int(x) for x in args.size.split("x")]

    robot = RemoteRobot(ip=args.ip, port=args.port)

    print(f"Streaming at {w}x{h}. Press Q to quit.")
    while True:
        frame = robot.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, (w, h))
            cv2.imshow("Robot Camera", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    robot.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
