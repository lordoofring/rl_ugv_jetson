"""
Deploy the Ball Push policy on the real robot.

Usage:
    python run_ball_push.py --model ball_push_camera --ip <JETSON_IP>
    python run_ball_push.py --model ball_push_camera --local
    python run_ball_push.py --calibrate --ip <JETSON_IP>

Controls: Q=quit, SPACE=pause/resume, R=reset
"""

import argparse
import math
import time
import sys
import cv2
import numpy as np
import yaml
#from stable_baselines3 import PPO
from ugv_rl.vision.frame_observer import FrameObserver

TURN_ANGLE_DEG = 5.0
STEP_DIST = 0.05
ACTION_NAMES = {0: "Rot L", 1: "Rot R", 2: "Fwd"}


def execute_action(robot, action, config):
    wb = config["robot"].get("wheel_base", 0.175)
    tws = config["robot"].get("turn_wheel_speed", 0.3)
    t90 = config["robot"].get("turn_time_90", 1.0)
    ms = config["robot"].get("max_speed", 0.5)

    if action == 0:
        w = tws / (wb / 2.0)
        robot.move(0.0, w)
        time.sleep(t90 * TURN_ANGLE_DEG / 90.0)
        robot.stop()
    elif action == 1:
        w = tws / (wb / 2.0)
        robot.move(0.0, -w)
        time.sleep(t90 * TURN_ANGLE_DEG / 90.0)
        robot.stop()
    elif action == 2:
        robot.move(ms, 0.0)
        time.sleep(STEP_DIST / ms)
        robot.stop()


def run_calibration(get_frame, observer):
    print("\n--- Calibration ---")
    print("Green=ball, Blue=tape, Yellow=gap. Q to quit.\n")
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        vis = observer.annotate_frame(frame)
        obs = observer.observe(frame)
        if obs is not None:
            txt = f"dist={obs[0]:.2f}m ang={math.degrees(obs[1]):.0f} gap={obs[2]:.2f}"
            cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "NO BALL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Calibrate", vis)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def run_policy(get_frame, robot, model, observer, config):
    print("\n--- Running Policy ---")
    print("Q=quit, SPACE=pause, R=reset\n")
    paused = False
    steps = 0

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        obs = observer.observe(frame)
        vis = observer.annotate_frame(frame)

        if obs is None:
            cv2.putText(vis, "SEARCHING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Ball Push", vis)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            if not paused:
                execute_action(robot, 0, config)
            continue

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        cv2.putText(vis, f"{ACTION_NAMES[action]} step:{steps} gap:{obs[2]:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Ball Push", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord("r"):
            steps = 0
            print("--- Reset ---")

        if not paused:
            execute_action(robot, action, config)
            steps += 1
            if obs[2] < 0.05:
                print(f"\nBall likely out at step {steps}!")
                paused = True

    cv2.destroyAllWindows()
    robot.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ball_push_camera")
    parser.add_argument("--ip", type=str, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    robot = None
    get_frame = None

    if args.ip:
        from ugv_rl.controllers.remote_robot import RemoteRobot
        robot = RemoteRobot(ip=args.ip, port=args.port)
        get_frame = robot.get_frame
        print(f"Connected to {args.ip}")
    elif args.local:
        from ugv_rl.controllers.real_robot import RealRobot
        robot = RealRobot(use_vision=True, vision_kwargs={
            "camera_index": config.get("vision", {}).get("camera_index", 0),
        })
        def local_frame():
            if robot.vision and robot.vision._camera_ok:
                ret, f = robot.vision.cap.read()
                return f if ret else None
            return None
        get_frame = local_frame
        print("Running locally")
    else:
        print("Specify --ip <JETSON_IP> or --local")
        sys.exit(1)

    bp = config.get("ball_push", {})
    observer = FrameObserver(
        ball_real_diameter=bp.get("ball_radius", 0.05) * 2,
        focal_length=280.0,
    )

    if args.calibrate:
        run_calibration(get_frame, observer)
    else:
        model = PPO.load(args.model)
        print(f"Model loaded: {args.model}")
        run_policy(get_frame, robot, model, observer, config)

    if robot:
        robot.stop()
        robot.close()


if __name__ == "__main__":
    main()
