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
from stable_baselines3 import PPO
from ugv_rl.vision.frame_observer import FrameObserver

TURN_ANGLE_DEG = 5.0
STEP_DIST = 0.05
ACTION_NAMES = {0: "Rot L", 1: "Rot R", 2: "Fwd"}


def send_action(robot, action, config):
    """Send movement command without blocking. No sleep, no stop — just set velocity."""
    wb = config["robot"].get("wheel_base", 0.175)
    tws = config["robot"].get("turn_wheel_speed", 0.3)
    ms = config["robot"].get("max_speed", 0.5)

    if action == 0:
        w = tws / (wb / 2.0)
        robot.move(0.0, w)
    elif action == 1:
        w = tws / (wb / 2.0)
        robot.move(0.0, -w)
    elif action == 2:
        robot.move(ms, 0.0)


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
    missed_frames = 0
    GRACE_FRAMES = 8  # ignore this many missed frames before searching

    logfile = open("policy_log.txt", "w")
    logfile.write("step,action,ball_dist,ball_angle,gap_dist,gap_angle\n")

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        obs = observer.observe(frame)
        vis = observer.annotate_frame(frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord("r"):
            steps = 0
            missed_frames = 0
            print("--- Reset ---")

        if paused:
            robot.stop()
            cv2.putText(vis, "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Ball Push", vis)
            continue

        # No ball visible — wait a bit, then search
        if obs is None:
            missed_frames += 1
            if missed_frames <= GRACE_FRAMES:
                robot.stop()
                cv2.putText(vis, f"LOST BALL ({missed_frames}/{GRACE_FRAMES})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.imshow("Ball Push", vis)
            else:
                cv2.putText(vis, "SEARCHING...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Ball Push", vis)
                time.sleep(.5)
                send_action(robot, 0, config)
                logfile.write(f"{steps},SEARCH,,,,\n")
                logfile.flush()
            continue

        # Ball found — reset grace counter
        missed_frames = 0

        # Ball visible — run the policy
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Clamp rotations: don't rotate the ball out of view
        ball_angle = obs[1]
        if action == 0 and ball_angle > 0.4:   # ball is left, don't rotate further left
            action = 2
        elif action == 1 and ball_angle < -0.4:  # ball is right, don't rotate further right
            action = 2

        logfile.write(f"{steps},{ACTION_NAMES[action]},{obs[0]:.4f},{obs[1]:.4f},{obs[2]:.4f},{obs[3]:.4f}\n")
        logfile.flush()

        cv2.putText(vis, f"{ACTION_NAMES[action]} step:{steps} gap:{obs[2]:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Ball Push", vis)

        send_action(robot, action, config)
        steps += 1

    logfile.close()
    print(f"Log saved to policy_log.txt ({steps} steps)")
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
        ball_real_diameter=bp.get("ball_radius", 0.015) * 2,  # 3cm default
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
