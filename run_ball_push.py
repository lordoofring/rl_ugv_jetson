"""
Deploy the Ball Push policy on the real robot (pure RL — no overrides).

Usage:
    python run_ball_push.py --model ball_push_ppo_final --ip <JETSON_IP>
    python run_ball_push.py --model ball_push_ppo_final --local
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

MIN_CMD_TIME = 0.25


def send_action(robot, action, config):
    """Send movement command, hold for MIN_CMD_TIME, then stop."""
    wb = config["robot"].get("wheel_base", 0.175)
    tws = config["robot"].get("turn_wheel_speed", 0.3)
    t90 = config["robot"].get("turn_time_90", 1.0)
    ms = config["robot"].get("max_speed", 0.5)

    settle = 0.5  # seconds for motors to fully stop before next frame

    if action == 0:
        w = tws / (wb / 2.0)
        robot.move(0.0, w)
        time.sleep(max(t90 * TURN_ANGLE_DEG / 90.0, MIN_CMD_TIME))
        robot.stop()
    elif action == 1:
        w = tws / (wb / 2.0)
        robot.move(0.0, -w)
        time.sleep(max(t90 * TURN_ANGLE_DEG / 90.0, MIN_CMD_TIME))
        robot.stop()
    elif action == 2:
        robot.move(ms, 0.0)
        time.sleep(max(STEP_DIST / ms, MIN_CMD_TIME))
        robot.stop()

    time.sleep(settle)


def collect_frames(get_frame, n):
    """Grab n non-None frames as fast as possible."""
    frames = []
    while len(frames) < n:
        f = get_frame()
        if f is not None:
            frames.append(f)
    return frames


def run_calibration(get_frame, observer):
    print("\n--- Calibration ---")
    print("Green=ball, Blue=tape, Yellow=gap. Q to quit.\n")
    while True:
        frames = collect_frames(get_frame, observer.n_frames)
        obs, vis = observer.observe_and_annotate_multi(frames)
        if obs is not None:
            txt = f"dist={obs[0]:.2f}m ang={math.degrees(obs[1]):.0f} gap={obs[2]:.2f}"
            cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "NO BALL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Calibrate", vis)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def run_policy(get_frame, robot, model, observer, config, wiggle=False):
    print("\n--- Running Policy (Pure RL) ---")
    print("Q=quit, SPACE=pause, R=reset\n")
    paused = False
    steps = 0

    logfile = open("policy_log.txt", "w")
    logfile.write("step,action,visible,ball_dist,ball_angle,gap_dist,gap_angle\n")

    while True:
        frames = collect_frames(get_frame, observer.n_frames)
        raw_obs, vis = observer.observe_and_annotate_multi(frames)

        # Build the 5-dim observation matching the sim env
        if raw_obs is not None:
            # FrameObserver returns [ball_dist, ball_angle, gap_dist, gap_angle]
            obs = np.array([1.0, raw_obs[0], raw_obs[1], raw_obs[2], raw_obs[3]], dtype=np.float32)
        else:
            obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord("r"):
            steps = 0
            print("--- Reset ---")

        if paused:
            robot.stop()
            cv2.putText(vis, "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Ball Push", vis)
            continue

        # Policy decides everything — search, align, approach, push
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        if wiggle and obs[0] > 0.5 and abs(obs[2]) < 0.2 and action != 2:
            action = 2

        visible = "Y" if obs[0] > 0.5 else "N"
        logfile.write(f"{steps},{ACTION_NAMES[action]},{visible},{obs[1]:.4f},{obs[2]:.4f},{obs[3]:.4f},{obs[4]:.4f}\n")
        logfile.flush()

        label = f"{ACTION_NAMES[action]} step:{steps}"
        if obs[0] > 0.5:
            label += f" d:{obs[1]:.2f} a:{math.degrees(obs[2]):.0f}"
            color = (0, 255, 0)
        else:
            label += " [blind]"
            color = (0, 0, 255)
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Ball Push", vis)

        # Ball at/past boundary — stop, don't follow it out
        gap_dist = obs[3] if obs[0] > 0.5 else 1.0
        if gap_dist < 0.05:
            print(f"\n*** BALL OUT at step {steps}! Stopping. ***")
            robot.stop()
            cv2.putText(vis, "BALL OUT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("Ball Push", vis)
            cv2.waitKey(3000)
            break

        send_action(robot, action, config)
        steps += 1

    logfile.close()
    print(f"Log saved to policy_log.txt ({steps} steps)")
    cv2.destroyAllWindows()
    robot.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ball_push_ppo_final")
    parser.add_argument("--ip", type=str, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--wiggle", action="store_true",
                        help="Force forward when ball is centered (overrides pure RL)")
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
        ball_real_diameter=bp.get("ball_radius", 0.07) * 2,
    )

    if args.calibrate:
        run_calibration(get_frame, observer)
    else:
        model = PPO.load(args.model)
        print(f"Model loaded: {args.model}")
        run_policy(get_frame, robot, model, observer, config, args.wiggle)

    if robot:
        robot.stop()
        robot.close()


if __name__ == "__main__":
    main()
