"""
Visualize the camera-trained Ball Push policy in Isaac Sim.
Paste in Script Editor after loading ball_push_scene.usd.
GUI stays responsive — watch the robot push the ball in real-time.
"""

import omni.usd
import omni.timeline
import omni.kit.app
from pxr import Gf
import numpy as np
import math
import asyncio
import sys

sys.path.insert(0, "/workspace/rl_ugv_jetson")

from stable_baselines3 import PPO
from ugv_rl.vision.frame_observer import FrameObserver
from omni.isaac.sensor import Camera as IsaacCamera

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

ball_prim = stage.GetPrimAtPath("/World/Ball")
robot_prim = stage.GetPrimAtPath("/World/Robot")

HALF = 1.0
PUSH_RADIUS = 0.10
STEP_DIST = 0.05
TURN_ANGLE = math.radians(5)

# Setup camera
CAM_WIDTH, CAM_HEIGHT = 640, 480
camera = IsaacCamera(prim_path="/World/Robot/base_footprint/base_link/FrontCamera", resolution=(CAM_WIDTH, CAM_HEIGHT))
camera.initialize()

observer = FrameObserver(frame_width=CAM_WIDTH, frame_height=CAM_HEIGHT)

model = PPO.load("/workspace/ball_push_camera")
print("Model loaded!")


def set_pos(prim, x, y, z):
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))


def set_robot_pose(x, y, z, theta):
    robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
    w = math.cos(theta / 2)
    qz = math.sin(theta / 2)
    robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(w, 0, 0, qz))


last_obs = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)


def get_camera_obs():
    global last_obs
    frame = camera.get_rgba()
    if frame is None:
        return last_obs
    bgr = frame[:, :, :3][:, :, ::-1].copy()
    obs = observer.observe(bgr)
    if obs is not None:
        last_obs = obs
        return obs
    return last_obs


async def run_episodes():
    timeline.play()

    for ep in range(5):
        rx = np.random.uniform(-0.8, 0.8)
        ry = np.random.uniform(-0.8, 0.8)
        rtheta = np.random.uniform(-math.pi, math.pi)
        bx = np.random.uniform(-0.7, 0.7)
        by = np.random.uniform(-0.7, 0.7)

        while math.sqrt((rx - bx) ** 2 + (ry - by) ** 2) < PUSH_RADIUS * 3:
            bx = np.random.uniform(-0.7, 0.7)
            by = np.random.uniform(-0.7, 0.7)

        set_robot_pose(rx, ry, 0.08, rtheta)
        set_pos(ball_prim, bx, by, 0.05)
        print(f"\n--- Episode {ep + 1} ---")

        # Let camera settle
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        for step in range(500):
            obs = get_camera_obs()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            if action == 0:
                rtheta += TURN_ANGLE
            elif action == 1:
                rtheta -= TURN_ANGLE
            elif action == 2:
                rx += math.cos(rtheta) * STEP_DIST
                ry += math.sin(rtheta) * STEP_DIST
            rtheta = (rtheta + math.pi) % (2 * math.pi) - math.pi

            set_robot_pose(rx, ry, 0.08, rtheta)

            dist = math.sqrt((rx - bx) ** 2 + (ry - by) ** 2)
            if dist < PUSH_RADIUS:
                dx, dy = bx - rx, by - ry
                if dist > 1e-6:
                    nx, ny = dx / dist, dy / dist
                else:
                    nx, ny = math.cos(rtheta), math.sin(rtheta)
                push = PUSH_RADIUS - dist + STEP_DIST * 0.5
                bx += nx * push
                by += ny * push
                set_pos(ball_prim, bx, by, 0.05)

            if abs(bx) > HALF or abs(by) > HALF:
                print(f"  BALL OUT at step {step}!")
                break

            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

        if abs(bx) <= HALF and abs(by) <= HALF:
            print(f"  Timeout after 500 steps")

        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

    timeline.stop()
    print("\nDone!")


asyncio.ensure_future(run_episodes())
