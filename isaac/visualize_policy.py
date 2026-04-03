"""
Visualize a trained Ball Push policy in Isaac Sim.
Paste this entire script into the Isaac Sim Script Editor and click Run.

Make sure:
  1. The ball_push_scene.usd is loaded
  2. The model file exists at /workspace/ball_push_isaac.zip
"""

import omni.usd
import omni.timeline
import omni.kit.app
from pxr import Gf
import numpy as np
import math
import asyncio

from stable_baselines3 import PPO

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

ball_prim = stage.GetPrimAtPath("/World/Ball")
robot_prim = stage.GetPrimAtPath("/World/Robot")

# --- Config ---
HALF = 1.0
PUSH_RADIUS = 0.10  # tighter push radius so ball visually touches robot
STEP_DIST = 0.05
TURN_ANGLE = math.radians(5)
NUM_EPISODES = 5
MAX_STEPS = 500

# --- Load model ---
model = PPO.load("/workspace/ball_push_isaac")
print("Model loaded!")


def set_pos(prim, x, y, z):
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))


def set_robot_pose(x, y, z, theta):
    robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
    w = math.cos(theta / 2)
    qz = math.sin(theta / 2)
    robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(w, 0, 0, qz))


def get_obs(rx, ry, rtheta, bx, by):
    dx, dy = bx - rx, by - ry
    bd = math.sqrt(dx * dx + dy * dy)
    ba = math.atan2(dy, dx) - rtheta
    ba = (ba + math.pi) % (2 * math.pi) - math.pi
    be = min(HALF - abs(bx), HALF - abs(by))
    d = {"r": HALF - bx, "l": HALF + bx, "t": HALF - by, "b": HALF + by}
    n = min(d, key=d.get)
    t = {"r": (HALF, by), "l": (-HALF, by), "t": (bx, HALF), "b": (bx, -HALF)}
    tx, ty = t[n]
    a = math.atan2(ty - ry, tx - rx) - rtheta
    bea = (a + math.pi) % (2 * math.pi) - math.pi
    re = min(HALF - abs(rx), HALF - abs(ry))
    return np.array([bd, ba, be, bea, re], dtype=np.float32)


async def run_episodes():
    timeline.play()

    for ep in range(NUM_EPISODES):
        rx = np.random.uniform(-0.8, 0.8)
        ry = np.random.uniform(-0.8, 0.8)
        rtheta = np.random.uniform(-math.pi, math.pi)
        bx = np.random.uniform(-0.7, 0.7)
        by = np.random.uniform(-0.7, 0.7)

        # Ensure ball not on top of robot
        while math.sqrt((rx - bx) ** 2 + (ry - by) ** 2) < PUSH_RADIUS * 3:
            bx = np.random.uniform(-0.7, 0.7)
            by = np.random.uniform(-0.7, 0.7)

        set_robot_pose(rx, ry, 0.08, rtheta)
        set_pos(ball_prim, bx, by, 0.05)
        print(f"\n--- Episode {ep + 1} ---")
        print(f"  Robot: ({rx:.2f}, {ry:.2f})  Ball: ({bx:.2f}, {by:.2f})")

        for step in range(MAX_STEPS):
            obs = get_obs(rx, ry, rtheta, bx, by)
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

            # Push ball if close enough
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

            # Yield to Isaac Sim so the viewport updates
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

        if abs(bx) <= HALF and abs(by) <= HALF:
            print(f"  Timeout after {MAX_STEPS} steps")

        # Pause between episodes so you can see the result
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

    timeline.stop()
    print("\nAll episodes done!")


asyncio.ensure_future(run_episodes())
