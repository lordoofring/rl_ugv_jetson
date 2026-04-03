"""
Train the Ball Push PPO policy inside Isaac Sim GUI.
Paste this entire script into the Isaac Sim Script Editor and click Run.

Make sure the ball_push_scene.usd is loaded first.
The GUI will freeze during training but the console shows progress.
Model is saved to /workspace/ball_push_isaac.zip when done.
"""

import omni.usd
import omni.timeline
from pxr import Gf
import numpy as np
import math

from stable_baselines3 import PPO

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

ball_prim = stage.GetPrimAtPath("/World/Ball")
robot_prim = stage.GetPrimAtPath("/World/Robot")

HALF = 1.0
PUSH_RADIUS = 0.10
STEP_DIST = 0.05
TURN_ANGLE = math.radians(5)
MAX_STEPS = 500
TOTAL_TIMESTEPS = 200000

import gymnasium as gym
from gymnasium import spaces


class IsaacBallPush(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        hi = np.array([3.0, math.pi, 2.0, math.pi, 2.0], dtype=np.float32)
        lo = np.array([0.0, -math.pi, 0.0, -math.pi, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.rx = self.ry = self.rtheta = 0.0
        self.bx = self.by = 0.0
        self.steps = 0

    def _set(self, prim, x, y, z):
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        m = 0.2
        self.rx = np.random.uniform(-HALF + m, HALF - m)
        self.ry = np.random.uniform(-HALF + m, HALF - m)
        self.rtheta = np.random.uniform(-math.pi, math.pi)
        bm = 0.15
        self.bx = np.random.uniform(-HALF + bm, HALF - bm)
        self.by = np.random.uniform(-HALF + bm, HALF - bm)
        while math.sqrt((self.rx - self.bx) ** 2 + (self.ry - self.by) ** 2) < PUSH_RADIUS * 3:
            self.bx = np.random.uniform(-HALF + bm, HALF - bm)
            self.by = np.random.uniform(-HALF + bm, HALF - bm)
        self._set(robot_prim, self.rx, self.ry, 0.08)
        self._set(ball_prim, self.bx, self.by, 0.05)
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)
        prev_edge = self._ball_edge()

        if action == 0:
            self.rtheta += TURN_ANGLE
        elif action == 1:
            self.rtheta -= TURN_ANGLE
        elif action == 2:
            self.rx += math.cos(self.rtheta) * STEP_DIST
            self.ry += math.sin(self.rtheta) * STEP_DIST
        self.rtheta = (self.rtheta + math.pi) % (2 * math.pi) - math.pi

        self._set(robot_prim, self.rx, self.ry, 0.08)

        # Push
        dist = math.sqrt((self.rx - self.bx) ** 2 + (self.ry - self.by) ** 2)
        if dist < PUSH_RADIUS:
            dx, dy = self.bx - self.rx, self.by - self.ry
            if dist > 1e-6:
                nx, ny = dx / dist, dy / dist
            else:
                nx, ny = math.cos(self.rtheta), math.sin(self.rtheta)
            push = PUSH_RADIUS - dist + STEP_DIST * 0.5
            self.bx += nx * push
            self.by += ny * push
            self._set(ball_prim, self.bx, self.by, 0.05)

        out = abs(self.bx) > HALF or abs(self.by) > HALF
        curr_edge = self._ball_edge()
        reward = -0.01 + (prev_edge - curr_edge) * 10.0
        if out:
            reward += 100.0
        if abs(self.rx) > HALF or abs(self.ry) > HALF:
            reward -= 5.0
        done = out or self.steps >= MAX_STEPS
        return self._obs(), reward, done, False, {"ball_outside": out}

    def _obs(self):
        dx, dy = self.bx - self.rx, self.by - self.ry
        bd = math.sqrt(dx * dx + dy * dy)
        ba = math.atan2(dy, dx) - self.rtheta
        ba = (ba + math.pi) % (2 * math.pi) - math.pi
        be = self._ball_edge()
        bea = self._edge_angle()
        re = min(HALF - abs(self.rx), HALF - abs(self.ry))
        return np.array([bd, ba, be, bea, re], dtype=np.float32)

    def _ball_edge(self):
        return min(HALF - abs(self.bx), HALF - abs(self.by))

    def _edge_angle(self):
        d = {"r": HALF - self.bx, "l": HALF + self.bx, "t": HALF - self.by, "b": HALF + self.by}
        n = min(d, key=d.get)
        t = {"r": (HALF, self.by), "l": (-HALF, self.by), "t": (self.bx, HALF), "b": (self.bx, -HALF)}
        tx, ty = t[n]
        a = math.atan2(ty - self.ry, tx - self.rx) - self.rtheta
        return (a + math.pi) % (2 * math.pi) - math.pi


# --- Train ---
print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
print("GUI will freeze — watch the terminal for PPO progress.")
timeline.play()

env = IsaacBallPush()
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("/workspace/ball_push_isaac")

timeline.stop()
print("Model saved to /workspace/ball_push_isaac.zip")
print("Now paste visualize_policy.py in the Script Editor to watch it!")
