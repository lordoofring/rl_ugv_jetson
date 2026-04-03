"""
Train Ball Push with camera-based observations in Isaac Sim.
Paste in Script Editor after loading ball_push_scene.usd + adding camera.

This trains using the SAME CV pipeline that runs on the real robot:
  Camera frame → detect red ball → detect blue tape → 4 observations → PPO

The GUI will freeze during training. Watch the terminal for progress.
"""

import omni.usd
import omni.timeline
from pxr import Gf
import numpy as np
import math
import sys

sys.path.insert(0, "/workspace/rl_ugv_jetson")

from stable_baselines3 import PPO
from ugv_rl.vision.frame_observer import FrameObserver

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

ball_prim = stage.GetPrimAtPath("/World/Ball")
robot_prim = stage.GetPrimAtPath("/World/Robot")

# --- Config ---
HALF = 1.0
PUSH_RADIUS = 0.10
STEP_DIST = 0.05
TURN_ANGLE = math.radians(5)
MAX_STEPS = 500
TOTAL_TIMESTEPS = 200000

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_PATH = "/World/Robot/base_footprint/base_link/FrontCamera"

# --- Setup camera rendering ---
from omni.isaac.sensor import Camera as IsaacCamera

camera = IsaacCamera(
    prim_path=CAM_PATH,
    resolution=(CAM_WIDTH, CAM_HEIGHT),
)
camera.initialize()

# --- Frame observer (same CV pipeline as real robot) ---
observer = FrameObserver(
    frame_width=CAM_WIDTH,
    frame_height=CAM_HEIGHT,
    focal_length=280.0,  # matches wide FOV OV5647 camera
    ball_real_diameter=0.10,
)

import gymnasium as gym
from gymnasium import spaces


class CameraBallPush(gym.Env):
    """Ball Push env with camera-based observations from Isaac Sim."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        # [ball_dist, ball_angle, gap_dist, gap_angle]
        hi = np.array([5.0, math.pi, 1.0, math.pi], dtype=np.float32)
        lo = np.array([0.0, -math.pi, 0.0, -math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.rx = self.ry = self.rtheta = 0.0
        self.bx = self.by = 0.0
        self.steps = 0
        # Fallback obs when ball not visible
        self._last_obs = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def _set_pos(self, prim, x, y, z):
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))

    def _set_robot_pose(self, x, y, z, theta):
        robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
        w = math.cos(theta / 2)
        qz = math.sin(theta / 2)
        robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(w, 0, 0, qz))

    def _get_camera_obs(self):
        """Render camera frame and extract observations via CV."""
        frame = camera.get_rgba()
        if frame is None:
            return self._last_obs

        # Convert RGBA to BGR for OpenCV
        bgr = frame[:, :, :3][:, :, ::-1].copy()

        obs = observer.observe(bgr)
        if obs is not None:
            self._last_obs = obs
            return obs
        return self._last_obs

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

        self._set_robot_pose(self.rx, self.ry, 0.08, self.rtheta)
        self._set_pos(ball_prim, self.bx, self.by, 0.05)

        # Render a few frames so camera picks up the new positions
        for _ in range(5):
            camera.get_rgba()

        return self._get_camera_obs(), {}

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

        self._set_robot_pose(self.rx, self.ry, 0.08, self.rtheta)

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
            self._set_pos(ball_prim, self.bx, self.by, 0.05)

        # Get observation from camera
        obs = self._get_camera_obs()

        out = abs(self.bx) > HALF or abs(self.by) > HALF
        curr_edge = self._ball_edge()
        reward = -0.01 + (prev_edge - curr_edge) * 10.0
        if out:
            reward += 100.0
        if abs(self.rx) > HALF or abs(self.ry) > HALF:
            reward -= 5.0
        done = out or self.steps >= MAX_STEPS
        return obs, reward, done, False, {"ball_outside": out}

    def _ball_edge(self):
        return min(HALF - abs(self.bx), HALF - abs(self.by))


# --- Train ---
print(f"Training with camera observations for {TOTAL_TIMESTEPS} timesteps...")
print("GUI will freeze. Watch terminal for progress.")
timeline.play()

env = CameraBallPush()

# Quick test: verify camera returns frames
test_obs, _ = env.reset()
print(f"Test observation: {test_obs}")
print(f"  ball_dist={test_obs[0]:.2f}, ball_angle={test_obs[1]:.2f}")
print(f"  gap_dist={test_obs[2]:.2f}, gap_angle={test_obs[3]:.2f}")

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("/workspace/ball_push_camera")

timeline.stop()
print("Model saved to /workspace/ball_push_camera.zip")
print("This model uses the SAME CV pipeline as the real robot!")
