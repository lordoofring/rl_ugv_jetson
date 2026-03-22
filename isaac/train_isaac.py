"""
Train the Ball Push agent in Isaac Sim.

Run inside the Apptainer container:
    /isaac-sim/python.sh /workspace/rl_ugv_jetson/isaac/train_isaac.py --train
    /isaac-sim/python.sh /workspace/rl_ugv_jetson/isaac/train_isaac.py --train --headless
    /isaac-sim/python.sh /workspace/rl_ugv_jetson/isaac/train_isaac.py --test
"""

import argparse
import math
import sys

# Parse args before starting SimulationApp
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--timesteps", type=int, default=300000)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--scene", type=str, default="/workspace/ball_push_scene.usd")
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Constants
ARENA_SIZE = 2.0
HALF = ARENA_SIZE / 2.0
BALL_RADIUS = 0.05
PUSH_RADIUS = 0.15
STEP_DIST = 0.05
TURN_ANGLE = math.radians(5)
MAX_STEPS = 500


class IsaacBallPushEnv(gym.Env):
    """Ball Push environment running in Isaac Sim."""

    def __init__(self, scene_path):
        super().__init__()

        self.action_space = spaces.Discrete(3)
        obs_high = np.array([ARENA_SIZE * 1.5, math.pi, ARENA_SIZE, math.pi, ARENA_SIZE], dtype=np.float32)
        obs_low = np.array([0.0, -math.pi, 0.0, -math.pi, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Load scene
        omni.usd.get_context().open_stage(scene_path)
        app.update()
        app.update()

        self.stage = omni.usd.get_context().get_stage()

        # Get references to prims
        self.ball_prim = self.stage.GetPrimAtPath("/World/Ball")
        self.robot_prim = self.stage.GetPrimAtPath("/World/Robot")

        if not self.ball_prim:
            print("ERROR: /World/Ball not found in scene!")
            sys.exit(1)
        if not self.robot_prim:
            print("ERROR: /World/Robot not found in scene!")
            sys.exit(1)

        # Robot state (tracked by us for discrete actions)
        self.robot_x = -0.5
        self.robot_y = -0.5
        self.robot_theta = 0.0
        self.ball_x = 0.3
        self.ball_y = 0.3
        self.steps = 0

        # Start simulation
        from omni.isaac.core.simulation_context import SimulationContext
        self.sim = SimulationContext(stage_units_in_meters=1.0)
        self.sim.initialize_physics()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        self.sim.stop()

        # Random robot position
        margin = 0.2
        self.robot_x = np.random.uniform(-HALF + margin, HALF - margin)
        self.robot_y = np.random.uniform(-HALF + margin, HALF - margin)
        self.robot_theta = np.random.uniform(-math.pi, math.pi)

        # Random ball position
        ball_margin = 0.15
        self.ball_x = np.random.uniform(-HALF + ball_margin, HALF - ball_margin)
        self.ball_y = np.random.uniform(-HALF + ball_margin, HALF - ball_margin)

        while self._dist_robot_to_ball() < PUSH_RADIUS * 2:
            self.ball_x = np.random.uniform(-HALF + ball_margin, HALF - ball_margin)
            self.ball_y = np.random.uniform(-HALF + ball_margin, HALF - ball_margin)

        # Set positions in sim
        self._set_prim_position(self.robot_prim, self.robot_x, self.robot_y, 0.08)
        self._set_prim_position(self.ball_prim, self.ball_x, self.ball_y, BALL_RADIUS + 0.01)

        self.sim.play()

        # Let things settle
        for _ in range(20):
            self.sim.step(render=not args.headless)

        # Read ball position after settling
        self._read_ball_position()

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)
        prev_ball_edge = self._ball_to_nearest_edge()

        # Apply action
        if action == 0:  # rotate left
            self.robot_theta += TURN_ANGLE
        elif action == 1:  # rotate right
            self.robot_theta -= TURN_ANGLE
        elif action == 2:  # forward
            self.robot_x += math.cos(self.robot_theta) * STEP_DIST
            self.robot_y += math.sin(self.robot_theta) * STEP_DIST

        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi

        # Update robot in sim
        self._set_prim_position(self.robot_prim, self.robot_x, self.robot_y, 0.08)

        # Step physics
        for _ in range(5):
            self.sim.step(render=not args.headless)

        # Check push — if robot is close to ball, nudge ball in sim
        dist = self._dist_robot_to_ball()
        if dist < PUSH_RADIUS:
            dx = self.ball_x - self.robot_x
            dy = self.ball_y - self.robot_y
            if dist > 1e-6:
                nx, ny = dx / dist, dy / dist
            else:
                nx = math.cos(self.robot_theta)
                ny = math.sin(self.robot_theta)
            push_amount = PUSH_RADIUS - dist + STEP_DIST * 0.5
            new_bx = self.ball_x + nx * push_amount
            new_by = self.ball_y + ny * push_amount
            self._set_prim_position(self.ball_prim, new_bx, new_by, BALL_RADIUS)
            for _ in range(5):
                self.sim.step(render=not args.headless)

        # Read ball position from sim
        self._read_ball_position()

        # Reward
        ball_outside = abs(self.ball_x) > HALF or abs(self.ball_y) > HALF
        curr_ball_edge = self._ball_to_nearest_edge()

        reward = -0.01
        reward += (prev_ball_edge - curr_ball_edge) * 10.0
        if ball_outside:
            reward += 100.0
        if abs(self.robot_x) > HALF or abs(self.robot_y) > HALF:
            reward -= 5.0

        done = ball_outside or self.steps >= MAX_STEPS

        info = {
            "ball_pos": (self.ball_x, self.ball_y),
            "ball_outside": ball_outside,
        }

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    # Sim helpers
    # ------------------------------------------------------------------

    def _set_prim_position(self, prim, x, y, z):
        attr = prim.GetAttribute("xformOp:translate")
        if attr:
            attr.Set(Gf.Vec3d(x, y, z))

    def _read_ball_position(self):
        attr = self.ball_prim.GetAttribute("xformOp:translate")
        if attr:
            pos = attr.Get()
            self.ball_x = float(pos[0])
            self.ball_y = float(pos[1])

    # ------------------------------------------------------------------
    # Observation helpers (same as BallPushEnv)
    # ------------------------------------------------------------------

    def _get_obs(self):
        dx = self.ball_x - self.robot_x
        dy = self.ball_y - self.robot_y
        ball_dist = math.sqrt(dx * dx + dy * dy)
        ball_angle = math.atan2(dy, dx) - self.robot_theta
        ball_angle = (ball_angle + math.pi) % (2 * math.pi) - math.pi
        ball_to_edge = self._ball_to_nearest_edge()
        ball_to_edge_angle = self._nearest_edge_angle()
        robot_to_edge = min(HALF - abs(self.robot_x), HALF - abs(self.robot_y))
        return np.array([ball_dist, ball_angle, ball_to_edge, ball_to_edge_angle, robot_to_edge], dtype=np.float32)

    def _dist_robot_to_ball(self):
        return math.sqrt((self.robot_x - self.ball_x) ** 2 + (self.robot_y - self.ball_y) ** 2)

    def _ball_to_nearest_edge(self):
        return min(HALF - abs(self.ball_x), HALF - abs(self.ball_y))

    def _nearest_edge_angle(self):
        dists = {"r": HALF - self.ball_x, "l": HALF + self.ball_x, "t": HALF - self.ball_y, "b": HALF + self.ball_y}
        nearest = min(dists, key=dists.get)
        targets = {"r": (HALF, self.ball_y), "l": (-HALF, self.ball_y), "t": (self.ball_x, HALF), "b": (self.ball_x, -HALF)}
        tx, ty = targets[nearest]
        world_angle = math.atan2(ty - self.robot_y, tx - self.robot_x)
        rel = world_angle - self.robot_theta
        return (rel + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    env = IsaacBallPushEnv(args.scene)

    if args.train:
        print(f"Training in Isaac Sim ({'headless' if args.headless else 'with GUI'})...")
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10)
        cb = CheckpointCallback(save_freq=25000, save_path="/workspace/models/", name_prefix="ball_push_isaac")
        model.learn(total_timesteps=args.timesteps, callback=cb)
        model.save("/workspace/ball_push_isaac_final")
        print("Done! Model saved to /workspace/ball_push_isaac_final.zip")

    if args.test:
        model_path = args.model or "/workspace/ball_push_isaac_final"
        model = PPO.load(model_path)
        print(f"Testing: {model_path}")

        for ep in range(10):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                ep_reward += reward
            result = "BALL OUT!" if info.get("ball_outside") else "timeout"
            print(f"  Episode {ep + 1}: {result}  Reward: {ep_reward:.1f}  Steps: {env.steps}")

    app.close()


if __name__ == "__main__":
    main()
