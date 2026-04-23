"""
Ball Push Environment (Pure RL with FOV).

The agent must push a red ball outside a square boundary.
The ball is only visible when within the robot's field of view.
The agent must learn to search, align, approach, and push — no hardcoded overrides.

Actions:
    0 = Rotate left 5°
    1 = Rotate right 5°
    2 = Move forward
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import yaml
from typing import Optional

from ugv_rl.core.robot_interface import RobotInterface
from ugv_rl.controllers.mock_robot import MockRobot

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2

ACTION_NAMES = {0: "Rotate Left", 1: "Rotate Right", 2: "Forward"}

TURN_ANGLE = math.radians(5)  # 5 degrees per turn action


class BallPushEnv(gym.Env):
    """
    2D ball push with realistic FOV constraints.

    The robot can only observe the ball when it falls within a limited
    field of view (default ±30°). When the ball is outside the FOV,
    the observation is zeroed out and a flag indicates "not visible".

    Observation (5-dim):
        [ball_visible, ball_dist, ball_angle, gap_dist, gap_angle]

        ball_visible: 1.0 if ball is in FOV, 0.0 if not
        ball_dist:    distance to ball (0 if not visible)
        ball_angle:   bearing to ball relative to heading (0 if not visible)
        gap_dist:     ball-to-edge normalized [0,1] (0 if not visible)
        gap_angle:    direction ball→edge relative to heading (0 if not visible)

    Actions (Discrete 3):
        0 = rotate left 5°
        1 = rotate right 5°
        2 = move forward
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config_path: str = "config.yaml", robot: RobotInterface = None):
        super().__init__()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        bp_cfg = self.config.get("ball_push", {})
        self.arena_size = bp_cfg.get("arena_size", 2.0)
        self.ball_radius = bp_cfg.get("ball_radius", 0.07)   # 14cm diameter ball
        self.step_dist = bp_cfg.get("step_dist", 0.05)
        self.push_radius = bp_cfg.get("push_radius", 0.20)   # larger for 14cm ball
        self.max_steps = bp_cfg.get("max_steps", 500)
        self.robot_radius = bp_cfg.get("robot_radius", 0.12)

        # FOV half-angle in radians (±30° = 60° total, matching cropped camera)
        self.fov_half = math.radians(bp_cfg.get("fov_half_deg", 30))

        # Camera params — must match FrameObserver exactly
        self.focal_length = 107.0
        self.cam_cx = 80.0    # center of 160px cropped frame
        self.frame_diag = math.sqrt(160**2 + 240**2)

        self.half_arena = self.arena_size / 2.0

        self.action_space = spaces.Discrete(3)

        # Observation: [ball_visible, ball_dist, ball_angle, gap_dist, gap_angle]
        obs_high = np.array([
            1.0,                      # ball_visible (0 or 1)
            self.arena_size * 1.5,    # ball_dist
            math.pi,                  # ball_angle
            1.0,                      # gap_dist
            math.pi,                  # gap_angle
        ], dtype=np.float32)
        obs_low = np.array([0.0, 0.0, -math.pi, 0.0, -math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.robot = robot if robot else MockRobot()

        # State
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.steps = 0
        self.last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = None

        margin = self.robot_radius + 0.05
        self.robot_x = self.np_random.uniform(-self.half_arena + margin, self.half_arena - margin)
        self.robot_y = self.np_random.uniform(-self.half_arena + margin, self.half_arena - margin)
        self.robot_theta = self.np_random.uniform(-math.pi, math.pi)

        ball_margin = self.ball_radius + 0.1
        self.ball_x = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)
        self.ball_y = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)

        while self._dist_robot_to_ball() < self.push_radius * 2:
            self.ball_x = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)
            self.ball_y = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)

        self.robot.reset(self.robot_x, self.robot_y, self.robot_theta)

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)
        self.last_action = action

        prev_ball_dist = self._dist_robot_to_ball()
        prev_ball_edge_dist = self._ball_to_nearest_edge()

        if action == ACTION_LEFT:
            self.robot_theta += TURN_ANGLE
        elif action == ACTION_RIGHT:
            self.robot_theta -= TURN_ANGLE
        elif action == ACTION_FORWARD:
            self.robot_x += math.cos(self.robot_theta) * self.step_dist
            self.robot_y += math.sin(self.robot_theta) * self.step_dist

        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi

        self._apply_push()

        ball_outside = self._ball_is_outside()
        curr_ball_dist = self._dist_robot_to_ball()
        curr_ball_edge_dist = self._ball_to_nearest_edge()
        ball_visible = self._ball_in_fov()
        robot_edge_dist = self._robot_to_nearest_edge()

        # --- Reward ---
        reward = -0.01  # step penalty

        # Approach: reward getting closer to ball
        approach = prev_ball_dist - curr_ball_dist
        reward += approach * 5.0

        # Push: reward pushing ball toward edge
        edge_progress = prev_ball_edge_dist - curr_ball_edge_dist
        reward += edge_progress * 10.0

        # Ball out = success
        if ball_outside:
            reward += 100.0

        # Robot too close to boundary — graduated penalty
        if robot_edge_dist < 0.1:
            reward -= 2.0
        # Robot leaves arena — harsh penalty + end episode
        robot_outside = (abs(self.robot_x) > self.half_arena or
                         abs(self.robot_y) > self.half_arena)
        if robot_outside:
            reward -= 10.0

        # Small penalty when ball not in FOV — encourages searching
        if not ball_visible:
            reward -= 0.02

        done = ball_outside or robot_outside or self.steps >= self.max_steps

        info = {
            "ball_pos": (self.ball_x, self.ball_y),
            "robot_pos": (self.robot_x, self.robot_y),
            "ball_outside": ball_outside,
            "ball_visible": ball_visible,
            "action": ACTION_NAMES.get(action, "?"),
        }

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    # FOV
    # ------------------------------------------------------------------

    def _ball_in_fov(self):
        """Check if ball is within the robot's field of view."""
        dx = self.ball_x - self.robot_x
        dy = self.ball_y - self.robot_y
        ball_world_angle = math.atan2(dy, dx)
        rel_angle = ball_world_angle - self.robot_theta
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        return abs(rel_angle) <= self.fov_half

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _apply_push(self):
        dist = self._dist_robot_to_ball()
        if dist < self.push_radius:
            dx = self.ball_x - self.robot_x
            dy = self.ball_y - self.robot_y
            if dist > 1e-6:
                nx, ny = dx / dist, dy / dist
            else:
                nx, ny = math.cos(self.robot_theta), math.sin(self.robot_theta)
            push_dist = self.push_radius - dist + self.step_dist * 0.5
            self.ball_x += nx * push_dist
            self.ball_y += ny * push_dist

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _dist_robot_to_ball(self):
        return math.sqrt((self.robot_x - self.ball_x) ** 2 +
                         (self.robot_y - self.ball_y) ** 2)

    def _ball_to_nearest_edge(self):
        return min(
            self.half_arena - abs(self.ball_x),
            self.half_arena - abs(self.ball_y),
        )

    def _ball_is_outside(self):
        return (abs(self.ball_x) > self.half_arena or
                abs(self.ball_y) > self.half_arena)

    def _robot_to_nearest_edge(self):
        return min(
            self.half_arena - abs(self.robot_x),
            self.half_arena - abs(self.robot_y),
        )

    def _nearest_edge_angle_from_ball(self):
        dists = {
            "right": self.half_arena - self.ball_x,
            "left": self.half_arena + self.ball_x,
            "top": self.half_arena - self.ball_y,
            "bottom": self.half_arena + self.ball_y,
        }
        nearest = min(dists, key=dists.get)

        if nearest == "right":
            tx, ty = self.half_arena, self.ball_y
        elif nearest == "left":
            tx, ty = -self.half_arena, self.ball_y
        elif nearest == "top":
            tx, ty = self.ball_x, self.half_arena
        else:
            tx, ty = self.ball_x, -self.half_arena

        world_angle = math.atan2(ty - self.robot_y, tx - self.robot_x)
        rel_angle = world_angle - self.robot_theta
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        return rel_angle

    # ------------------------------------------------------------------
    # Camera projection
    # ------------------------------------------------------------------

    def _world_to_image_x(self, world_x, world_y):
        """Project a world point to camera image x. Returns None if behind camera."""
        dx = world_x - self.robot_x
        dy = world_y - self.robot_y
        forward = dx * math.cos(self.robot_theta) + dy * math.sin(self.robot_theta)
        right = dx * math.sin(self.robot_theta) - dy * math.cos(self.robot_theta)
        if forward <= 0.01:
            return None
        return self.cam_cx + (right / forward) * self.focal_length

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        if not self._ball_in_fov():
            return np.zeros(5, dtype=np.float32)

        dx = self.ball_x - self.robot_x
        dy = self.ball_y - self.robot_y
        ball_dist = math.sqrt(dx * dx + dy * dy)
        ball_world_angle = math.atan2(dy, dx)
        ball_angle = ball_world_angle - self.robot_theta
        ball_angle = (ball_angle + math.pi) % (2 * math.pi) - math.pi

        # Project ball and nearest edge point into image space
        ball_px = self._world_to_image_x(self.ball_x, self.ball_y)

        # Find nearest edge target point (same logic as before)
        dists = {
            "right":  self.half_arena - self.ball_x,
            "left":   self.half_arena + self.ball_x,
            "top":    self.half_arena - self.ball_y,
            "bottom": self.half_arena + self.ball_y,
        }
        nearest = min(dists, key=dists.get)
        if nearest == "right":   tx, ty = self.half_arena,  self.ball_y
        elif nearest == "left":  tx, ty = -self.half_arena, self.ball_y
        elif nearest == "top":   tx, ty = self.ball_x,      self.half_arena
        else:                    tx, ty = self.ball_x,      -self.half_arena

        tape_px = self._world_to_image_x(tx, ty)

        if ball_px is not None and tape_px is not None:
            gap_pixels = abs(ball_px - tape_px)
            gap_dist = gap_pixels / self.frame_diag
            gap_angle = math.atan2(tape_px - ball_px, self.focal_length)
        else:
            gap_dist = 1.0
            gap_angle = 0.0

        return np.array([1.0, ball_dist, ball_angle, gap_dist, gap_angle], dtype=np.float32)

    def render(self, mode="human"):
        pass
