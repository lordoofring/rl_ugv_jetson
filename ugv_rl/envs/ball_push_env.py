"""
Ball Push Environment.

The agent must push a red ball outside a square boundary.
Observation is ego-centric (relative to the robot) — no global position needed.

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
    Continuous 2D environment where the robot pushes a ball out of a square region.

    The square is centered at the origin with side length `arena_size`.
    The ball and robot start at random positions inside the square.

    Observation (5-dim, all ego-centric):
        [ball_dist, ball_angle, ball_to_edge_dist, ball_to_edge_angle, robot_to_edge_dist]

    Actions (Discrete 3):
        0 = rotate left 5°
        1 = rotate right 5°
        2 = move forward `step_dist` meters
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config_path: str = "config.yaml", robot: RobotInterface = None):
        super().__init__()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        bp_cfg = self.config.get("ball_push", {})
        self.arena_size = bp_cfg.get("arena_size", 2.0)       # side length in meters
        self.ball_radius = bp_cfg.get("ball_radius", 0.05)    # ball radius in meters
        self.step_dist = bp_cfg.get("step_dist", 0.05)        # forward move distance
        self.push_radius = bp_cfg.get("push_radius", 0.12)    # robot-ball contact distance
        self.max_steps = bp_cfg.get("max_steps", 500)
        self.robot_radius = bp_cfg.get("robot_radius", 0.12)  # approximate robot body radius

        self.half_arena = self.arena_size / 2.0

        # Action space: rotate left, rotate right, forward
        self.action_space = spaces.Discrete(3)

        # Observation: [ball_dist, ball_angle, ball_to_edge_dist, ball_to_edge_angle, robot_to_edge_dist]
        # All values are bounded reasonably
        obs_high = np.array([
            self.arena_size * 1.5,    # ball_dist (max ~ diagonal)
            math.pi,                  # ball_angle (-pi to pi)
            self.arena_size,          # ball_to_edge_dist
            math.pi,                  # ball_to_edge_angle
            self.arena_size,          # robot_to_edge_dist
        ], dtype=np.float32)
        obs_low = np.array([0.0, -math.pi, 0.0, -math.pi, 0.0], dtype=np.float32)
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

        # Random robot position inside the arena (with margin)
        margin = self.robot_radius + 0.05
        self.robot_x = self.np_random.uniform(-self.half_arena + margin, self.half_arena - margin)
        self.robot_y = self.np_random.uniform(-self.half_arena + margin, self.half_arena - margin)
        self.robot_theta = self.np_random.uniform(-math.pi, math.pi)

        # Random ball position inside the arena (with margin from edges so it's not trivial)
        ball_margin = self.ball_radius + 0.1
        self.ball_x = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)
        self.ball_y = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)

        # Ensure ball isn't spawned right on top of robot
        while self._dist_robot_to_ball() < self.push_radius * 2:
            self.ball_x = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)
            self.ball_y = self.np_random.uniform(-self.half_arena + ball_margin, self.half_arena - ball_margin)

        self.robot.reset(self.robot_x, self.robot_y, self.robot_theta)

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)
        self.last_action = action

        prev_ball_edge_dist = self._ball_to_nearest_edge()

        if action == ACTION_LEFT:
            self.robot_theta += TURN_ANGLE
        elif action == ACTION_RIGHT:
            self.robot_theta -= TURN_ANGLE
        elif action == ACTION_FORWARD:
            new_x = self.robot_x + math.cos(self.robot_theta) * self.step_dist
            new_y = self.robot_y + math.sin(self.robot_theta) * self.step_dist
            self.robot_x = new_x
            self.robot_y = new_y

        # Normalize theta
        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi

        # Check if robot pushes the ball
        self._apply_push()

        # Check if ball is outside the arena
        ball_outside = self._ball_is_outside()
        curr_ball_edge_dist = self._ball_to_nearest_edge()

        # --- Reward ---
        reward = -0.01  # step penalty

        # Reward for pushing ball closer to edge
        edge_progress = prev_ball_edge_dist - curr_ball_edge_dist
        reward += edge_progress * 10.0

        # Big reward for getting the ball out
        if ball_outside:
            reward += 100.0

        # Penalty if robot leaves the arena
        robot_outside = (abs(self.robot_x) > self.half_arena or
                         abs(self.robot_y) > self.half_arena)
        if robot_outside:
            reward -= 5.0

        done = ball_outside or self.steps >= self.max_steps

        info = {
            "ball_pos": (self.ball_x, self.ball_y),
            "robot_pos": (self.robot_x, self.robot_y),
            "ball_outside": ball_outside,
            "action": ACTION_NAMES.get(action, "?"),
        }

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _apply_push(self):
        """If robot is close enough to the ball, push it away."""
        dist = self._dist_robot_to_ball()
        if dist < self.push_radius:
            # Push direction: from robot center to ball center
            dx = self.ball_x - self.robot_x
            dy = self.ball_y - self.robot_y
            if dist > 1e-6:
                nx, ny = dx / dist, dy / dist
            else:
                nx, ny = math.cos(self.robot_theta), math.sin(self.robot_theta)

            # Push the ball so it's just outside the contact radius
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
        """Shortest distance from ball center to any arena edge."""
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
        """Angle from robot to the point on the nearest edge closest to the ball."""
        # Find which edge is closest to the ball
        dists = {
            "right": self.half_arena - self.ball_x,
            "left": self.half_arena + self.ball_x,
            "top": self.half_arena - self.ball_y,
            "bottom": self.half_arena + self.ball_y,
        }
        nearest = min(dists, key=dists.get)

        # Target point: project ball onto that edge
        if nearest == "right":
            tx, ty = self.half_arena, self.ball_y
        elif nearest == "left":
            tx, ty = -self.half_arena, self.ball_y
        elif nearest == "top":
            tx, ty = self.ball_x, self.half_arena
        else:
            tx, ty = self.ball_x, -self.half_arena

        # Angle from robot to that target point, relative to robot heading
        world_angle = math.atan2(ty - self.robot_y, tx - self.robot_x)
        rel_angle = world_angle - self.robot_theta
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        return rel_angle

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        # Ball distance and angle relative to robot
        dx = self.ball_x - self.robot_x
        dy = self.ball_y - self.robot_y
        ball_dist = math.sqrt(dx * dx + dy * dy)
        ball_world_angle = math.atan2(dy, dx)
        ball_angle = ball_world_angle - self.robot_theta
        ball_angle = (ball_angle + math.pi) % (2 * math.pi) - math.pi

        ball_to_edge = self._ball_to_nearest_edge()
        ball_to_edge_angle = self._nearest_edge_angle_from_ball()
        robot_to_edge = self._robot_to_nearest_edge()

        obs = np.array([
            ball_dist,
            ball_angle,
            ball_to_edge,
            ball_to_edge_angle,
            robot_to_edge,
        ], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        pass
