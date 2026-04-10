"""
Extract observations from a camera frame for the Ball Push task.

Detects:
  - Red ball (HSV thresholding) → ball_dist, ball_angle
  - Blue tape (HSV thresholding) → gap_dist, gap_angle

Works identically on real camera frames and Isaac Sim rendered frames.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameObserver:
    """Process a BGR camera frame into the 4-dim observation vector."""

    def __init__(
        self,
        ball_real_diameter: float = 0.10,
        focal_length: float = 280.0,  # ~120° FOV on 640px wide frame
        frame_width: int = 640,
        frame_height: int = 480,
        # Red ball HSV ranges (red wraps around in HSV)
        ball_hsv_low1=(25, 100, 80),
        ball_hsv_high1=(10, 255, 255),
        ball_hsv_low2=(165, 100, 80),
        ball_hsv_high2=(180, 255, 255),
        # Blue tape HSV range
        tape_hsv_low=(90, 80, 50),
        tape_hsv_high=(130, 255, 255),
    ):
        self.ball_real_diameter = ball_real_diameter
        self.focal_length = focal_length
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0

        self.ball_hsv_low1 = np.array(ball_hsv_low1)
        self.ball_hsv_high1 = np.array(ball_hsv_high1)
        self.ball_hsv_low2 = np.array(ball_hsv_low2)
        self.ball_hsv_high2 = np.array(ball_hsv_high2)

        self.tape_hsv_low = np.array(tape_hsv_low)
        self.tape_hsv_high = np.array(tape_hsv_high)

    def observe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract [ball_dist, ball_angle, gap_dist, gap_angle] from a BGR frame.

        Returns:
            4-element float32 array, or None if ball not visible.
        """
        ball = self._detect_ball(frame)
        if ball is None:
            return None

        ball_cx, ball_cy, ball_radius_px = ball

        # Ball distance from apparent size
        diameter_px = ball_radius_px * 2
        ball_dist = (self.ball_real_diameter * self.focal_length) / max(diameter_px, 1)

        # Ball angle from center of frame
        ball_angle = math.atan2(ball_cx - self.cx, self.focal_length)

        # Find nearest tape to the ball
        tape = self._detect_nearest_tape(frame, ball_cx, ball_cy)
        if tape is not None:
            tape_cx, tape_cy = tape
            # Gap in pixels between ball center and tape center
            gap_px = math.sqrt((ball_cx - tape_cx) ** 2 + (ball_cy - tape_cy) ** 2)
            # Normalize gap by frame diagonal for scale invariance
            frame_diag = math.sqrt(self.frame_width ** 2 + self.frame_height ** 2)
            gap_dist = gap_px / frame_diag

            # Gap angle: direction from ball to tape, relative to frame center
            gap_angle = math.atan2(tape_cx - ball_cx, self.focal_length)
        else:
            # No tape visible — ball is far from edges
            gap_dist = 1.0  # max normalized distance
            gap_angle = 0.0

        obs = np.array([ball_dist, ball_angle, gap_dist, gap_angle], dtype=np.float32)
        return obs

    def _detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect red ball. Returns (center_x, center_y, radius) in pixels or None."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.ball_hsv_low1, self.ball_hsv_high1)
        mask2 = cv2.inRange(hsv, self.ball_hsv_low2, self.ball_hsv_high2)
        mask = mask1 | mask2

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        ((cx, cy), radius) = cv2.minEnclosingCircle(largest)

        if radius < 3:
            return None

        return float(cx), float(cy), float(radius)

    def _detect_nearest_tape(self, frame: np.ndarray, ball_cx: float, ball_cy: float) -> Optional[Tuple[float, float]]:
        """Detect blue tape and return the point on tape nearest to the ball.

        Returns (tape_x, tape_y) in pixels or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.tape_hsv_low, self.tape_hsv_high)

        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find all blue tape pixels
        tape_points = np.column_stack(np.where(mask > 0))  # (row, col) format
        if len(tape_points) == 0:
            return None

        # Find the tape pixel closest to the ball center
        # tape_points is (row, col) = (y, x)
        dists = np.sqrt((tape_points[:, 1] - ball_cx) ** 2 + (tape_points[:, 0] - ball_cy) ** 2)
        nearest_idx = np.argmin(dists)
        nearest = tape_points[nearest_idx]

        return float(nearest[1]), float(nearest[0])  # (x, y)

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw detections on the frame for debugging. Returns annotated copy."""
        vis = frame.copy()

        ball = self._detect_ball(frame)
        if ball is not None:
            cx, cy, r = ball
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            cv2.putText(vis, "BALL", (int(cx - 20), int(cy - r - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            tape = self._detect_nearest_tape(frame, cx, cy)
            if tape is not None:
                tx, ty = tape
                cv2.circle(vis, (int(tx), int(ty)), 5, (255, 0, 0), -1)
                cv2.line(vis, (int(cx), int(cy)), (int(tx), int(ty)), (255, 255, 0), 2)
                gap = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                cv2.putText(vis, f"gap:{gap:.0f}px", (int(tx + 10), int(ty)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return vis
