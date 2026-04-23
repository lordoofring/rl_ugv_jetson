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
        ball_real_diameter: float = 0.14,  # 14cm diameter ball
        focal_length: float = 107.0,  # calibrated: 60cm true dist, 14cm ball, 12.5px radius
        frame_width: int = 320,
        frame_height: int = 240,
        ball_hsv_low1=(25, 100, 80),
        ball_hsv_high1=(10, 255, 255),
        ball_hsv_low2=(165, 100, 80),
        ball_hsv_high2=(180, 255, 255),
        tape_hsv_low=(90, 80, 50),
        tape_hsv_high=(130, 255, 255),
    ):
        self.ball_real_diameter = ball_real_diameter
        self.focal_length = focal_length
        self.frame_width = frame_width
        self.frame_height = frame_height

        # After _crop_center(), frame is 50% of original width
        cropped_w = frame_width // 2
        self.cx = cropped_w / 2.0
        self.cy = frame_height / 2.0
        self.frame_diag = math.sqrt(cropped_w ** 2 + frame_height ** 2)

        self.ball_hsv_low1 = np.array(ball_hsv_low1)
        self.ball_hsv_high1 = np.array(ball_hsv_high1)
        self.ball_hsv_low2 = np.array(ball_hsv_low2)
        self.ball_hsv_high2 = np.array(ball_hsv_high2)

        self.tape_hsv_low = np.array(tape_hsv_low)
        self.tape_hsv_high = np.array(tape_hsv_high)

    def _crop_center(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        margin = w // 4
        return frame[:, margin:w - margin]

    def _process(self, frame: np.ndarray):
        """Run all detections once. Returns (ball, tape, hsv) or (None, None, hsv)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ball detection
        mask1 = cv2.inRange(hsv, self.ball_hsv_low1, self.ball_hsv_high1)
        mask2 = cv2.inRange(hsv, self.ball_hsv_low2, self.ball_hsv_high2)
        ball_mask = cv2.dilate(cv2.erode(mask1 | mask2, None, iterations=2), None, iterations=2)

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((cx, cy), radius) = cv2.minEnclosingCircle(largest)
            if radius >= 3:
                ball = (float(cx), float(cy), float(radius))

        # Tape detection (only if ball found — saves time)
        tape = None
        if ball is not None:
            tape_mask = cv2.dilate(cv2.erode(
                cv2.inRange(hsv, self.tape_hsv_low, self.tape_hsv_high),
                None, iterations=1), None, iterations=2)
            tape_points = np.column_stack(np.where(tape_mask > 0))
            if len(tape_points) > 0:
                ball_cx, ball_cy = ball[0], ball[1]
                dists = np.sqrt((tape_points[:, 1] - ball_cx) ** 2 +
                                (tape_points[:, 0] - ball_cy) ** 2)
                nearest = tape_points[np.argmin(dists)]
                tape = (float(nearest[1]), float(nearest[0]))

        return ball, tape, hsv

    def observe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract [ball_dist, ball_angle, gap_dist, gap_angle]. Returns None if no ball."""
        frame = self._crop_center(frame)
        ball, tape, _ = self._process(frame)

        if ball is None:
            return None

        ball_cx, ball_cy, ball_radius_px = ball
        ball_dist = (self.ball_real_diameter * self.focal_length) / max(ball_radius_px * 2, 1)
        ball_angle = math.atan2(self.cx - ball_cx, self.focal_length)

        if tape is not None:
            tape_cx, tape_cy = tape
            gap_px = math.sqrt((ball_cx - tape_cx) ** 2 + (ball_cy - tape_cy) ** 2)
            gap_dist = gap_px / self.frame_diag
            gap_angle = math.atan2(tape_cx - ball_cx, self.focal_length)
        else:
            gap_dist = 1.0
            gap_angle = 0.0

        return np.array([ball_dist, ball_angle, gap_dist, gap_angle], dtype=np.float32)

    def observe_and_annotate(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Run detection once, return (obs, annotated_frame) together."""
        frame = self._crop_center(frame)
        ball, tape, _ = self._process(frame)
        vis = frame.copy()

        obs = None
        if ball is not None:
            ball_cx, ball_cy, ball_radius_px = ball
            ball_dist = (self.ball_real_diameter * self.focal_length) / max(ball_radius_px * 2, 1)
            ball_angle = math.atan2(self.cx - ball_cx, self.focal_length)

            if tape is not None:
                tape_cx, tape_cy = tape
                gap_px = math.sqrt((ball_cx - tape_cx) ** 2 + (ball_cy - tape_cy) ** 2)
                gap_dist = gap_px / self.frame_diag
                gap_angle = math.atan2(tape_cx - ball_cx, self.focal_length)
            else:
                gap_dist = 1.0
                gap_angle = 0.0

            obs = np.array([ball_dist, ball_angle, gap_dist, gap_angle], dtype=np.float32)

            cv2.circle(vis, (int(ball_cx), int(ball_cy)), int(ball_radius_px), (0, 255, 0), 2)
            cv2.putText(vis, "BALL", (int(ball_cx - 20), int(ball_cy - ball_radius_px - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if tape is not None:
                tx, ty = tape
                cv2.circle(vis, (int(tx), int(ty)), 5, (255, 0, 0), -1)
                cv2.line(vis, (int(ball_cx), int(ball_cy)), (int(tx), int(ty)), (255, 255, 0), 2)

        return obs, vis

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw detections on the frame for debugging. Returns annotated copy."""
        _, vis = self.observe_and_annotate(frame)
        return vis
