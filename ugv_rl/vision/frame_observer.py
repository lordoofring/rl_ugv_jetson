"""
Extract observations from a camera frame for the Ball Push task.

Detects:
  - Red ball (HSV thresholding + optional circularity filter) → ball_dist, ball_angle
  - Blue tape (HSV thresholding) → gap_dist, gap_angle

Multi-frame averaging
---------------------
When n_frames > 1, `observe_and_annotate_multi(frames)` averages ball and tape
pixel positions across all frames where detection succeeds, then derives the
observation values from the averaged geometry.  Averaging in pixel space before
computing atan2 / distance is more numerically correct than averaging the final
values (especially angles).
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


class FrameObserver:
    """Process BGR camera frame(s) into the 4-dim observation vector."""

    def __init__(
        self,
        ball_real_diameter: float = 0.14,
        focal_length: float = 107.0,
        frame_width: int = 320,
        frame_height: int = 240,
        ball_hsv_low1=(25, 100, 80),
        ball_hsv_high1=(10, 255, 255),
        ball_hsv_low2=(165, 100, 80),
        ball_hsv_high2=(180, 255, 255),
        tape_hsv_low=(90, 80, 50),
        tape_hsv_high=(130, 255, 255),
        require_roundness: bool = True,
        min_circularity: float = 0.75,
        n_frames: int = 3,
    ):
        """
        Args:
            require_roundness: Reject red contours with circularity < min_circularity.
                               Filters out non-circular red patches (floor markings, etc.).
            min_circularity:   Threshold for 4π·Area/Perimeter² in [0, 1].
            n_frames:          Number of frames to average per observation call.
                               observe() / observe_and_annotate() use a single frame;
                               observe_multi() / observe_and_annotate_multi() use n_frames.
        """
        self.ball_real_diameter = ball_real_diameter
        self.focal_length = focal_length
        self.frame_width = frame_width
        self.frame_height = frame_height

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

        self.require_roundness = require_roundness
        self.min_circularity = min_circularity
        self.n_frames = n_frames

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crop_center(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        margin = w // 4
        return frame[:, margin:w - margin]

    @staticmethod
    def _circularity(contour) -> float:
        """4π·Area / Perimeter².  1.0 = perfect circle."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            return 0.0
        return (4 * math.pi * area) / (perimeter ** 2)

    # ------------------------------------------------------------------
    # Single-frame detection
    # ------------------------------------------------------------------

    def _process(self, frame: np.ndarray):
        """Detect ball and tape in one frame. Returns (ball, tape, hsv).

        ball = (cx, cy, radius_px) or None
        tape = (cx, cy) or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.ball_hsv_low1, self.ball_hsv_high1)
        mask2 = cv2.inRange(hsv, self.ball_hsv_low2, self.ball_hsv_high2)
        ball_mask = cv2.dilate(cv2.erode(mask1 | mask2, None, iterations=2), None, iterations=2)

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball = None
        if contours:
            if self.require_roundness:
                candidates = [c for c in contours if self._circularity(c) >= self.min_circularity]
            else:
                candidates = contours

            if candidates:
                largest = max(candidates, key=cv2.contourArea)
                ((cx, cy), radius) = cv2.minEnclosingCircle(largest)
                if radius >= 3:
                    ball = (float(cx), float(cy), float(radius))

        tape = None
        if ball is not None:
            tape_mask = cv2.dilate(cv2.erode(
                cv2.inRange(hsv, self.tape_hsv_low, self.tape_hsv_high),
                None, iterations=1), None, iterations=2)
            tape_points = np.column_stack(np.where(tape_mask > 0))
            if len(tape_points) > 0:
                dists = np.sqrt((tape_points[:, 1] - ball[0]) ** 2 +
                                (tape_points[:, 0] - ball[1]) ** 2)
                nearest = tape_points[np.argmin(dists)]
                tape = (float(nearest[1]), float(nearest[0]))

        return ball, tape, hsv

    # ------------------------------------------------------------------
    # Multi-frame averaging
    # ------------------------------------------------------------------

    def _process_multi(self, frames: List[np.ndarray]):
        """Run _process on each frame and return averaged ball/tape positions.

        Averages in pixel space so that downstream atan2/distance are computed
        from the averaged geometry rather than averaging angles directly.

        Returns (ball, tape) with same types as _process(), or (None, None).
        """
        balls, tapes = [], []
        last_frame = frames[-1]

        for frame in frames:
            ball, tape, _ = self._process(frame)
            if ball is not None:
                balls.append(ball)
                if tape is not None:
                    tapes.append(tape)

        if not balls:
            return None, None

        avg_ball = (
            float(np.mean([b[0] for b in balls])),
            float(np.mean([b[1] for b in balls])),
            float(np.mean([b[2] for b in balls])),
        )
        avg_tape = (
            float(np.mean([t[0] for t in tapes])),
            float(np.mean([t[1] for t in tapes])),
        ) if tapes else None

        return avg_ball, avg_tape

    # ------------------------------------------------------------------
    # Observation builders (shared logic)
    # ------------------------------------------------------------------

    def _obs_from_detection(self, ball, tape) -> np.ndarray:
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

    def _annotate(self, vis: np.ndarray, ball, tape) -> np.ndarray:
        if ball is not None:
            cx, cy, r = int(ball[0]), int(ball[1]), int(ball[2])
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.putText(vis, "BALL", (cx - 20, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if tape is not None:
                tx, ty = int(tape[0]), int(tape[1])
                cv2.circle(vis, (tx, ty), 5, (255, 0, 0), -1)
                cv2.line(vis, (cx, cy), (tx, ty), (255, 255, 0), 2)
        return vis

    # ------------------------------------------------------------------
    # Public API — single frame
    # ------------------------------------------------------------------

    def observe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """[ball_dist, ball_angle, gap_dist, gap_angle] from one frame, or None."""
        frame = self._crop_center(frame)
        ball, tape, _ = self._process(frame)
        if ball is None:
            return None
        return self._obs_from_detection(ball, tape)

    def observe_and_annotate(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Single-frame detection with annotated frame for display."""
        frame = self._crop_center(frame)
        ball, tape, _ = self._process(frame)
        vis = self._annotate(frame.copy(), ball, tape)
        obs = self._obs_from_detection(ball, tape) if ball is not None else None
        return obs, vis

    # ------------------------------------------------------------------
    # Public API — multi-frame (preferred for real robot)
    # ------------------------------------------------------------------

    def observe_multi(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Average detection across multiple frames, return obs or None."""
        cropped = [self._crop_center(f) for f in frames]
        ball, tape = self._process_multi(cropped)
        if ball is None:
            return None
        return self._obs_from_detection(ball, tape)

    def observe_and_annotate_multi(
        self, frames: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Average detection across frames. Annotates the last frame for display."""
        cropped = [self._crop_center(f) for f in frames]
        ball, tape = self._process_multi(cropped)
        vis = self._annotate(cropped[-1].copy(), ball, tape)
        obs = self._obs_from_detection(ball, tape) if ball is not None else None
        return obs, vis

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        _, vis = self.observe_and_annotate(frame)
        return vis
