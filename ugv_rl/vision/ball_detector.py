"""
Red ball detector using HSV color thresholding.

Returns the ball's position relative to the camera frame:
  - angle (left/right offset from center)
  - estimated distance (from apparent size)
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


class BallDetector:
    """Detect a red ball in a camera frame and estimate relative position."""

    def __init__(
        self,
        camera_index=0,
        ball_real_diameter: float = 0.10,
        focal_length: float = 500.0,
        frame_width: int = 640,
        frame_height: int = 480,
        hsv_low1=(0, 120, 70),
        hsv_high1=(10, 255, 255),
        hsv_low2=(170, 120, 70),
        hsv_high2=(180, 255, 255),
    ):
        """
        Args:
            camera_index:      Camera device index or pipeline.
            ball_real_diameter: Actual ball diameter in meters.
            focal_length:      Approximate camera focal length in pixels.
            frame_width/height: Expected frame dimensions.
            hsv_low1/high1:    Lower red HSV range.
            hsv_low2/high2:    Upper red HSV range (red wraps around in HSV).
        """
        self.ball_real_diameter = ball_real_diameter
        self.focal_length = focal_length
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0

        self.hsv_low1 = np.array(hsv_low1)
        self.hsv_high1 = np.array(hsv_high1)
        self.hsv_low2 = np.array(hsv_low2)
        self.hsv_high2 = np.array(hsv_high2)

        self.cap = None
        self._camera_ok = False
        if camera_index is not None:
            from ugv_rl.vision.localizer import VisionLocalizer
            self.cap, self._camera_ok = VisionLocalizer._open_camera(camera_index)

    def detect(self) -> Optional[Tuple[float, float]]:
        """Capture a frame and detect the ball.

        Returns:
            (distance, angle) in meters and radians, or None if not found.
            angle is negative=left, positive=right relative to camera center.
        """
        if not self._camera_ok:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        return self.detect_from_frame(frame)

    def detect_from_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect the ball in a provided frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around in HSV, so combine two ranges
        mask1 = cv2.inRange(hsv, self.hsv_low1, self.hsv_high1)
        mask2 = cv2.inRange(hsv, self.hsv_low2, self.hsv_high2)
        mask = mask1 | mask2

        # Clean up noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour = the ball
        largest = max(contours, key=cv2.contourArea)
        ((cx, cy), radius_px) = cv2.minEnclosingCircle(largest)

        if radius_px < 5:
            return None

        # Estimate distance from apparent size
        diameter_px = radius_px * 2
        distance = (self.ball_real_diameter * self.focal_length) / diameter_px

        # Angle from center of frame
        offset_px = cx - self.cx
        angle = math.atan2(offset_px, self.focal_length)

        return distance, angle

    def release(self):
        if self._camera_ok and self.cap is not None:
            self.cap.release()
            self._camera_ok = False
