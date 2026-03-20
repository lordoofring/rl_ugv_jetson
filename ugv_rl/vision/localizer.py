"""
ArUco marker-based vision localizer.

Detects ArUco markers via the robot's camera and computes the robot's
grid position + heading.  Each marker has a unique ID that maps to a
known grid coordinate (loaded from marker_map.json).

Coordinate convention
---------------------
The marker map stores the grid-cell (gx, gy) of each marker.
The camera extrinsics (height, pitch, roll) are set in config.yaml.
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VisionLocalizer:
    """Estimate the robot's grid pose from ArUco markers visible to the camera."""

    def __init__(
        self,
        camera_index: int = 0,
        marker_map_path: str = "marker_map.json",
        cell_size: float = 0.5,
        marker_size: float = 0.05,
        aruco_dict_name: str = "DICT_4X4_50",
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        """
        Args:
            camera_index:    /dev/videoN index or GStreamer pipeline string.
            marker_map_path: JSON file mapping marker IDs -> grid coords.
            cell_size:       Physical size of one grid cell in meters.
            marker_size:     Physical side-length of printed markers in meters.
            aruco_dict_name: OpenCV ArUco dictionary name (e.g. DICT_4X4_50).
            camera_matrix:   3x3 intrinsic matrix.  If None a rough default is used.
            dist_coeffs:     Distortion coefficients.  If None zeros are used.
        """
        self.cell_size = cell_size
        self.marker_size = marker_size

        # --- ArUco setup (supports both OpenCV 4.7+ and older versions) ---
        dict_id = getattr(cv2.aruco, aruco_dict_name, None)
        if dict_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {aruco_dict_name}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        # New API (OpenCV 4.7+): ArucoDetector class
        # Old API (OpenCV <4.7): DetectorParameters_create() function
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._use_new_api = True
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None
            self._use_new_api = False

        # --- Camera ---
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.warning("Camera %s could not be opened – vision disabled.", camera_index)
            self._camera_ok = False
        else:
            self._camera_ok = True

        # Intrinsics (will be replaced if you calibrate)
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            # Rough default for a 640x480 camera
            self.camera_matrix = np.array([
                [500.0,   0.0, 320.0],
                [  0.0, 500.0, 240.0],
                [  0.0,   0.0,   1.0],
            ])
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        # --- Marker map: {marker_id: (gx, gy)} ---
        self.marker_map: Dict[int, Tuple[int, int]] = {}
        self._load_marker_map(marker_map_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(self) -> Optional[Tuple[int, int, float]]:
        """Capture a frame, detect markers, and return the estimated grid pose.

        Returns:
            (grid_x, grid_y, theta) if at least one known marker is visible,
            None otherwise (caller should fall back to dead reckoning).
        """
        if not self._camera_ok:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        return self.localize_from_frame(frame)

    def localize_from_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Run localization on an already-captured frame (useful for testing)."""
        detections = self._detect_markers(frame)
        if not detections:
            return None

        # Filter to only markers that exist in our map
        known = [(mid, tvec, rvec) for mid, tvec, rvec in detections if mid in self.marker_map]
        if not known:
            return None

        # If multiple markers visible, average the position estimates
        gx_estimates: List[float] = []
        gy_estimates: List[float] = []
        theta_estimates: List[float] = []

        for marker_id, tvec, rvec in known:
            mgx, mgy = self.marker_map[marker_id]
            # tvec is the marker position in the camera frame (x_right, y_down, z_forward)
            # The marker's world position is (mgx * cell_size, mgy * cell_size)
            # The robot is offset from the marker by -tvec (rotated into world frame)
            R, _ = cv2.Rodrigues(rvec)
            # Marker position in camera coords
            cam_to_marker = tvec.flatten()
            # Robot heading from rotation matrix (yaw around world Z)
            # R transforms from marker frame to camera frame
            # We want the camera's yaw in the world/marker frame
            # The camera's forward is +Z in camera coords, which in world is R^T @ [0,0,1]
            forward_world = R.T @ np.array([0.0, 0.0, 1.0])
            theta = math.atan2(forward_world[0], forward_world[2])

            # Robot world position: marker_world_pos - R^T @ tvec (projected to ground plane)
            offset_world = R.T @ cam_to_marker
            robot_x = mgx * self.cell_size - offset_world[0]
            robot_y = mgy * self.cell_size - offset_world[2]  # Z forward -> world Y

            est_gx = robot_x / self.cell_size
            est_gy = robot_y / self.cell_size

            gx_estimates.append(est_gx)
            gy_estimates.append(est_gy)
            theta_estimates.append(theta)

        avg_gx = int(round(np.mean(gx_estimates)))
        avg_gy = int(round(np.mean(gy_estimates)))
        avg_theta = float(np.mean(theta_estimates))

        return avg_gx, avg_gy, avg_theta

    def release(self):
        """Release the camera."""
        if self._camera_ok:
            self.cap.release()
            self._camera_ok = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_marker_map(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.warning("Marker map %s not found – vision localizer has no markers.", path)
            return
        with open(p) as f:
            raw = json.load(f)
        # Accept both {"0": [gx, gy], ...} and {"markers": [{"id":0, "gx":0, "gy":0}, ...]}
        if "markers" in raw:
            for entry in raw["markers"]:
                self.marker_map[int(entry["id"])] = (int(entry["gx"]), int(entry["gy"]))
        else:
            for k, v in raw.items():
                self.marker_map[int(k)] = (int(v[0]), int(v[1]))
        logger.info("Loaded %d markers from %s", len(self.marker_map), path)

    def _detect_markers(self, frame: np.ndarray):
        """Return list of (marker_id, tvec, rvec) for all detected markers."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return []

        # Estimate pose for each marker
        results = []
        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i : i + 1], self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            results.append((int(marker_id), tvec[0][0], rvec[0][0]))
        return results
