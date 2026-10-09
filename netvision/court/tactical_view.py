"""Map broadcast pixels to real court meters with a homography.

Steps:

1. Validate the detected court keypoints. A keypoint is dropped when its
   distance ratios to other keypoints disagree with the real court (the
   tutorial's proportional check), and with five or more points a RANSAC fit
   removes any remaining outliers.
2. Fit a homography from the remaining image points to court meters.
3. Project each player's foot point onto the court. Positions far outside
   the court (fans, bench) are dropped.
"""
from __future__ import annotations

import itertools
import logging
from typing import Optional, Sequence

import cv2
import numpy as np

from ..utils.bbox import get_foot_position
from .geometry import CourtGeometry

logger = logging.getLogger(__name__)

CourtPositions = dict[int, tuple[float, float]]


class TacticalViewConverter:
    def __init__(
        self,
        geometry: Optional[CourtGeometry] = None,
        tolerance: float = 0.8,
        min_keypoints: int = 4,
        carry_frames: int = 30,
        off_court_margin: float = 1.5,
        ransac_threshold_m: float = 0.75,
        max_reprojection_error_m: float = 1.0,
    ):
        self.geometry = geometry or CourtGeometry()
        self.court_keypoints = self.geometry.keypoints
        self.tolerance = tolerance
        self.min_keypoints = max(4, min_keypoints)
        self.carry_frames = carry_frames
        self.off_court_margin = off_court_margin
        self.ransac_threshold_m = ransac_threshold_m
        self.max_reprojection_error_m = max_reprojection_error_m

    # ------------------------------------------------------------------ validation
    def validate_keypoints(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return a copy with keypoints that fail the proportion check set to NaN."""
        if keypoints is None:
            return None
        points = np.array(keypoints, dtype=float, copy=True)
        detected = [i for i in range(len(points)) if np.all(np.isfinite(points[i]))]
        if len(detected) < 3:
            return points
        invalid = []
        for i in detected:
            others = [j for j in detected if j != i]
            errors = []
            for j, k in itertools.islice(itertools.combinations(others, 2), 15):
                image_ratio = _safe_ratio(points[i], points[j], points[k])
                court_ratio = _safe_ratio(self.court_keypoints[i], self.court_keypoints[j], self.court_keypoints[k])
                if image_ratio is None or court_ratio is None:
                    continue
                errors.append(abs(image_ratio - court_ratio) / court_ratio)
            if errors and float(np.median(errors)) > self.tolerance:
                invalid.append(i)
        for i in invalid:
            points[i] = np.nan
        return points

    # ------------------------------------------------------------------ homography
    def compute_homography(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Homography from image pixels to court meters, or None when not enough points."""
        if keypoints is None:
            return None
        valid = [i for i in range(len(keypoints)) if np.all(np.isfinite(keypoints[i]))]
        if len(valid) < self.min_keypoints:
            return None
        source = np.array([keypoints[i] for i in valid], dtype=np.float32)
        target = np.array([self.court_keypoints[i] for i in valid], dtype=np.float32)
        try:
            if len(valid) >= 5:
                homography, _ = cv2.findHomography(source, target, cv2.RANSAC, self.ransac_threshold_m)
            else:
                homography, _ = cv2.findHomography(source, target, 0)
        except cv2.error as error:
            logger.debug("findHomography failed: %s", error)
            return None
        if homography is None or not np.all(np.isfinite(homography)):
            return None
        projected = project_points(homography, source)
        errors = np.linalg.norm(projected - target, axis=1)
        if float(np.median(errors)) > self.max_reprojection_error_m:
            return None
        return homography

    def build_homographies(self, keypoints_per_frame: Sequence[Optional[np.ndarray]]) -> list[Optional[np.ndarray]]:
        """One homography per frame. The last good one is reused for up to ``carry_frames`` frames."""
        output: list[Optional[np.ndarray]] = []
        last_good: Optional[np.ndarray] = None
        last_good_frame = -10**9
        for frame, keypoints in enumerate(keypoints_per_frame):
            homography = self.compute_homography(self.validate_keypoints(keypoints))
            if homography is not None:
                last_good, last_good_frame = homography, frame
            elif last_good is not None and frame - last_good_frame <= self.carry_frames:
                homography = last_good
            output.append(homography)
        return output

    # ------------------------------------------------------------------ projection
    def players_to_court(
        self, player_tracks: Sequence[dict[int, Sequence[float]]], homographies: Sequence[Optional[np.ndarray]]
    ) -> list[CourtPositions]:
        output: list[CourtPositions] = []
        for players, homography in zip(player_tracks, homographies):
            frame_positions: CourtPositions = {}
            if homography is not None and players:
                ids = list(players.keys())
                feet = np.array([get_foot_position(players[i]) for i in ids], dtype=np.float32)
                court = project_points(homography, feet)
                for track_id, (x, y) in zip(ids, court):
                    if np.isfinite(x) and np.isfinite(y) and self.geometry.contains((x, y), self.off_court_margin):
                        frame_positions[track_id] = (float(x), float(y))
            output.append(frame_positions)
        return output


def project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)


def _safe_ratio(a, b, c) -> Optional[float]:
    d_ab = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    d_ac = float(np.linalg.norm(np.asarray(a) - np.asarray(c)))
    if d_ac < 1e-6 or d_ab < 1e-6:
        return None
    return d_ab / d_ac
