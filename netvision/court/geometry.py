"""Basketball court geometry in meters.

Coordinates: x runs along the court length from the left baseline (0) to the
right baseline (``length``). y runs across the court from the far sideline
(0) to the near sideline (``width``).

The 18 keypoints follow the ordering used by the court keypoint dataset from
the tutorial:

====  ==========================================
 0-5  left baseline: corner, corner three, lane, lane, corner three, corner
 6-7  half court line: near sideline, far sideline
 8-9  left free throw line: lane top, lane bottom
10-15 right baseline (near corner to far corner)
16-17 right free throw line: lane top, lane bottom
====  ==========================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NUM_KEYPOINTS = 18


@dataclass(frozen=True)
class CourtGeometry:
    length: float = 28.0
    width: float = 15.0
    corner_three_offset: float = 0.9
    lane_width: float = 4.9
    free_throw_distance: float = 5.8
    rim_from_baseline: float = 1.575
    three_point_radius: float = 6.75
    center_circle_radius: float = 1.8
    backboard_from_baseline: float = 1.2

    @property
    def lane_top(self) -> float:
        return self.width / 2 - self.lane_width / 2

    @property
    def lane_bottom(self) -> float:
        return self.width / 2 + self.lane_width / 2

    @property
    def rims(self) -> np.ndarray:
        """Rim centers (left, right) in meters."""
        mid = self.width / 2
        return np.array([[self.rim_from_baseline, mid], [self.length - self.rim_from_baseline, mid]])

    @property
    def keypoints(self) -> np.ndarray:
        L, W, c3 = self.length, self.width, self.corner_three_offset
        top, bottom, ft = self.lane_top, self.lane_bottom, self.free_throw_distance
        return np.array(
            [
                (0, 0), (0, c3), (0, top), (0, bottom), (0, W - c3), (0, W),
                (L / 2, W), (L / 2, 0),
                (ft, top), (ft, bottom),
                (L, W), (L, W - c3), (L, bottom), (L, top), (L, c3), (L, 0),
                (L - ft, top), (L - ft, bottom),
            ],
            dtype=float,
        )

    def contains(self, point, margin: float = 0.0) -> bool:
        x, y = point
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin

    def distance_to_nearest_rim(self, point) -> float:
        return float(np.min(np.linalg.norm(self.rims - np.asarray(point, dtype=float), axis=1)))

    def line_polylines(self, arc_steps: int = 48) -> list[np.ndarray]:
        """Court markings as a list of polylines in meters (for drawing)."""
        L, W = self.length, self.width
        lines: list[np.ndarray] = []
        lines.append(np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)], dtype=float))
        lines.append(np.array([(L / 2, 0), (L / 2, W)], dtype=float))
        lines.append(_circle((L / 2, W / 2), self.center_circle_radius, arc_steps))
        for side in (0, 1):
            sign = 1 if side == 0 else -1
            base = 0.0 if side == 0 else L
            ft_x = base + sign * self.free_throw_distance
            # lane
            lines.append(np.array(
                [(base, self.lane_top), (ft_x, self.lane_top), (ft_x, self.lane_bottom), (base, self.lane_bottom)],
                dtype=float,
            ))
            lines.append(_circle((ft_x, W / 2), self.center_circle_radius, arc_steps))
            # three point line: straight corner segments joined by an arc
            rim_x = base + sign * self.rim_from_baseline
            c3 = self.corner_three_offset
            dy = W / 2 - c3
            dx = math.sqrt(max(self.three_point_radius ** 2 - dy ** 2, 0.0))
            start_angle = math.atan2(-dy, dx)
            angles = np.linspace(start_angle, -start_angle, arc_steps)
            arc = np.stack([rim_x + sign * self.three_point_radius * np.cos(angles),
                            W / 2 + self.three_point_radius * np.sin(angles)], axis=1)
            corner_top = np.array([(base, c3), (rim_x + sign * dx, c3)])
            corner_bottom = np.array([(rim_x + sign * dx, W - c3), (base, W - c3)])
            lines.append(np.vstack([corner_top, arc, corner_bottom]))
            # backboard and rim
            bb_x = base + sign * self.backboard_from_baseline
            lines.append(np.array([(bb_x, W / 2 - 0.9), (bb_x, W / 2 + 0.9)], dtype=float))
            lines.append(_circle((rim_x, W / 2), 0.225, 16))
        return lines


def _circle(center, radius: float, steps: int) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, steps)
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=1)
