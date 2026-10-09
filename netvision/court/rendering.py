"""Draw a top-down court image from the geometry (no image asset needed)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import CourtGeometry


@dataclass(frozen=True)
class CourtCanvas:
    geometry: CourtGeometry
    px_per_m: float
    margin_px: int = 10

    @property
    def size(self) -> tuple[int, int]:
        width = int(round(self.geometry.length * self.px_per_m)) + 2 * self.margin_px
        height = int(round(self.geometry.width * self.px_per_m)) + 2 * self.margin_px
        return width, height

    def to_pixels(self, points_m) -> np.ndarray:
        points = np.asarray(points_m, dtype=float).reshape(-1, 2)
        return points * self.px_per_m + self.margin_px


def render_court(
    canvas: CourtCanvas,
    floor_color=(110, 170, 215),
    line_color=(255, 255, 255),
    outside_color=(60, 60, 60),
    line_thickness: int = 2,
) -> np.ndarray:
    width, height = canvas.size
    image = np.full((height, width, 3), outside_color, dtype=np.uint8)
    top_left = canvas.to_pixels([(0, 0)])[0]
    bottom_right = canvas.to_pixels([(canvas.geometry.length, canvas.geometry.width)])[0]
    cv2.rectangle(image, tuple(int(v) for v in top_left), tuple(int(v) for v in bottom_right), floor_color, -1)
    for polyline in canvas.geometry.line_polylines():
        pixels = np.round(canvas.to_pixels(polyline)).astype(np.int32)
        cv2.polylines(image, [pixels], False, line_color, line_thickness, cv2.LINE_AA)
    return image
