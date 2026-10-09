"""Bounding box and point geometry helpers.

Boxes are always ``(x1, y1, x2, y2)`` in pixels with ``x1 <= x2`` and ``y1 <= y2``.
"""
from __future__ import annotations

import math
from typing import Sequence

Box = Sequence[float]
Point = Sequence[float]


def get_center(bbox: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_width(bbox: Box) -> float:
    return float(bbox[2] - bbox[0])


def get_height(bbox: Box) -> float:
    return float(bbox[3] - bbox[1])


def get_foot_position(bbox: Box) -> tuple[float, float]:
    """Bottom center of a player box, which is where the player touches the floor."""
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def measure_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def measure_xy_distance(p1: Point, p2: Point) -> tuple[float, float]:
    return p1[0] - p2[0], p1[1] - p2[1]


def box_area(bbox: Box) -> float:
    return max(0.0, get_width(bbox)) * max(0.0, get_height(bbox))


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def containment_ratio(inner: Box, outer: Box) -> float:
    """Fraction of ``inner`` that lies inside ``outer`` (0.0 to 1.0)."""
    area = box_area(inner)
    if area <= 0:
        return 0.0
    return intersection_area(inner, outer) / area


def clip_box(bbox: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip a box to the image and convert it to integer pixel indices."""
    x1 = int(max(0, min(width, math.floor(bbox[0]))))
    y1 = int(max(0, min(height, math.floor(bbox[1]))))
    x2 = int(max(0, min(width, math.ceil(bbox[2]))))
    y2 = int(max(0, min(height, math.ceil(bbox[3]))))
    return x1, y1, x2, y2
