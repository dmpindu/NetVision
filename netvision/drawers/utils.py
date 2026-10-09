"""Small OpenCV drawing helpers shared by the drawers."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..utils.bbox import get_center, get_width

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_ellipse(frame: np.ndarray, bbox: Sequence[float], color, track_id=None) -> np.ndarray:
    """Ellipse under the player's feet with an optional ID label."""
    y2 = int(bbox[3])
    x_center, _ = get_center(bbox)
    width = get_width(bbox)
    cv2.ellipse(frame, (int(x_center), y2), (int(width), int(0.35 * width)), 0.0, -45, 235, color, 2, cv2.LINE_4)
    if track_id is not None:
        rect_w, rect_h = 40, 20
        x1 = int(x_center - rect_w / 2)
        y1 = int(y2 - rect_h / 2) + 15
        cv2.rectangle(frame, (x1, y1), (x1 + rect_w, y1 + rect_h), color, cv2.FILLED)
        text_x = x1 + (12 if int(track_id) < 10 else 6 if int(track_id) < 100 else 2)
        cv2.putText(frame, str(track_id), (text_x, y1 + 15), FONT, 0.5, _text_color(color), 2)
    return frame


def draw_triangle(frame: np.ndarray, bbox: Sequence[float], color) -> np.ndarray:
    """Small triangle pointing at the top of a box."""
    y = int(bbox[1])
    x, _ = get_center(bbox)
    points = np.array([[int(x), y], [int(x) - 10, y - 20], [int(x) + 10, y - 20]], dtype=np.int32)
    cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
    return frame


def draw_panel(frame: np.ndarray, top_left, bottom_right, alpha: float = 0.7, color=(255, 255, 255)) -> np.ndarray:
    """Semi-transparent rectangle used as a background for text."""
    x1, y1 = (int(v) for v in top_left)
    x2, y2 = (int(v) for v in bottom_right)
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    region = frame[y1:y2, x1:x2]
    overlay = np.full_like(region, color)
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)
    return frame


def put_text(frame: np.ndarray, text: str, origin, scale: float = 0.6, color=(0, 0, 0), thickness: int = 2) -> None:
    cv2.putText(frame, text, (int(origin[0]), int(origin[1])), FONT, scale, color, thickness, cv2.LINE_AA)


def _text_color(background) -> tuple[int, int, int]:
    b, g, r = background
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)
