"""Ball (and rim) selection plus ball track cleaning."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ..detection import Detection, matches
from ..utils.bbox import get_center, measure_distance

BallBox = Optional[list[float]]


class SingleObjectSelector:
    """Keeps the single most confident detection of a class in each frame."""

    def __init__(self, class_names: Iterable[str], min_confidence: float):
        self.class_names = tuple(class_names)
        self.min_confidence = min_confidence

    def select(self, detections: Sequence[Detection]) -> BallBox:
        best: Optional[Detection] = None
        for detection in detections:
            if not matches(detection.class_name, self.class_names):
                continue
            if detection.confidence < self.min_confidence:
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection
        return list(best.bbox) if best else None


class BallTracker(SingleObjectSelector):
    def __init__(self, class_names: Iterable[str] = ("ball", "basketball", "sports ball"), min_confidence: float = 0.15):
        super().__init__(class_names, min_confidence)


class RimDetector(SingleObjectSelector):
    def __init__(self, class_names: Iterable[str] = ("rim", "hoop"), min_confidence: float = 0.3):
        super().__init__(class_names, min_confidence)


def remove_wrong_detections(positions: Sequence[BallBox], max_px_per_frame: float = 25.0) -> list[BallBox]:
    """Drop detections that jump further than the ball could travel.

    The allowed distance grows with the number of frames since the last
    trusted detection (tutorial rule: 25 px multiplied by the frame gap).
    """
    cleaned: list[BallBox] = [list(p) if p is not None else None for p in positions]
    last_good_index = -1
    for index, box in enumerate(cleaned):
        if box is None:
            continue
        if last_good_index == -1:
            last_good_index = index
            continue
        gap = index - last_good_index
        allowed = max_px_per_frame * gap
        distance = measure_distance(get_center(cleaned[last_good_index]), get_center(box))
        if distance > allowed:
            cleaned[index] = None
        else:
            last_good_index = index
    return cleaned


def interpolate_positions(positions: Sequence[BallBox], max_gap: int = 30) -> list[BallBox]:
    """Linearly fill gaps of at most ``max_gap`` missing frames between two detections.

    Gaps before the first or after the last detection are left empty because
    there is nothing to anchor them to.
    """
    result: list[BallBox] = [list(p) if p is not None else None for p in positions]
    known = [i for i, p in enumerate(result) if p is not None]
    for start, end in zip(known, known[1:]):
        gap = end - start - 1
        if gap <= 0 or gap > max_gap:
            continue
        a = np.array(result[start], dtype=float)
        b = np.array(result[end], dtype=float)
        for offset in range(1, gap + 1):
            t = offset / (gap + 1)
            result[start + offset] = list(a + (b - a) * t)
    return result
