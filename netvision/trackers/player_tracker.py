"""Multi-object tracking of players with ByteTrack."""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import supervision as sv

from ..detection import Detection, matches

PlayerFrame = dict[int, list[float]]


def _make_bytetrack(frame_rate: float, lost_track_buffer: int, activation: float, matching: float):
    """Prefer the maintained ``trackers`` package, fall back to supervision."""
    try:
        from trackers import ByteTrackTracker

        return ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=activation,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=matching,
        )
    except ImportError:  # pragma: no cover - depends on installed packages
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return sv.ByteTrack(
                track_activation_threshold=activation,
                lost_track_buffer=lost_track_buffer,
                frame_rate=int(round(frame_rate)),
            )


class PlayerTracker:
    """Turns per-frame player detections into tracks with stable IDs."""

    def __init__(
        self,
        class_names: Iterable[str] = ("player", "person"),
        min_confidence: float = 0.5,
        frame_rate: float = 30.0,
        lost_track_buffer: int = 60,
        track_activation_threshold: float = 0.5,
        minimum_matching_threshold: float = 0.1,
        tracker: str = "bytetrack",
    ):
        if tracker != "bytetrack":
            raise ValueError(f"Unsupported tracker '{tracker}'. Only 'bytetrack' is available in this release.")
        self.class_names = tuple(class_names)
        self.min_confidence = min_confidence
        self._tracker = _make_bytetrack(frame_rate, lost_track_buffer, track_activation_threshold, minimum_matching_threshold)

    def _to_supervision(self, detections: Sequence[Detection]) -> sv.Detections:
        players = [d for d in detections if matches(d.class_name, self.class_names) and d.confidence >= self.min_confidence]
        if not players:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([d.bbox for d in players], dtype=float),
            confidence=np.array([d.confidence for d in players], dtype=float),
            class_id=np.zeros(len(players), dtype=int),
        )

    def update(self, detections: Sequence[Detection]) -> PlayerFrame:
        """Feed one frame of detections and get ``{track_id: [x1, y1, x2, y2]}`` back."""
        sv_detections = self._to_supervision(detections)
        if hasattr(self._tracker, "update_with_detections"):
            tracked = self._tracker.update_with_detections(sv_detections)
        else:
            tracked = self._tracker.update(sv_detections)
        frame: PlayerFrame = {}
        if tracked.tracker_id is None:
            return frame
        for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
            if track_id is None or int(track_id) < 0:
                continue
            frame[int(track_id)] = [float(v) for v in box]
        return frame
