"""Data containers passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .events import Event
from .metrics.shots import Shot
from .utils.video import VideoInfo


@dataclass
class RawDetections:
    """Everything pass 1 collects from the models (this is what gets cached)."""

    player_tracks: list[dict[int, list[float]]]
    ball_positions: list[Optional[list[float]]]
    rim_positions: Optional[list[Optional[list[float]]]]
    keypoints: Optional[list[Optional[np.ndarray]]]
    team_crops: dict[int, list[np.ndarray]]

    def __len__(self) -> int:
        return len(self.player_tracks)


@dataclass
class AnalysisResult:
    video: VideoInfo
    player_tracks: list[dict[int, list[float]]]
    ball_positions: list[Optional[list[float]]]
    raw_ball_positions: list[Optional[list[float]]]
    rim_positions: Optional[list[Optional[list[float]]]]
    keypoints: Optional[list[Optional[np.ndarray]]]
    team_assignment: dict[int, int]
    possession: list[int]
    team_possession: list[int]
    events: list[Event]
    shots: list[Shot]
    homographies: Optional[list[Optional[np.ndarray]]] = None
    court_positions: Optional[list[dict[int, tuple[float, float]]]] = None
    distances: Optional[list[dict[int, float]]] = None
    speeds: Optional[list[dict[int, float]]] = None
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.player_tracks)

    @property
    def has_court_mapping(self) -> bool:
        return self.court_positions is not None and any(self.court_positions)
