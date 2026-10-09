"""Configuration for the NetVision pipeline.

All tunable numbers live here so the rest of the code has no magic values.
A config can be loaded from a YAML file and any field can be overridden from
the command line.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

# Pixel based thresholds in the tutorial were tuned on 1280x720 video.
# NetVision scales them by (frame height / REFERENCE_HEIGHT).
REFERENCE_HEIGHT = 720


@dataclass
class ModelConfig:
    """Paths to model weights and the class names each model uses."""

    # Fine-tuned weights go in models/. When a path does not exist the
    # pipeline falls back to the stock COCO model below.
    player_model: str = "models/player_detector.pt"
    ball_model: str = "models/ball_detector_model.pt"
    rim_model: Optional[str] = "models/rim_detector.pt"
    court_keypoint_model: Optional[str] = "models/court_keypoint_detector.pt"
    fallback_model: str = "yolo11n.pt"

    # A JSON file with pixel positions of court keypoints for a fixed camera.
    # Used when no keypoint model is available.
    court_calibration: Optional[str] = None

    player_class_names: tuple[str, ...] = ("player", "person")
    ball_class_names: tuple[str, ...] = ("ball", "basketball", "sports ball")
    rim_class_names: tuple[str, ...] = ("rim", "hoop", "basket", "basketball hoop")

    player_confidence: float = 0.5
    ball_confidence: float = 0.15
    rim_confidence: float = 0.3
    keypoint_confidence: float = 0.5

    device: Optional[str] = None  # "cpu", "cuda", "mps" or None for auto
    batch_size: int = 16


@dataclass
class TrackingConfig:
    tracker: str = "bytetrack"
    lost_track_buffer: int = 60
    track_activation_threshold: float = 0.5
    minimum_matching_threshold: float = 0.1
    # Ball cleaning (tutorial: 25 px of allowed travel per missing frame)
    ball_max_px_per_frame: float = 25.0
    ball_max_interpolation_gap: int = 30


@dataclass
class TeamConfig:
    classifier: str = "auto"  # "auto", "clip", "color"
    clip_model: str = "patrickjohncyh/fashion-clip"
    team1_label: str = "white shirt"
    team2_label: str = "dark blue shirt"
    sample_every_n_frames: int = 5
    max_samples_per_track: int = 15
    # Color classifier: tracks this far from both team colors become team 0
    # (referees, fans, bench). Set to 0 to disable.
    outlier_factor: float = 3.0
    outlier_min_distance: float = 25.0


@dataclass
class PossessionConfig:
    possession_threshold_px: float = 50.0
    min_possession_frames: int = 11
    containment_threshold: float = 0.8
    # Once a holder is confirmed, also mark the frames that led up to it.
    backfill_confirmed: bool = True


@dataclass
class CourtConfig:
    length_m: float = 28.0
    width_m: float = 15.0
    keypoint_tolerance: float = 0.8
    min_keypoints: int = 4
    homography_carry_frames: int = 30
    off_court_margin_m: float = 1.5


@dataclass
class MetricsConfig:
    speed_window_frames: int = 10
    position_smoothing_frames: int = 9
    max_speed_kmh: float = 40.0
    # Shots
    min_flight_frames: int = 8
    max_flight_seconds: float = 3.0
    min_rise_ratio: float = 0.3
    min_descent_ratio: float = 0.25
    player_height_m: float = 2.0
    release_fit_frames: int = 6


@dataclass
class OutputConfig:
    output_dir: str = "output"
    write_video: bool = True
    video_extension: str = ".mp4"
    use_stubs: bool = False
    stub_dir: str = "stubs"
    draw_keypoints: bool = True
    draw_tactical_view: bool = True
    tactical_view_width_px: int = 300


@dataclass
class NetVisionConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    max_frames: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetVisionConfig":
        config = cls()
        for section_name, values in (data or {}).items():
            if section_name == "max_frames":
                config.max_frames = values
                continue
            if not hasattr(config, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            section = getattr(config, section_name)
            valid = {f.name for f in fields(section)}
            for key, value in (values or {}).items():
                if key not in valid:
                    raise ValueError(f"Unknown config key: {section_name}.{key}")
                if isinstance(getattr(section, key), tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(section, key, value)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetVisionConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


def pixel_scale(frame_height: int) -> float:
    """Factor to scale tutorial pixel thresholds to the current video size."""
    if frame_height <= 0:
        return 1.0
    return frame_height / REFERENCE_HEIGHT
