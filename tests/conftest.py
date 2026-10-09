"""Shared fixtures.

The synthetic clip and one full pipeline run are created once per test
session because rendering video takes a few seconds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netvision.config import NetVisionConfig  # noqa: E402
from netvision.pipeline import NetVisionPipeline  # noqa: E402
from netvision.testing.synthetic import GroundTruthDetector, GroundTruthKeypoints, generate_clip  # noqa: E402

BALL_DROPOUTS = (50, 51, 52, 100, 160, 161)
FALSE_BALLS = {120: [100.0, 650.0, 115.0, 665.0]}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def map_ground_truth_ids(clip, player_tracks, frame: int = 10) -> dict[int, int]:
    """Ground truth player ID -> tracker ID, matched by box overlap."""
    mapping = {}
    for gt_id, gt_box in clip.player_boxes[frame].items():
        track_id, box = max(player_tracks[frame].items(), key=lambda item: iou(item[1], gt_box))
        assert iou(box, gt_box) > 0.8
        mapping[gt_id] = track_id
    return mapping


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    return generate_clip(path)


def make_fake_pipeline(clip, config=None, keypoints=True, rim_detector=None, **detector_kwargs):
    detector_kwargs.setdefault("ball_dropouts", BALL_DROPOUTS)
    detector_kwargs.setdefault("false_balls", FALSE_BALLS)
    detector = GroundTruthDetector(clip, **detector_kwargs)
    keypoint_source = GroundTruthKeypoints(clip, missing_frames=range(200, 210), corrupt={5: (3, (900.0, 100.0))}) if keypoints else None
    return NetVisionPipeline(config or NetVisionConfig(), detector, keypoint_source=keypoint_source, rim_detector=rim_detector)


@pytest.fixture(scope="session")
def pipeline_run(synthetic_clip, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("out")
    pipeline = make_fake_pipeline(synthetic_clip)
    outputs = pipeline.run(synthetic_clip.path, out_dir)
    mapping = map_ground_truth_ids(synthetic_clip, outputs.result.player_tracks)
    return outputs, mapping


@pytest.fixture
def blank_frame():
    return np.full((720, 1280, 3), 90, dtype=np.uint8)


def real_video_path():
    path = os.environ.get("NETVISION_REAL_VIDEO")
    return Path(path) if path and Path(path).exists() else None
