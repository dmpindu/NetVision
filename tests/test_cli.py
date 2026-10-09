"""Tests for the command line interface and the optional real-model smoke test."""
import json
import os

import numpy as np
import pytest

from conftest import make_fake_pipeline, real_video_path
from netvision import cli
from netvision.config import NetVisionConfig


def test_cli_arguments_override_config(tmp_path):
    """TC-CLI-01: command line flags override the YAML config."""
    config_file = tmp_path / "c.yaml"
    config_file.write_text("team:\n  classifier: clip\n")
    args = cli.build_parser().parse_args([
        "clip.mp4", "-c", str(config_file), "-o", "out", "--team-classifier", "color", "--ball-model", "b.pt",
        "--court-calibration", "cal.json", "--max-frames", "50", "--use-stubs", "--no-video", "--device", "cpu",
    ])
    config = cli.config_from_args(args)
    assert config.team.classifier == "color"
    assert config.models.ball_model == "b.pt"
    assert config.models.court_calibration == "cal.json"
    assert config.output.output_dir == "out"
    assert config.max_frames == 50 and config.output.use_stubs and not config.output.write_video
    assert config.models.device == "cpu"


def test_cli_missing_video_returns_error(tmp_path, capsys, monkeypatch):
    """TC-CLI-02: a missing input video prints an error and exits with code 1."""
    monkeypatch.setattr("netvision.pipeline.NetVisionPipeline.from_config", classmethod(lambda cls, c: make_fake_pipeline(None, c, keypoints=False)))
    assert cli.main([str(tmp_path / "nope.mp4"), "-o", str(tmp_path)]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_end_to_end(synthetic_clip, tmp_path, capsys, monkeypatch):
    """TC-CLI-03: the CLI runs the whole pipeline, prints a summary and writes the files."""
    monkeypatch.setattr(
        "netvision.pipeline.NetVisionPipeline.from_config",
        classmethod(lambda cls, config: make_fake_pipeline(synthetic_clip, config)),
    )
    code = cli.main([str(synthetic_clip.path), "-o", str(tmp_path), "--no-video"])
    output = capsys.readouterr().out
    assert code == 0
    assert "NetVision summary" in output and "Shot by #" in output
    stats = json.loads((tmp_path / "synthetic_stats.json").read_text())
    assert stats["teams"]["1"]["passes"] == 1
    assert not (tmp_path / "synthetic_annotated.mp4").exists()


def test_synthetic_generator_cli(tmp_path):
    """TC-CLI-04: the synthetic clip generator writes a video and a ground truth JSON."""
    from netvision.testing.synthetic import main

    assert main([str(tmp_path / "s.mp4")]) == 0
    truth = json.loads((tmp_path / "s.truth.json").read_text())
    assert truth["num_frames"] == 330 and len(truth["events"]) == 4


@pytest.mark.real_model
def test_real_yolo_detector_on_frame():
    """TC-REAL-01: the real YOLO wrapper returns detections in the expected format (needs weights)."""
    weights = os.environ.get("NETVISION_YOLO_WEIGHTS")
    if not weights or not os.path.exists(weights):
        pytest.skip("set NETVISION_YOLO_WEIGHTS to a YOLO .pt file")
    from netvision.detection import YoloDetector

    detector = YoloDetector(weights, confidence=0.1)
    output = detector.detect([np.zeros((360, 640, 3), np.uint8)] * 2)
    assert len(output) == 2 and all(isinstance(frame, list) for frame in output)
    assert detector.has_class(["person"]) or detector.has_class(["player"])


@pytest.mark.real_model
def test_real_video_smoke(tmp_path):
    """TC-REAL-02: the full pipeline with real YOLO weights runs on a real NBA clip (needs weights and video)."""
    video = real_video_path()
    weights = os.environ.get("NETVISION_YOLO_WEIGHTS")
    if video is None or not weights:
        pytest.skip("set NETVISION_REAL_VIDEO and NETVISION_YOLO_WEIGHTS")
    from netvision.pipeline import NetVisionPipeline

    config = NetVisionConfig()
    config.models.player_model = weights
    config.models.fallback_model = weights
    config.team.classifier = "color"
    config.max_frames = 60
    outputs = NetVisionPipeline.from_config(config).run(video, tmp_path)
    stats = json.loads(outputs.stats_path.read_text())
    assert stats["metadata"]["frames"] == 60
    assert len(stats["players"]) > 0
    assert outputs.video_path.exists()


@pytest.mark.real_model
def test_real_keypoint_detector_format():
    """TC-REAL-03: the YOLO pose wrapper returns one (18, 2) array per frame with NaN for missing points (needs pose weights)."""
    weights = os.environ.get("NETVISION_POSE_WEIGHTS")
    video = real_video_path()
    if not weights or not os.path.exists(weights) or video is None:
        pytest.skip("set NETVISION_POSE_WEIGHTS to a YOLO pose .pt file and NETVISION_REAL_VIDEO")
    from netvision.detection import YoloKeypointDetector
    from netvision.utils import read_video

    frames = read_video(video, max_frames=2) + [np.zeros((360, 640, 3), np.uint8)]
    output = YoloKeypointDetector(weights, confidence=0.3).detect(frames)
    assert len(output) == 3
    assert output[2] is None  # nothing in a black frame
    found = [o for o in output[:2] if o is not None]
    assert found and all(o.shape == (18, 2) for o in found)
    assert any(np.isfinite(o).any() for o in found)
