"""Unit tests for netvision.utils and netvision.config."""

import numpy as np
import pytest

from netvision.config import NetVisionConfig, pixel_scale
from netvision.utils import (
    clip_box,
    containment_ratio,
    get_center,
    get_foot_position,
    get_height,
    get_video_info,
    get_width,
    iter_batches,
    measure_distance,
    measure_xy_distance,
    read_stub,
    read_video,
    save_stub,
    save_video,
    stub_key,
)


# ---------------------------------------------------------------- bbox
def test_bbox_center_width_height_foot():
    """TC-UTL-01: center, width, height and foot point of a box."""
    box = (10, 20, 50, 100)
    assert get_center(box) == (30, 60)
    assert get_width(box) == 40
    assert get_height(box) == 80
    assert get_foot_position(box) == (30, 100)


def test_measure_distance():
    """TC-UTL-02: Euclidean distance (3-4-5 triangle) and signed xy distance."""
    assert measure_distance((0, 0), (3, 4)) == 5
    assert measure_xy_distance((5, 7), (2, 3)) == (3, 4)


@pytest.mark.parametrize(
    "inner,outer,expected",
    [
        ((10, 10, 20, 20), (0, 0, 100, 100), 1.0),
        ((90, 90, 110, 110), (0, 0, 100, 100), 0.25),
        ((200, 200, 210, 210), (0, 0, 100, 100), 0.0),
        ((5, 5, 5, 5), (0, 0, 100, 100), 0.0),
    ],
)
def test_containment_ratio(inner, outer, expected):
    """TC-UTL-03: containment ratio for full, partial, no overlap and zero-area boxes."""
    assert containment_ratio(inner, outer) == pytest.approx(expected)


def test_clip_box_to_image():
    """TC-UTL-04: boxes are clipped to the image and converted to ints."""
    assert clip_box((-5.2, 10.5, 1300.1, 800), 1280, 720) == (0, 10, 1280, 720)


# ---------------------------------------------------------------- config
def test_config_defaults_match_tutorial_thresholds():
    """TC-CFG-01: default thresholds match the tutorial (25 px, 50 px, 11 frames, 80 percent)."""
    config = NetVisionConfig()
    assert config.tracking.ball_max_px_per_frame == 25
    assert config.possession.possession_threshold_px == 50
    assert config.possession.min_possession_frames == 11
    assert config.possession.containment_threshold == 0.8
    assert (config.court.length_m, config.court.width_m) == (28, 15)


def test_config_from_dict_overrides_and_validates():
    """TC-CFG-02: overrides apply, lists become tuples, unknown keys are rejected."""
    config = NetVisionConfig.from_dict({"possession": {"min_possession_frames": 5},
                                        "models": {"ball_class_names": ["ball"]}, "max_frames": 10})
    assert config.possession.min_possession_frames == 5
    assert config.models.ball_class_names == ("ball",)
    assert config.max_frames == 10
    with pytest.raises(ValueError):
        NetVisionConfig.from_dict({"possession": {"nope": 1}})
    with pytest.raises(ValueError):
        NetVisionConfig.from_dict({"nope": {}})


def test_config_yaml_round_trip(tmp_path):
    """TC-CFG-03: the shipped configs/default.yaml loads and equals the defaults."""
    from pathlib import Path

    default_yaml = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert NetVisionConfig.from_yaml(default_yaml).to_dict() == NetVisionConfig().to_dict()
    custom = tmp_path / "c.yaml"
    custom.write_text("team:\n  classifier: color\n")
    assert NetVisionConfig.from_yaml(custom).team.classifier == "color"


def test_pixel_scale():
    """TC-CFG-04: pixel thresholds scale with video height (720p = 1.0)."""
    assert pixel_scale(720) == 1.0
    assert pixel_scale(1080) == 1.5
    assert pixel_scale(0) == 1.0


# ---------------------------------------------------------------- video
def _frames(n=7, h=48, w=64):
    return [np.full((h, w, 3), i * 30, dtype=np.uint8) for i in range(n)]


def test_video_write_read_round_trip(tmp_path):
    """TC-VID-01: saved video can be read back with the same size and frame count."""
    path = tmp_path / "v.mp4"
    save_video(_frames(), path, fps=24)
    info = get_video_info(path)
    assert (info.width, info.height, info.frame_count) == (64, 48, 7)
    assert info.fps == pytest.approx(24)
    assert info.duration_s == pytest.approx(7 / 24)
    assert len(read_video(path)) == 7
    assert len(read_video(path, max_frames=3)) == 3


def test_iter_batches(tmp_path):
    """TC-VID-02: frames are streamed in batches with a smaller last batch."""
    path = tmp_path / "v.avi"
    save_video(_frames(), path)
    assert [len(b) for b in iter_batches(path, 3)] == [3, 3, 1]
    with pytest.raises(ValueError):
        list(iter_batches(path, 0))


def test_video_errors(tmp_path):
    """TC-VID-03: missing file, bad extension and empty frame list raise clear errors."""
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / "missing.mp4")
    with pytest.raises(ValueError):
        save_video(_frames(), tmp_path / "v.xyz")
    with pytest.raises(ValueError):
        save_video([], tmp_path / "v.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(IOError):
        get_video_info(bad)


# ---------------------------------------------------------------- stubs
def test_stub_round_trip_and_validation(tmp_path):
    """TC-STB-01: stubs round trip, and missing, wrong length or corrupt stubs return None."""
    path = tmp_path / "stubs" / "s.pkl"
    save_stub(path, [1, 2, 3])
    assert read_stub(path) == [1, 2, 3]
    assert read_stub(path, expected_length=3) == [1, 2, 3]
    assert read_stub(path, expected_length=4) is None
    assert read_stub(tmp_path / "missing.pkl") is None
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"\x80garbage")
    assert read_stub(corrupt) is None


def test_stub_key_depends_on_settings(tmp_path):
    """TC-STB-02: the cache key changes when model settings change."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 10)
    assert stub_key(video, "a") == stub_key(video, "a")
    assert stub_key(video, "a") != stub_key(video, "b")
    assert stub_key(video, "a").startswith("clip_")
