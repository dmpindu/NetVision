"""Unit tests for detection helpers, player tracking and ball cleaning."""
import json

import numpy as np
import pytest

from netvision.detection import Detection, StaticCourtCalibration, matches, resolve_weights
from netvision.trackers import BallTracker, PlayerTracker, RimDetector, interpolate_positions, remove_wrong_detections


def det(box, conf=0.9, name="player"):
    return Detection(tuple(float(v) for v in box), conf, name)


# ---------------------------------------------------------------- detection helpers
def test_class_name_matching_is_case_and_separator_insensitive():
    """TC-DET-01: class names match regardless of case, underscores or dashes."""
    assert matches("Sports_Ball", ["sports ball"])
    assert matches("basketball-hoop", ["Basketball Hoop"])
    assert not matches("person", ["ball"])


def test_resolve_weights(tmp_path):
    """TC-DET-02: an existing weights file is used, otherwise the fallback."""
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    assert resolve_weights(str(weights), "yolo11n.pt") == str(weights)
    assert resolve_weights(str(tmp_path / "missing.pt"), "yolo11n.pt") == "yolo11n.pt"
    assert resolve_weights(str(tmp_path / "missing.pt"), None) is None
    assert resolve_weights(None, None) is None


def test_static_court_calibration(tmp_path):
    """TC-DET-03: a calibration JSON gives the same keypoints every frame; bad files are rejected."""
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"keypoints": {"0": [1, 2], "5": [3, 4], "10": [5, 6], "15": [7, 8]}}))
    calibration = StaticCourtCalibration.from_json(path)
    points = calibration.detect([None, None])
    assert len(points) == 2
    assert points[0][5].tolist() == [3, 4]
    assert np.isnan(points[0][1]).all()
    with pytest.raises(ValueError):
        StaticCourtCalibration({0: (1, 1), 1: (2, 2), 2: (3, 3)})
    with pytest.raises(ValueError):
        StaticCourtCalibration({0: (1, 1), 1: (2, 2), 2: (3, 3), 40: (4, 4)})


# ---------------------------------------------------------------- player tracker
def test_player_tracker_keeps_ids_for_moving_players():
    """TC-TRK-01: two moving players keep the same track IDs over 30 frames."""
    tracker = PlayerTracker(frame_rate=30)
    seen = []
    for f in range(30):
        frame = tracker.update([det((100 + 3 * f, 100, 150 + 3 * f, 250)), det((600 - 2 * f, 120, 650 - 2 * f, 270))])
        seen.append(frame)
    final_ids = set(seen[-1])
    assert len(final_ids) == 2
    for frame in seen[5:]:
        assert set(frame) == final_ids
    left_id = min(seen[-1], key=lambda i: seen[-1][i][0])
    assert all(frame[left_id][0] < 400 for frame in seen[5:])


def test_player_tracker_filters_classes_and_confidence():
    """TC-TRK-02: only player classes above the confidence threshold are tracked."""
    tracker = PlayerTracker(class_names=("player",), min_confidence=0.5)
    for _ in range(3):
        out = tracker.update([det((0, 0, 50, 100)), det((200, 0, 250, 100), name="ball"), det((400, 0, 450, 100), conf=0.2)])
    assert len(out) == 1
    assert tracker.update([]) == {}


def test_player_tracker_survives_short_dropout():
    """TC-TRK-03: a player missed for a few frames comes back with the same ID."""
    tracker = PlayerTracker(frame_rate=30, lost_track_buffer=30)
    ids = None
    for f in range(20):
        detections = [] if 10 <= f < 14 else [det((100 + f, 100, 150 + f, 250))]
        out = tracker.update(detections)
        if f == 9:
            ids = set(out)
    assert set(out) == ids


def test_unsupported_tracker_rejected():
    """TC-TRK-04: asking for a tracker that is not implemented raises an error."""
    with pytest.raises(ValueError):
        PlayerTracker(tracker="deepsort")


# ---------------------------------------------------------------- ball / rim selection
def test_ball_selector_picks_most_confident():
    """TC-BAL-01: the most confident ball is kept; other classes and weak detections are ignored."""
    tracker = BallTracker(min_confidence=0.2)
    detections = [det((0, 0, 10, 10), 0.3, "ball"), det((50, 50, 60, 60), 0.7, "Basketball"),
                  det((90, 90, 99, 99), 0.95, "player"), det((5, 5, 9, 9), 0.1, "ball")]
    assert tracker.select(detections) == [50, 50, 60, 60]
    assert tracker.select([det((0, 0, 1, 1), 0.1, "ball")]) is None
    assert RimDetector(("hoop",), 0.3).select([det((1, 1, 5, 5), 0.5, "Hoop")]) == [1, 1, 5, 5]


def test_remove_wrong_detections_drops_jumps():
    """TC-BAL-02: a detection that jumps further than 25 px per frame is removed."""
    positions = [[0, 0, 10, 10], [5, 0, 15, 10], [500, 500, 510, 510], [10, 0, 20, 10]]
    cleaned = remove_wrong_detections(positions, 25)
    assert cleaned[2] is None
    assert cleaned[3] == [10, 0, 20, 10]
    assert positions[2] is not None  # input not modified


def test_remove_wrong_detections_scales_with_gap():
    """TC-BAL-03: after missing frames the allowed travel grows (25 px times the gap)."""
    positions = [[0, 0, 10, 10], None, None, None, [70, 0, 80, 10]]
    assert remove_wrong_detections(positions, 25)[4] == [70, 0, 80, 10]  # 70 px over 4 frames is fine
    positions = [[0, 0, 10, 10], None, [70, 0, 80, 10]]
    assert remove_wrong_detections(positions, 25)[2] is None  # 70 px over 2 frames is not


def test_interpolate_positions():
    """TC-BAL-04: short gaps are filled linearly; long gaps and the edges stay empty."""
    positions = [None, [0, 0, 10, 10], None, None, [30, 0, 40, 10], None]
    filled = interpolate_positions(positions, max_gap=5)
    assert filled[0] is None and filled[5] is None
    assert filled[2] == pytest.approx([10, 0, 20, 10])
    assert filled[3] == pytest.approx([20, 0, 30, 10])
    long_gap = [[0, 0, 1, 1]] + [None] * 10 + [[10, 0, 11, 1]]
    assert interpolate_positions(long_gap, max_gap=5)[5] is None
