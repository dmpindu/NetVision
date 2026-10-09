"""End to end tests on the synthetic clip with known ground truth.

The detectors are replaced with fakes that replay the ground truth with
pixel noise, missed ball frames and a false ball detection. Everything else
(tracking, team colors from the rendered pixels, possession, events, court
mapping, speed, shots, drawing, export) is the real code.
"""
import json
import math

import cv2
import numpy as np
import pytest

from conftest import BALL_DROPOUTS, make_fake_pipeline, map_ground_truth_ids
from netvision.config import NetVisionConfig
from netvision.detection import StaticCourtCalibration
from netvision.metrics import MADE
from netvision.possession import NO_PLAYER
from netvision.testing.synthetic import GroundTruthDetector, generate_clip
from netvision.utils import get_video_info

pytestmark = pytest.mark.slow


def test_synthetic_ground_truth_is_consistent(synthetic_clip):
    """TC-E2E-01: the generated clip has the scripted events, all keypoints in frame and a readable video."""
    truth = synthetic_clip.ground_truth
    assert [e["type"] for e in truth["events"]] == ["pass", "interception", "pass", "rebound"]
    assert truth["shot"]["shooter"] == 5 and truth["shot"]["release_frame"] == 240
    keypoints = synthetic_clip.keypoints
    assert ((keypoints[:, 0] >= 0) & (keypoints[:, 0] < 1280) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < 720)).all()
    info = get_video_info(synthetic_clip.path)
    assert (info.width, info.height, info.frame_count) == (1280, 720, 330)


def test_outputs_are_written(pipeline_run, synthetic_clip):
    """TC-E2E-02: annotated video (same length and size), JSON, tracking CSV and events CSV exist."""
    outputs, _ = pipeline_run
    info = get_video_info(outputs.video_path)
    assert (info.width, info.height, info.frame_count) == (1280, 720, synthetic_clip.num_frames)
    stats = json.loads(outputs.stats_path.read_text())
    assert set(stats) == {"schema_version", "metadata", "teams", "players", "events", "shots"}
    assert outputs.tracking_csv_path.stat().st_size > 10_000
    assert outputs.events_csv_path.read_text().count("\n") == 6  # header + 4 events + 1 shot


def test_every_player_keeps_one_id(pipeline_run, synthetic_clip):
    """TC-E2E-03: each of the 7 people keeps a single track ID for the whole clip (no ID switches)."""
    outputs, mapping = pipeline_run
    tracks = outputs.result.player_tracks
    assert len(set(mapping.values())) == 7
    for frame in range(10, synthetic_clip.num_frames):
        assert set(tracks[frame]) == set(mapping.values())


def test_team_assignment_matches_jerseys(pipeline_run, synthetic_clip):
    """TC-E2E-04: teams come out right from the rendered jersey colors, and the referee is team 0."""
    outputs, mapping = pipeline_run
    for gt_id, team in synthetic_clip.teams.items():
        assert outputs.result.team_assignment[mapping[gt_id]] == team


def test_events_match_ground_truth(pipeline_run, synthetic_clip):
    """TC-E2E-05: pass, interception, pass and defensive rebound are found with the right players and teams."""
    outputs, mapping = pipeline_run
    found = [(e.type, e.from_player, e.to_player, e.team) for e in outputs.result.events]
    expected = [(e["type"], mapping[e["from"]], mapping[e["to"]], e["team"]) for e in synthetic_clip.ground_truth["events"]]
    assert found == expected
    for event, truth in zip(outputs.result.events, synthetic_clip.ground_truth["events"]):
        assert abs(event.frame - truth["frame"]) <= 12
    assert outputs.result.events[-1].detail == "defensive"
    teams = outputs.result.stats["teams"]
    assert (teams["1"]["passes"], teams["2"]["passes"], teams["2"]["interceptions"], teams["1"]["rebounds"]) == (1, 1, 1, 1)


def test_possession_matches_ground_truth(pipeline_run, synthetic_clip):
    """TC-E2E-06: the per-frame ball holder matches ground truth away from the moments the ball changes hands."""
    outputs, mapping = pipeline_run
    inverse = {track: gt for gt, track in mapping.items()}
    truth = synthetic_clip.holders
    changes = [f for f in range(1, len(truth)) if truth[f] != truth[f - 1]]
    checked = agree = 0
    for frame, holder in enumerate(outputs.result.possession):
        if any(abs(frame - c) <= 6 for c in changes):
            continue
        checked += 1
        agree += inverse.get(holder, NO_PLAYER) == truth[frame]
    assert checked > 200
    assert agree / checked >= 0.95


def test_shot_metrics_match_ground_truth(pipeline_run, synthetic_clip):
    """TC-E2E-07: one shot with release frame +/-3, angle +/-5 deg, arc +/-10 percent, contest +/-0.5 m."""
    outputs, mapping = pipeline_run
    truth = synthetic_clip.ground_truth["shot"]
    assert len(outputs.result.shots) == 1
    shot = outputs.result.shots[0]
    assert shot.shooter_id == mapping[truth["shooter"]] and shot.team == truth["team"]
    assert abs(shot.release_frame - truth["release_frame"]) <= 3
    assert shot.release_angle_deg == pytest.approx(truth["release_angle_deg"], abs=5)
    assert shot.apex_height_m == pytest.approx(truth["apex_height_m"], rel=0.10)
    assert shot.contest_distance_m == pytest.approx(truth["contest_distance_m"], abs=0.5)
    assert shot.closest_defender_id == mapping[truth["closest_defender"]]
    assert math.dist(shot.shooter_court_xy, truth["shooter_court_xy"]) < 0.3


def test_distance_matches_ground_truth(pipeline_run, synthetic_clip):
    """TC-E2E-08: total distance within 5 percent, and each player within 0.5 m or 5 percent."""
    outputs, mapping = pipeline_run
    players = outputs.result.stats["players"]
    truth = synthetic_clip.ground_truth["distances_m"]
    estimated = {gt: players[str(track)]["distance_m"] for gt, track in mapping.items()}
    assert sum(estimated.values()) == pytest.approx(sum(truth.values()), rel=0.05)
    for gt_id, value in estimated.items():
        assert abs(value - truth[gt_id]) <= max(0.5, 0.05 * truth[gt_id]), gt_id


def test_ball_cleaning_in_pipeline(pipeline_run, synthetic_clip):
    """TC-E2E-09: the false far-away ball is removed and dropped ball frames are filled in."""
    outputs, _ = pipeline_run
    result = outputs.result
    assert result.raw_ball_positions[120] == [100.0, 650.0, 115.0, 665.0]
    cx = (result.ball_positions[120][0] + result.ball_positions[120][2]) / 2
    true_cx = (synthetic_clip.ball_boxes[120][0] + synthetic_clip.ball_boxes[120][2]) / 2
    assert abs(cx - true_cx) < 15
    for frame in BALL_DROPOUTS:
        assert result.raw_ball_positions[frame] is None and result.ball_positions[frame] is not None


def test_court_mapping_survives_bad_and_missing_keypoints(pipeline_run):
    """TC-E2E-10: a corrupted keypoint is rejected and 10 frames without keypoints reuse the last homography."""
    outputs, _ = pipeline_run
    result = outputs.result
    assert all(h is not None for h in result.homographies)
    assert result.stats["metadata"]["court_mapping"] is True
    assert not any("Court mapping available" in n for n in result.notes)


def test_annotated_video_has_overlays(pipeline_run, synthetic_clip):
    """TC-E2E-11: annotated frames differ from the input and the tactical map area is drawn."""
    outputs, _ = pipeline_run
    original = cv2.VideoCapture(str(synthetic_clip.path))
    annotated = cv2.VideoCapture(str(outputs.video_path))
    for _ in range(100):
        ok1, a = original.read()
        ok2, b = annotated.read()
    original.release()
    annotated.release()
    assert ok1 and ok2
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) > 2
    assert np.mean(np.abs(a[40:200, 20:320].astype(int) - b[40:200, 20:320].astype(int))) > 20


def test_stub_cache_skips_detection(synthetic_clip, tmp_path):
    """TC-E2E-12: with stubs on, the second run loads cached detections and never calls the detector."""
    config = NetVisionConfig()
    config.output.use_stubs = True
    config.output.stub_dir = str(tmp_path / "stubs")
    config.output.write_video = False
    first = make_fake_pipeline(synthetic_clip, config).run(synthetic_clip.path, tmp_path / "a")

    class ExplodingDetector:
        def detect(self, frames):
            raise AssertionError("detector should not run when a stub exists")

    second_pipeline = make_fake_pipeline(synthetic_clip, config)
    second_pipeline.player_detector = second_pipeline.ball_detector = ExplodingDetector()
    second = second_pipeline.run(synthetic_clip.path, tmp_path / "b")
    assert second.result.possession == first.result.possession
    assert len(list((tmp_path / "stubs").glob("*.pkl"))) == 1


def test_runs_without_court_model(synthetic_clip, tmp_path):
    """TC-E2E-13: with no keypoints the pipeline still finds events and shots and explains what is off."""
    config = NetVisionConfig()
    config.output.write_video = False
    result = make_fake_pipeline(synthetic_clip, config, keypoints=False).run(synthetic_clip.path, tmp_path).result
    assert result.court_positions is None and result.speeds is None
    assert [e.type for e in result.events] == ["pass", "interception", "pass", "rebound"]
    assert len(result.shots) == 1 and result.shots[0].contest_distance_m is None
    assert any("No court keypoint model" in n for n in result.notes)
    assert result.stats["players"]["0"]["distance_m"] is None


def test_manual_calibration_matches_keypoint_model(synthetic_clip, tmp_path):
    """TC-E2E-14: a 6-point manual calibration gives the same court positions as the keypoint model."""
    config = NetVisionConfig()
    config.output.write_video = False
    calibration = StaticCourtCalibration({i: synthetic_clip.keypoints[i] for i in (0, 5, 6, 7, 10, 15)})
    pipeline = make_fake_pipeline(synthetic_clip, config, keypoints=False)
    pipeline.keypoint_source = calibration
    result = pipeline.run(synthetic_clip.path, tmp_path).result
    mapping = map_ground_truth_ids(synthetic_clip, result.player_tracks)
    for gt_id in (1, 5):
        frame = 200
        expected = synthetic_clip.ground_truth["shot"]["shooter_court_xy"] if gt_id == 5 else None
        position = result.court_positions[frame][mapping[gt_id]]
        if expected:
            assert math.dist(position, expected) < 0.3
        assert 0 <= position[0] <= 28 and 0 <= position[1] <= 15


def test_rim_detector_makes_the_shot(synthetic_clip, tmp_path):
    """TC-E2E-15: with a rim placed on the ball's way down, the shot is reported as made."""
    frame = 268  # descending part of the arc
    box = synthetic_clip.ball_boxes[frame]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rim = (cx - 25, cy - 4, cx + 25, cy + 4)

    class FakeRim:
        def detect(self, frames):
            from netvision.detection import Detection

            return [[Detection(rim, 0.9, "rim")] for _ in frames]

    config = NetVisionConfig()
    config.output.write_video = False
    result = make_fake_pipeline(synthetic_clip, config, rim_detector=FakeRim()).run(synthetic_clip.path, tmp_path).result
    assert result.shots[0].result == MADE
    assert result.stats["teams"]["2"]["shots_made"] == 1


def test_scaled_video_uses_scaled_thresholds(tmp_path):
    """TC-E2E-16: pixel thresholds scale with resolution (same events on a 1080p-scaled detection set)."""
    clip = generate_clip(tmp_path / "small.mp4", write_video=True)
    scale = 1.5
    for frame in clip.player_boxes:
        for pid in frame:
            frame[pid] = [v * scale for v in frame[pid]]
    clip.ball_boxes = [[v * scale for v in b] if b else None for b in clip.ball_boxes]
    clip.keypoints = clip.keypoints * scale
    # Write a 1080p version of the video so the video size matches the boxes
    src = cv2.VideoCapture(str(clip.path))
    out = cv2.VideoWriter(str(tmp_path / "big.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    while True:
        ok, frame = src.read()
        if not ok:
            break
        out.write(cv2.resize(frame, (1920, 1080)))
    src.release()
    out.release()
    config = NetVisionConfig()
    config.output.write_video = False
    detector = GroundTruthDetector(clip, noise_px=1.5)
    from netvision.pipeline import NetVisionPipeline
    from netvision.testing.synthetic import GroundTruthKeypoints

    pipeline = NetVisionPipeline(config, detector, keypoint_source=GroundTruthKeypoints(clip))
    result = pipeline.run(tmp_path / "big.mp4", tmp_path / "out").result
    assert [e.type for e in result.events] == ["pass", "interception", "pass", "rebound"]
    assert len(result.shots) == 1
