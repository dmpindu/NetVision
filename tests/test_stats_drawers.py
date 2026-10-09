"""Unit tests for statistics export and the video drawers."""
import csv
import json

import numpy as np
import pytest

from netvision.drawers import (
    BallTracksDrawer,
    CourtKeypointsDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    PlayerTracksDrawer,
    ShotDrawer,
    SpeedDistanceDrawer,
    TacticalViewDrawer,
    TeamBallControlDrawer,
    default_drawers,
)
from netvision.events import PassInterceptionDetector
from netvision.metrics import Shot, build_stats, write_events_csv, write_json, write_tracking_csv
from netvision.metrics.stats import EVENT_COLUMNS, TRACKING_COLUMNS
from netvision.possession import team_possession
from netvision.results import AnalysisResult
from netvision.utils import VideoInfo


def small_result(with_court=True):
    """Four frames, two players per team, one pass and one shot."""
    players = [{1: [100, 100, 140, 200], 2: [300, 100, 340, 200], 3: [500, 100, 540, 200]} for _ in range(4)]
    possession = [1, 1, 2, 3]
    teams = {1: 1, 2: 1, 3: 2}
    events = PassInterceptionDetector().detect(possession, teams)
    shot = Shot(3, 2, 3, 3, 3, 52.0, 1.8, 1.0, 1.2, 2, (20.0, 7.0), 6.4, "unknown", [(3, 520.0, 90.0)])
    result = AnalysisResult(
        video=VideoInfo(30.0, 1280, 720, 4),
        player_tracks=players,
        ball_positions=[[130, 140, 140, 150], None, [330, 140, 340, 150], [530, 140, 540, 150]],
        raw_ball_positions=[None] * 4,
        rim_positions=None,
        keypoints=[np.array([[10.0, 10.0], [np.nan, np.nan]]), None, None, None],
        team_assignment=teams,
        possession=possession,
        team_possession=team_possession(possession, teams),
        events=events,
        shots=[shot],
    )
    if with_court:
        result.court_positions = [{1: (5.0, 5.0), 2: (8.0, 5.0), 3: (20.0, 7.0)}] * 4
        result.distances = [{1: 0.5 * f, 2: 0.0, 3: 1.0} for f in range(4)]
        result.speeds = [{1: 6.0, 2: 0.0, 3: 3.6} for _ in range(4)]
    return result


def test_build_stats_values():
    """TC-STA-01: team and player stats add up (possession percent, passes, interceptions, shots, distance)."""
    stats = build_stats(small_result(), {"models": {"player_model": "fake"}})
    assert stats["teams"]["1"]["possession_pct"] == pytest.approx(75.0)
    assert stats["teams"]["2"]["possession_pct"] == pytest.approx(25.0)
    assert stats["teams"]["1"]["passes"] == 1
    assert stats["teams"]["2"]["interceptions"] == 1
    assert stats["teams"]["2"]["shots"] == 1
    assert stats["teams"]["2"]["avg_release_angle_deg"] == pytest.approx(52.0)
    assert stats["players"]["1"]["passes_made"] == 1 and stats["players"]["2"]["passes_received"] == 1
    assert stats["players"]["2"]["turnovers"] == 1 and stats["players"]["3"]["interceptions"] == 1
    assert stats["players"]["1"]["distance_m"] == pytest.approx(1.5)
    assert stats["metadata"]["ball_visible_pct"] == pytest.approx(75.0)
    assert stats["metadata"]["models"]["player_model"] == "fake"
    assert stats["shots"][0]["release_angle_deg"] == 52.0


def test_build_stats_without_court_mapping():
    """TC-STA-02: without a court mapping, distance and speed are null instead of failing."""
    stats = build_stats(small_result(with_court=False))
    assert stats["players"]["1"]["distance_m"] is None
    assert stats["metadata"]["court_mapping"] is False


def test_export_files(tmp_path):
    """TC-STA-03: JSON, tracking CSV (one row per player per frame) and events CSV are written correctly."""
    result = small_result()
    result.stats = build_stats(result)
    data = json.loads(write_json(result.stats, tmp_path / "s.json").read_text())
    assert data["schema_version"] == "1.0"
    with open(write_tracking_csv(result, tmp_path / "t.csv")) as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == TRACKING_COLUMNS
    assert len(rows) == 12
    assert rows[0]["has_ball"] == "1" and rows[0]["court_x_m"] == "5.0"
    with open(write_events_csv(result, tmp_path / "e.csv")) as handle:
        events = list(csv.DictReader(handle))
    assert list(events[0].keys()) == EVENT_COLUMNS
    assert [e["type"] for e in events] == ["pass", "interception", "shot"]


@pytest.mark.parametrize(
    "drawer",
    [PlayerTracksDrawer(), BallTracksDrawer(), CourtKeypointsDrawer(), TeamBallControlDrawer(),
     PassInterceptionDrawer(), SpeedDistanceDrawer(), TacticalViewDrawer(), ShotDrawer(), FrameNumberDrawer()],
    ids=lambda d: type(d).__name__,
)
def test_each_drawer_changes_the_frame(drawer):
    """TC-DRW-01: every drawer draws something on a frame without errors."""
    result = small_result()
    drawer.prepare(result)
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    index = 3 if isinstance(drawer, ShotDrawer) else 0
    out = drawer.draw(frame.copy(), index, result)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_drawers_skip_missing_data():
    """TC-DRW-02: drawers that need court data or keypoints leave the frame alone when it is missing."""
    result = small_result(with_court=False)
    result.keypoints = None
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    for drawer in (TacticalViewDrawer(), SpeedDistanceDrawer(), CourtKeypointsDrawer()):
        assert np.array_equal(drawer.draw(frame.copy(), 0, result), frame)
    assert np.array_equal(CourtKeypointsDrawer().draw(frame.copy(), 1, small_result()), frame)


def test_team_ball_control_running_percentages():
    """TC-DRW-03: the ball control overlay shows the running share up to the current frame."""
    drawer = TeamBallControlDrawer()
    drawer.prepare(small_result())
    assert drawer.percentages[0] == (100.0, 0.0)
    assert drawer.percentages[3] == (75.0, 25.0)


def test_default_drawers_options():
    """TC-DRW-04: keypoint and tactical view drawers can be switched off."""
    names = [type(d).__name__ for d in default_drawers(draw_keypoints=False, draw_tactical_view=False)]
    assert "CourtKeypointsDrawer" not in names and "TacticalViewDrawer" not in names
    assert len(default_drawers()) == 9
