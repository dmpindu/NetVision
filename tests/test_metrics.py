"""Unit tests for speed/distance and shot analytics."""
import math

import numpy as np
import pytest

from netvision.metrics import MADE, MISSED, UNKNOWN, ShotAnalyzer, SpeedDistanceCalculator, summarize_movement
from netvision.possession import NO_PLAYER

FPS = 30.0


# ---------------------------------------------------------------- speed and distance
def test_constant_speed():
    """TC-SPD-01: a player moving 5 m/s for 2 s covers 10 m at 18 km/h."""
    positions = [{1: (5.0 * f / FPS, 7.0)} for f in range(61)]
    distances, speeds = SpeedDistanceCalculator(FPS, window_frames=10, smoothing_frames=1).calculate(positions)
    assert distances[-1][1] == pytest.approx(10.0, abs=1e-6)
    assert speeds[-1][1] == pytest.approx(18.0, abs=1e-6)
    summary = summarize_movement(distances, speeds)
    assert summary[1]["distance_m"] == pytest.approx(10.0)
    assert summary[1]["max_speed_kmh"] == pytest.approx(18.0)


def test_impossible_jump_is_ignored():
    """TC-SPD-02: a teleport faster than 40 km/h (ID switch) is not counted as distance."""
    positions = [{1: (0.0, 0.0)} for _ in range(10)] + [{1: (20.0, 0.0)} for _ in range(11)]
    distances, _ = SpeedDistanceCalculator(FPS, window_frames=5, smoothing_frames=1).calculate(positions)
    assert distances[-1][1] == pytest.approx(0.0)


def test_smoothing_reduces_jitter_distance():
    """TC-SPD-03: a standing player with 3 cm jitter shows far less distance with smoothing."""
    rng = np.random.default_rng(0)
    positions = [{1: tuple(np.array([10.0, 5.0]) + rng.normal(0, 0.03, 2))} for _ in range(300)]
    raw, _ = SpeedDistanceCalculator(FPS, window_frames=1, smoothing_frames=1).calculate(positions)
    smooth, _ = SpeedDistanceCalculator(FPS, window_frames=10, smoothing_frames=9).calculate(positions)
    assert raw[-1][1] > 5.0
    assert smooth[-1][1] < min(1.0, raw[-1][1] / 5)


def test_speed_distance_edge_cases():
    """TC-SPD-04: invalid fps raises; players appear and disappear without errors."""
    with pytest.raises(ValueError):
        SpeedDistanceCalculator(0)
    positions = [{1: (0.0, 0.0)}, {}, {2: (1.0, 1.0)}, {1: (0.1, 0.0), 2: (1.0, 1.0)}]
    distances, speeds = SpeedDistanceCalculator(FPS, window_frames=1).calculate(positions)
    assert distances[1] == {} and set(distances[3]) == {1, 2}
    assert summarize_movement([], []) == {}


# ---------------------------------------------------------------- shots
SHOOTER_BOX = [500.0, 300.0, 540.0, 390.0]  # 90 px tall player
OPPONENT_BOX = [700.0, 300.0, 740.0, 390.0]


def build_shot(vx=8.0, angle_deg=55.0, apex_frames=15, flight=30, hold=15, tail=15, rise_scale=1.0):
    """Ball held for ``hold`` frames, a parabolic flight, then caught by player 2."""
    hand = (535.0, 345.0)
    vy = vx * math.tan(math.radians(angle_deg)) * rise_scale
    gravity = vy / apex_frames
    balls, possession = [], []
    for f in range(hold):
        balls.append([hand[0] - 5, hand[1] - 5, hand[0] + 5, hand[1] + 5])
        possession.append(1)
    for t in range(1, flight + 1):
        x, y = hand[0] + vx * t, hand[1] - vy * t + 0.5 * gravity * t * t
        balls.append([x - 5, y - 5, x + 5, y + 5])
        possession.append(NO_PLAYER)
    for _ in range(tail):
        balls.append(list(balls[-1]))
        possession.append(2)
    players = [{1: SHOOTER_BOX, 2: OPPONENT_BOX} for _ in balls]
    truth = {"release": hold, "angle": math.degrees(math.atan2(vy, vx)), "apex_m": vy * vy / (2 * gravity) / 90 * 2.0}
    return possession, players, balls, truth


def test_shot_detected_with_correct_metrics():
    """TC-SHT-01: a parabolic flight is a shot with the right release frame, angle and apex height."""
    possession, players, balls, truth = build_shot()
    shots = ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2})
    assert len(shots) == 1
    shot = shots[0]
    assert (shot.shooter_id, shot.team) == (1, 1)
    assert shot.release_frame == truth["release"]
    assert shot.release_angle_deg == pytest.approx(truth["angle"], abs=0.5)
    assert shot.apex_height_m == pytest.approx(truth["apex_m"], rel=0.02)
    assert shot.apex_frame == truth["release"] + 14
    assert shot.result == UNKNOWN
    assert shot.contest_distance_m is None
    assert "trajectory" not in shot.to_dict() and "trajectory" in shot.to_dict(include_trajectory=True)


def test_shot_release_found_when_possession_ends_late():
    """TC-SHT-02: if possession lasts a few frames into the flight, the release is walked back to the true frame."""
    possession, players, balls, truth = build_shot()
    for f in range(truth["release"], truth["release"] + 3):
        possession[f] = 1
    shot = ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2})[0]
    assert shot.release_frame == truth["release"]
    assert shot.release_angle_deg == pytest.approx(truth["angle"], abs=0.5)


def test_flat_pass_is_not_a_shot():
    """TC-SHT-03: a flight that never rises above the shooter's head is not a shot."""
    possession, players, balls, _ = build_shot(angle_deg=10)
    assert ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2}) == []


def test_pass_upward_on_screen_is_not_a_shot():
    """TC-SHT-04: a straight pass toward a player higher on screen (no fall after the peak) is not a shot."""
    possession = [1] * 10 + [NO_PLAYER] * 12 + [2] * 15
    balls = [[530, 340, 540, 350]] * 10 + [[530 + 8 * t, 340 - 15 * t, 540 + 8 * t, 350 - 15 * t] for t in range(1, 13)]
    balls += [balls[-1]] * 15
    players = [{1: SHOOTER_BOX, 2: [600, 80, 640, 170]} for _ in balls]
    assert ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 1}) == []


def test_short_or_unseen_flight_is_not_a_shot():
    """TC-SHT-05: flights shorter than 8 visible frames, or with no shooter box, are ignored."""
    possession, players, balls, _ = build_shot(flight=6)
    assert ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2}) == []
    possession, players, balls, _ = build_shot()
    players = [{2: OPPONENT_BOX} for _ in balls]
    assert ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2}) == []


def test_contest_distance_and_shot_location():
    """TC-SHT-06: contest distance is the court distance to the closest opponent at release."""
    possession, players, balls, _ = build_shot()
    court = [{1: (20.0, 7.5), 2: (21.5, 9.5), 3: (20.5, 7.5)} for _ in balls]  # 3 is a teammate
    shot = ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2, 3: 1}, court)[0]
    assert shot.contest_distance_m == pytest.approx(2.5)
    assert shot.closest_defender_id == 2
    assert shot.shooter_court_xy == (20.0, 7.5)
    assert shot.distance_to_basket_m == pytest.approx(6.425)


def test_make_and_miss_with_rim():
    """TC-SHT-07: the ball dropping through the rim is a make; seen rim without that is a miss."""
    possession, players, balls, _ = build_shot()
    # Rim placed on the descending part of the arc, 25 frames after release
    cx, cy = (balls[15 + 24][0] + balls[15 + 24][2]) / 2, (balls[15 + 24][1] + balls[15 + 24][3]) / 2
    rim = [cx - 20, cy - 3, cx + 20, cy + 3]
    made = ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2}, rim_positions=[rim] * len(balls))
    assert made[0].result == MADE
    far_rim = [rim[0] + 300, rim[1], rim[2] + 300, rim[3]]
    missed = ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2}, rim_positions=[far_rim] * len(balls))
    assert missed[0].result == MISSED
    unseen = ShotAnalyzer(FPS).analyze(possession, players, balls, {1: 1, 2: 2}, rim_positions=[None] * len(balls))
    assert unseen[0].result == UNKNOWN
