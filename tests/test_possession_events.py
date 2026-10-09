"""Unit tests for ball possession and pass / interception / rebound events."""
import pytest

from netvision.events import INTERCEPTION, PASS, REBOUND, PassInterceptionDetector, cumulative_event_counts
from netvision.possession import NO_PLAYER, BallAcquisitionDetector, possession_percentages, team_possession

PLAYER_A = [100, 100, 150, 250]
PLAYER_B = [400, 100, 450, 250]


def ball_at(x, y, r=5):
    return [x - r, y - r, x + r, y + r]


def test_key_points_include_nearest_edge_points():
    """TC-POS-01: key points include the box edges at the ball's height and width."""
    points = BallAcquisitionDetector.get_key_basketball_player_assignment_points(PLAYER_A, (170, 180))
    assert (100, 180) in points and (150, 180) in points
    assert (125, 100) in points and (150, 250) in points


def test_contained_ball_wins_over_distance():
    """TC-POS-02: a ball inside a player's box (>80 percent) goes to that player."""
    detector = BallAcquisitionDetector()
    assert detector.find_best_candidate_for_possession(ball_at(140, 180), {1: PLAYER_A, 2: PLAYER_B}) == 1


def test_distance_threshold():
    """TC-POS-03: the closest player within 50 px gets the ball, nobody beyond that."""
    detector = BallAcquisitionDetector(possession_threshold=50)
    assert detector.find_best_candidate_for_possession(ball_at(180, 180), {1: PLAYER_A, 2: PLAYER_B}) == 1
    assert detector.find_best_candidate_for_possession(ball_at(275, 180), {1: PLAYER_A, 2: PLAYER_B}) == NO_PLAYER
    assert detector.find_best_candidate_for_possession(ball_at(275, 180), {}) == NO_PLAYER


def test_minimum_consecutive_frames_and_backfill():
    """TC-POS-04: possession needs 11 straight frames; with backfill the lead-up frames are filled in."""
    players = [{1: PLAYER_A}] * 20
    ball = [ball_at(140, 180)] * 20
    possession = BallAcquisitionDetector(min_frames=11, backfill_confirmed=False).detect_ball_possession(players, ball)
    assert possession[:10] == [NO_PLAYER] * 10 and possession[10:] == [1] * 10
    possession = BallAcquisitionDetector(min_frames=11, backfill_confirmed=True).detect_ball_possession(players, ball)
    assert possession == [1] * 20
    short = BallAcquisitionDetector(min_frames=11).detect_ball_possession(players[:10], ball[:10])
    assert short == [NO_PLAYER] * 10


def test_missing_ball_breaks_the_streak():
    """TC-POS-05: a frame without a ball resets the consecutive frame counter."""
    players = [{1: PLAYER_A}] * 15
    ball = [ball_at(140, 180)] * 15
    ball[6] = None
    possession = BallAcquisitionDetector(min_frames=11).detect_ball_possession(players, ball)
    assert possession == [NO_PLAYER] * 15
    with pytest.raises(ValueError):
        BallAcquisitionDetector().detect_ball_possession(players, ball[:3])


def test_team_possession_and_percentages():
    """TC-POS-06: holder IDs map to teams and percentages ignore frames with no holder."""
    teams = team_possession([1, 1, 2, NO_PLAYER, 3], {1: 1, 2: 2})
    assert teams == [1, 1, 2, 0, 0]
    assert possession_percentages(teams) == {1: pytest.approx(66.666, 0.01), 2: pytest.approx(33.333, 0.01)}
    assert possession_percentages([0, 0]) == {1: 0.0, 2: 0.0}


TEAMS = {1: 1, 2: 1, 3: 2, 4: 2, 9: 0}


def test_pass_and_interception():
    """TC-EVT-01: same-team change is a pass, other-team change is an interception."""
    possession = [1, 1, -1, -1, 2, 2, -1, 3, 3]
    events = PassInterceptionDetector().detect(possession, TEAMS)
    assert [(e.type, e.frame, e.from_player, e.to_player, e.team) for e in events] == [
        (PASS, 4, 1, 2, 1),
        (INTERCEPTION, 7, 2, 3, 2),
    ]


def test_rebounds_after_shot():
    """TC-EVT-02: a change after the previous holder's shot is a rebound (defensive or offensive)."""
    detector = PassInterceptionDetector()
    events = detector.detect([1, 1, 1, -1, -1, 3, 3], TEAMS, {3: 1})
    assert [(e.type, e.detail, e.team) for e in events] == [(REBOUND, "defensive", 2)]
    events = detector.detect([1, 1, -1, -1, 2, 2], TEAMS, {2: 1})
    assert [(e.type, e.detail) for e in events] == [(REBOUND, "offensive")]
    events = detector.detect([1, 1, -1, -1, 1, 1], TEAMS, {2: 1})  # shooter got his own miss
    assert [(e.type, e.detail, e.to_player) for e in events] == [(REBOUND, "offensive", 1)]
    # release slightly before possession officially ended still counts
    events = detector.detect([1, 1, 1, 1, -1, 3], TEAMS, {2: 1})
    assert events[0].type == REBOUND


def test_no_event_for_unknown_team_or_same_player():
    """TC-EVT-03: changes involving team 0 and regains by the same player without a shot create no event."""
    detector = PassInterceptionDetector()
    assert detector.detect([1, 1, 9, 9], TEAMS) == []
    assert detector.detect([1, -1, -1, 1], TEAMS) == []
    assert detector.detect([], TEAMS) == []


def test_cumulative_event_counts():
    """TC-EVT-04: running pass and interception counts per team for the overlay."""
    events = PassInterceptionDetector().detect([1, 2, 3, 4], TEAMS)
    counts = cumulative_event_counts(events, 4)
    assert counts[0][PASS] == {1: 0, 2: 0}
    assert counts[1][PASS] == {1: 1, 2: 0}
    assert counts[3] == {PASS: {1: 1, 2: 1}, INTERCEPTION: {1: 0, 2: 1}}
