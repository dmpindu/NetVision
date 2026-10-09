"""Decide which player has the ball in each frame.

Logic from the tutorial:

1. If the ball box sits mostly inside a player's box (containment of at
   least 80 percent), that player is a strong candidate.
2. Otherwise measure the distance from the ball center to key points around
   each player's box and take the closest player within 50 px.
3. A candidate only counts as the holder after holding the ball for 11
   consecutive frames. This filters out the ball flying past or in front of
   a player, which happens a lot because the camera only sees 2D.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..utils.bbox import containment_ratio, get_center, measure_distance

NO_PLAYER = -1


class BallAcquisitionDetector:
    def __init__(
        self,
        possession_threshold: float = 50.0,
        min_frames: int = 11,
        containment_threshold: float = 0.8,
        backfill_confirmed: bool = True,
    ):
        self.possession_threshold = possession_threshold
        self.min_frames = max(1, min_frames)
        self.containment_threshold = containment_threshold
        self.backfill_confirmed = backfill_confirmed

    @staticmethod
    def get_key_basketball_player_assignment_points(
        player_bbox: Sequence[float], ball_center: Sequence[float]
    ) -> list[tuple[float, float]]:
        """Points on a player's box that the ball could be closest to.

        Includes the four corners, the four edge midpoints, and the nearest
        point on the left/right edges at the ball's height (and top/bottom
        edges at the ball's x) when the ball lines up with the box.
        """
        ball_x, ball_y = ball_center
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1
        points: list[tuple[float, float]] = []
        if y1 < ball_y < y2:
            points.append((x1, ball_y))
            points.append((x2, ball_y))
        if x1 < ball_x < x2:
            points.append((ball_x, y1))
            points.append((ball_x, y2))
        points += [
            (x1 + width / 2, y1),  # top center
            (x2, y1),  # top right
            (x1, y1),  # top left
            (x2, y1 + height / 2),  # center right
            (x1, y1 + height / 2),  # center left
            (x1 + width / 2, y1 + height / 2),  # center
            (x2, y2),  # bottom right
            (x1, y2),  # bottom left
            (x1 + width / 2, y2),  # bottom center
        ]
        return points

    def find_minimum_distance_to_ball(self, ball_center: Sequence[float], player_bbox: Sequence[float]) -> float:
        points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        return min(measure_distance(ball_center, p) for p in points)

    @staticmethod
    def calculate_ball_containment_ratio(player_bbox: Sequence[float], ball_bbox: Sequence[float]) -> float:
        return containment_ratio(ball_bbox, player_bbox)

    def find_best_candidate_for_possession(
        self, ball_bbox: Sequence[float], players: dict[int, Sequence[float]]
    ) -> int:
        if not players:
            return NO_PLAYER
        ball_center = get_center(ball_bbox)
        high_containment: list[tuple[int, float]] = []
        regular: list[tuple[int, float]] = []
        for player_id, player_bbox in players.items():
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)
            if containment > self.containment_threshold:
                high_containment.append((player_id, distance))
            else:
                regular.append((player_id, distance))

        # A contained ball wins, ties broken by distance to the box key points
        # (the center point is included, so the more central box wins).
        if high_containment:
            return min(high_containment, key=lambda item: (item[1], item[0]))[0]
        if regular:
            best_id, best_distance = min(regular, key=lambda item: (item[1], item[0]))
            if best_distance < self.possession_threshold:
                return best_id
        return NO_PLAYER

    def detect_ball_possession(
        self, player_tracks: Sequence[dict[int, Sequence[float]]], ball_positions: Sequence[Optional[Sequence[float]]]
    ) -> list[int]:
        """Return the holder's track ID for every frame, or -1 when nobody has it."""
        if len(player_tracks) != len(ball_positions):
            raise ValueError("player_tracks and ball_positions must have the same length")
        possession = [NO_PLAYER] * len(ball_positions)
        current_candidate = NO_PLAYER
        streak = 0
        for frame_index, (players, ball) in enumerate(zip(player_tracks, ball_positions)):
            candidate = NO_PLAYER
            if ball is not None:
                candidate = self.find_best_candidate_for_possession(ball, players)
            if candidate == NO_PLAYER:
                current_candidate, streak = NO_PLAYER, 0
                continue
            if candidate == current_candidate:
                streak += 1
            else:
                current_candidate, streak = candidate, 1
            if streak >= self.min_frames:
                possession[frame_index] = candidate
                if streak == self.min_frames and self.backfill_confirmed:
                    for back in range(frame_index - self.min_frames + 1, frame_index):
                        possession[back] = candidate
        return possession


def team_possession(possession: Sequence[int], team_assignment: dict[int, int]) -> list[int]:
    """Map the per-frame holder to the holder's team (0 when nobody or unknown)."""
    return [team_assignment.get(p, 0) if p != NO_PLAYER else 0 for p in possession]


def possession_percentages(team_by_frame: Sequence[int]) -> dict[int, float]:
    """Share of controlled frames for teams 1 and 2, in percent."""
    team1 = sum(1 for t in team_by_frame if t == 1)
    team2 = sum(1 for t in team_by_frame if t == 2)
    total = team1 + team2
    if total == 0:
        return {1: 0.0, 2: 0.0}
    return {1: 100.0 * team1 / total, 2: 100.0 * team2 / total}
