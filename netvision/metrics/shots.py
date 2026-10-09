"""Shot detection and shot metrics.

A flight starts on the first frame after the holder loses the ball and ends
when someone gains possession (or after ``max_flight_seconds``). A flight is
a shot when:

* it lasts at least ``min_flight_frames`` frames with a visible ball,
* the ball rises above the shooter's head by at least ``min_rise_ratio`` of
  the shooter's box height, and
* the path fits a parabola that opens downward on screen and the ball
  falls back at least ``min_descent_ratio`` of its rise after the peak.

Metrics per shot:

* ``release_angle_deg``: launch angle from the start of the fitted parabola.
* ``apex_height_m``: peak height above the release point. Pixels are turned
  into meters with the shooter's box height as a ruler (default 2.0 m).
* ``flight_time_s``: time from release to the end of the arc.
* ``contest_distance_m``: court distance from the shooter to the closest
  opponent at release (needs a court mapping).
* ``distance_to_basket_m``: shooter distance to the nearest rim.
* ``result``: made / missed when a rim detector ran, otherwise unknown.

Arc and angle are measured in the image plane, so they are most accurate
when the shot travels across the screen rather than toward the camera.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..court.geometry import CourtGeometry
from ..possession import NO_PLAYER
from ..utils.bbox import get_center, get_height

MADE, MISSED, UNKNOWN = "made", "missed", "unknown"


@dataclass
class Shot:
    shooter_id: int
    team: int
    release_frame: int
    end_frame: int
    apex_frame: int
    release_angle_deg: float
    apex_height_m: float
    flight_time_s: float
    contest_distance_m: Optional[float] = None
    closest_defender_id: Optional[int] = None
    shooter_court_xy: Optional[tuple[float, float]] = None
    distance_to_basket_m: Optional[float] = None
    result: str = UNKNOWN
    trajectory: list[tuple[int, float, float]] = field(default_factory=list)

    def to_dict(self, include_trajectory: bool = False) -> dict:
        data = asdict(self)
        if not include_trajectory:
            data.pop("trajectory")
        return data


class ShotAnalyzer:
    def __init__(
        self,
        fps: float,
        min_flight_frames: int = 8,
        max_flight_seconds: float = 3.0,
        min_rise_ratio: float = 0.3,
        player_height_m: float = 2.0,
        release_fit_frames: int = 6,
        geometry: Optional[CourtGeometry] = None,
        min_descent_ratio: float = 0.25,
    ):
        self.fps = fps
        self.min_flight_frames = min_flight_frames
        self.max_flight_frames = max(1, int(round(max_flight_seconds * fps)))
        self.min_rise_ratio = min_rise_ratio
        self.player_height_m = player_height_m
        self.release_fit_frames = max(2, release_fit_frames)
        self.geometry = geometry or CourtGeometry()
        self.min_descent_ratio = min_descent_ratio

    # ------------------------------------------------------------------ public
    def analyze(
        self,
        possession: Sequence[int],
        player_tracks: Sequence[dict[int, Sequence[float]]],
        ball_positions: Sequence[Optional[Sequence[float]]],
        team_assignment: dict[int, int],
        court_positions: Optional[Sequence[dict[int, tuple[float, float]]]] = None,
        rim_positions: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> list[Shot]:
        shots: list[Shot] = []
        frame_count = len(possession)
        frame = 1
        while frame < frame_count:
            holder = possession[frame - 1]
            if holder == NO_PLAYER or possession[frame] != NO_PLAYER:
                frame += 1
                continue
            release = frame
            end = release
            while end < frame_count and possession[end] == NO_PLAYER and end - release < self.max_flight_frames:
                end += 1
            shot = self._evaluate_flight(
                holder, release, end, player_tracks, ball_positions, team_assignment, court_positions, rim_positions
            )
            if shot is not None:
                shots.append(shot)
            frame = max(end, release + 1)
        return shots

    # ------------------------------------------------------------------ internals
    def _shooter_box(self, shooter: int, release: int, player_tracks) -> Optional[Sequence[float]]:
        for offset in range(1, 6):
            index = release - offset
            if 0 <= index < len(player_tracks) and shooter in player_tracks[index]:
                return player_tracks[index][shooter]
        return None

    def _evaluate_flight(self, shooter, release, end, player_tracks, ball_positions, team_assignment,
                         court_positions, rim_positions) -> Optional[Shot]:
        points = [
            (f, *get_center(ball_positions[f])) for f in range(release, end) if ball_positions[f] is not None
        ]
        if len(points) < self.min_flight_frames:
            return None
        shooter_box = self._shooter_box(shooter, release, player_tracks)
        if shooter_box is None:
            return None
        shooter_height_px = get_height(shooter_box)
        if shooter_height_px <= 0:
            return None

        frames = np.array([p[0] for p in points], dtype=float)
        xs = np.array([p[1] for p in points], dtype=float)
        ys = np.array([p[2] for p in points], dtype=float)
        apex_index = int(np.argmin(ys))
        rise_above_head = shooter_box[1] - ys[apex_index]
        if rise_above_head < self.min_rise_ratio * shooter_height_px:
            return None
        if apex_index < 2:
            return None
        # A real shot comes back down after the peak. A pass to a player who
        # is higher on screen (further from the camera) only goes up.
        rise = ys[0] - ys[apex_index]
        descent = float(np.max(ys[apex_index:]) - ys[apex_index])
        if apex_index > len(ys) - 3 or descent < self.min_descent_ratio * rise:
            return None

        # Fit the rising part and an equal stretch of the falling part.
        fit_end = min(len(points), 2 * apex_index + 1)
        fit_x = np.polyfit(frames[:fit_end], xs[:fit_end], 1)
        fit_y = np.polyfit(frames[:fit_end], ys[:fit_end], 2)
        if fit_y[0] <= 0:  # must curve back down on screen
            return None

        # Possession ends a few frames after the ball really leaves the hand
        # (the ball is still within reach). Walk back while earlier ball
        # positions still sit on the fitted arc to find the launch point.
        tolerance = max(3.0, 0.05 * shooter_height_px)
        launch_frame = int(frames[0])
        for back in range(1, self.release_fit_frames + 3):
            frame = int(frames[0]) - back
            if frame < 0 or ball_positions[frame] is None:
                break
            bx, by = get_center(ball_positions[frame])
            if abs(np.polyval(fit_x, frame) - bx) > tolerance or abs(np.polyval(fit_y, frame) - by) > tolerance:
                break
            launch_frame = frame
        if launch_frame < int(frames[0]):
            extra = [(f, *get_center(ball_positions[f])) for f in range(launch_frame, int(frames[0]))]
            frames = np.concatenate([[p[0] for p in extra], frames])
            xs = np.concatenate([[p[1] for p in extra], xs])
            ys = np.concatenate([[p[2] for p in extra], ys])
            apex_index = int(np.argmin(ys))
            fit_end = min(len(frames), 2 * apex_index + 1)
            fit_x = np.polyfit(frames[:fit_end], xs[:fit_end], 1)
            fit_y = np.polyfit(frames[:fit_end], ys[:fit_end], 2)

        # The launch point is the last frame the ball is in the hand: the
        # first point of the arc when we walked back, else one frame before.
        origin = frames[0] if launch_frame < release else frames[0] - 1
        release_frame = int(origin) + 1
        vx = float(fit_x[0])
        vy = -float(2 * fit_y[0] * origin + fit_y[1])  # upward speed on screen, px per frame
        if vy <= 0:
            return None
        release_angle = math.degrees(math.atan2(vy, abs(vx)))

        meters_per_px = self.player_height_m / shooter_height_px
        start_y = float(np.polyval(fit_y, origin))
        apex_height_m = max(0.0, (start_y - ys[apex_index]) * meters_per_px)

        team = team_assignment.get(shooter, 0)
        shot = Shot(
            shooter_id=shooter,
            team=team,
            release_frame=release_frame,
            end_frame=int(frames[fit_end - 1]),
            apex_frame=int(frames[apex_index]),
            release_angle_deg=round(release_angle, 2),
            apex_height_m=round(float(apex_height_m), 3),
            flight_time_s=round(float(frames[fit_end - 1] - origin) / self.fps, 3),
            trajectory=[(int(f), float(x), float(y)) for f, x, y in zip(frames, xs, ys)],
        )
        release = release_frame
        points = list(zip(frames.astype(int), xs, ys))
        if court_positions is not None:
            self._add_court_metrics(shot, shooter, release, team, team_assignment, court_positions)
        if rim_positions is not None:
            shot.result = self._classify_result(points, apex_index, rim_positions)
        return shot

    def _add_court_metrics(self, shot: Shot, shooter, release, team, team_assignment, court_positions) -> None:
        for offset in (1, 2, 0, 3, 4, 5):
            index = release - offset
            if not 0 <= index < len(court_positions):
                continue
            positions = court_positions[index]
            if shooter not in positions:
                continue
            shooter_xy = positions[shooter]
            shot.shooter_court_xy = (round(shooter_xy[0], 3), round(shooter_xy[1], 3))
            shot.distance_to_basket_m = round(self.geometry.distance_to_nearest_rim(shooter_xy), 3)
            opponents = [
                (pid, math.dist(shooter_xy, xy))
                for pid, xy in positions.items()
                if pid != shooter and team_assignment.get(pid, 0) in (1, 2) and team_assignment.get(pid, 0) != team
            ]
            if opponents:
                defender, distance = min(opponents, key=lambda item: item[1])
                shot.closest_defender_id = defender
                shot.contest_distance_m = round(distance, 3)
            return

    @staticmethod
    def _classify_result(points, apex_index, rim_positions) -> str:
        saw_rim = False
        for (f0, x0, y0), (f1, x1, y1) in zip(points[apex_index:], points[apex_index + 1:]):
            rim = rim_positions[f1] if f1 < len(rim_positions) else None
            if rim is None:
                continue
            saw_rim = True
            rx1, ry1, rx2, ry2 = rim
            rim_y = (ry1 + ry2) / 2
            margin = 0.15 * (rx2 - rx1)
            crosses_down = y0 < rim_y <= y1
            if crosses_down:
                x_at_rim = x0 + (x1 - x0) * (rim_y - y0) / (y1 - y0) if y1 != y0 else x1
                if rx1 + margin <= x_at_rim <= rx2 - margin:
                    return MADE
        return MISSED if saw_rim else UNKNOWN
