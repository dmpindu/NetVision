"""Video overlays.

Every drawer has ``draw(frame, frame_index, result)`` and returns the frame.
Drawers only read the analysis result, so they work on a streamed video.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from ..court.geometry import CourtGeometry
from ..court.rendering import CourtCanvas, render_court
from ..events import INTERCEPTION, PASS, cumulative_event_counts
from ..possession import NO_PLAYER
from ..utils.bbox import get_center
from .utils import draw_ellipse, draw_panel, draw_triangle, put_text

if TYPE_CHECKING:
    from ..results import AnalysisResult

# BGR colors
TEAM_COLORS = {0: (160, 160, 160), 1: (255, 245, 238), 2: (128, 0, 0)}
HOLDER_COLOR = (0, 0, 255)
BALL_COLOR = (0, 255, 0)


class Drawer:
    def prepare(self, result: "AnalysisResult") -> None:
        """Optional precomputation before the render pass."""

    def draw(self, frame: np.ndarray, frame_index: int, result: "AnalysisResult") -> np.ndarray:
        raise NotImplementedError


class PlayerTracksDrawer(Drawer):
    def __init__(self, team_colors: Optional[dict[int, tuple]] = None):
        self.team_colors = team_colors or TEAM_COLORS

    def draw(self, frame, frame_index, result):
        holder = result.possession[frame_index]
        for track_id, bbox in result.player_tracks[frame_index].items():
            team = result.team_assignment.get(track_id, 0)
            draw_ellipse(frame, bbox, self.team_colors.get(team, TEAM_COLORS[0]), track_id)
            if track_id == holder:
                draw_triangle(frame, bbox, HOLDER_COLOR)
        return frame


class BallTracksDrawer(Drawer):
    def draw(self, frame, frame_index, result):
        ball = result.ball_positions[frame_index]
        if ball is not None:
            draw_triangle(frame, ball, BALL_COLOR)
        return frame


class TeamBallControlDrawer(Drawer):
    """Running share of frames each team has controlled the ball."""

    def prepare(self, result):
        team1 = team2 = 0
        self.percentages = []
        for team in result.team_possession:
            team1 += team == 1
            team2 += team == 2
            total = team1 + team2
            self.percentages.append((100 * team1 / total, 100 * team2 / total) if total else (0.0, 0.0))

    def draw(self, frame, frame_index, result):
        height, width = frame.shape[:2]
        x1, y1 = int(width * 0.60), int(height * 0.75)
        x2, y2 = int(width * 0.99), int(height * 0.90)
        draw_panel(frame, (x1, y1), (x2, y2))
        team1, team2 = self.percentages[frame_index]
        scale = height / 720
        put_text(frame, f"Team 1 Ball Control: {team1:.1f}%", (x1 + 10, y1 + 35 * scale), 0.7 * scale)
        put_text(frame, f"Team 2 Ball Control: {team2:.1f}%", (x1 + 10, y1 + 75 * scale), 0.7 * scale)
        return frame


class PassInterceptionDrawer(Drawer):
    def prepare(self, result):
        self.counts = cumulative_event_counts(result.events, result.num_frames, (PASS, INTERCEPTION))

    def draw(self, frame, frame_index, result):
        height, width = frame.shape[:2]
        x1, y1 = int(width * 0.16), int(height * 0.75)
        x2, y2 = int(width * 0.55), int(height * 0.90)
        draw_panel(frame, (x1, y1), (x2, y2))
        counts = self.counts[frame_index]
        scale = height / 720
        put_text(frame, f"Team 1 - Passes: {counts[PASS][1]}  Interceptions: {counts[INTERCEPTION][1]}",
                 (x1 + 10, y1 + 35 * scale), 0.6 * scale)
        put_text(frame, f"Team 2 - Passes: {counts[PASS][2]}  Interceptions: {counts[INTERCEPTION][2]}",
                 (x1 + 10, y1 + 75 * scale), 0.6 * scale)
        return frame


class CourtKeypointsDrawer(Drawer):
    def draw(self, frame, frame_index, result):
        if not result.keypoints:
            return frame
        points = result.keypoints[frame_index]
        if points is None:
            return frame
        for index, (x, y) in enumerate(points):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cv2.circle(frame, (int(x), int(y)), 5, (255, 20, 147), -1)
            put_text(frame, str(index), (x + 6, y - 6), 0.45, (255, 255, 255), 1)
        return frame


class SpeedDistanceDrawer(Drawer):
    def draw(self, frame, frame_index, result):
        if result.speeds is None:
            return frame
        speeds = result.speeds[frame_index]
        distances = result.distances[frame_index]
        for track_id, bbox in result.player_tracks[frame_index].items():
            if track_id not in speeds:
                continue
            x = int((bbox[0] + bbox[2]) / 2) - 35
            y = int(bbox[3]) + 45
            put_text(frame, f"{speeds[track_id]:.1f} km/h", (x, y), 0.45, (0, 0, 0), 2)
            put_text(frame, f"{distances[track_id]:.1f} m", (x, y + 18), 0.45, (0, 0, 0), 2)
        return frame


class TacticalViewDrawer(Drawer):
    """Top-down mini map of the court in the top-left corner."""

    def __init__(self, geometry: Optional[CourtGeometry] = None, width_px: int = 300, alpha: float = 0.8):
        self.geometry = geometry or CourtGeometry()
        self.canvas = CourtCanvas(self.geometry, px_per_m=width_px / self.geometry.length, margin_px=8)
        self.court_image = render_court(self.canvas, line_thickness=1)
        self.alpha = alpha

    def draw(self, frame, frame_index, result):
        if not result.has_court_mapping:
            return frame
        court = self.court_image.copy()
        holder = result.possession[frame_index]
        for track_id, position in result.court_positions[frame_index].items():
            team = result.team_assignment.get(track_id, 0)
            x, y = self.canvas.to_pixels([position])[0].astype(int)
            cv2.circle(court, (x, y), 5, TEAM_COLORS.get(team, TEAM_COLORS[0]), -1)
            cv2.circle(court, (x, y), 5, (0, 0, 0), 1)
            if track_id == holder:
                cv2.circle(court, (x, y), 8, HOLDER_COLOR, 2)
        height, width = court.shape[:2]
        x0, y0 = 20, 40
        if y0 + height > frame.shape[0] or x0 + width > frame.shape[1]:
            return frame
        region = frame[y0 : y0 + height, x0 : x0 + width]
        cv2.addWeighted(court, self.alpha, region, 1 - self.alpha, 0, dst=region)
        return frame


class ShotDrawer(Drawer):
    """Ball trail during a shot and a summary label for two seconds afterwards."""

    def __init__(self, label_seconds: float = 2.0):
        self.label_seconds = label_seconds

    def draw(self, frame, frame_index, result):
        hold = int(self.label_seconds * (result.video.fps or 30))
        for shot in result.shots:
            if not shot.release_frame <= frame_index <= shot.end_frame + hold:
                continue
            trail = [(int(x), int(y)) for f, x, y in shot.trajectory if f <= frame_index]
            if len(trail) >= 2:
                cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
            parts = [f"SHOT #{shot.shooter_id}", f"angle {shot.release_angle_deg:.0f} deg", f"arc {shot.apex_height_m:.1f} m"]
            if shot.contest_distance_m is not None:
                parts.append(f"contest {shot.contest_distance_m:.1f} m")
            if shot.result != "unknown":
                parts.append(shot.result.upper())
            text = " | ".join(parts)
            width = frame.shape[1]
            draw_panel(frame, (width * 0.35, 10), (width * 0.99, 45))
            put_text(frame, text, (width * 0.35 + 10, 35), 0.6)
        return frame


class FrameNumberDrawer(Drawer):
    def draw(self, frame, frame_index, result):
        put_text(frame, f"Frame: {frame_index}", (10, 25), 0.7, (0, 0, 0), 2)
        return frame


def default_drawers(draw_keypoints: bool = True, draw_tactical_view: bool = True,
                    geometry: Optional[CourtGeometry] = None, tactical_width_px: int = 300) -> list[Drawer]:
    drawers: list[Drawer] = [PlayerTracksDrawer(), BallTracksDrawer()]
    if draw_keypoints:
        drawers.append(CourtKeypointsDrawer())
    drawers += [TeamBallControlDrawer(), PassInterceptionDrawer(), SpeedDistanceDrawer(), ShotDrawer()]
    if draw_tactical_view:
        drawers.append(TacticalViewDrawer(geometry, tactical_width_px))
    drawers.append(FrameNumberDrawer())
    return drawers
