"""Aggregate statistics and export them as JSON and CSV."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..events import INTERCEPTION, PASS, REBOUND
from ..possession import NO_PLAYER, possession_percentages
from .speed_distance import summarize_movement

if TYPE_CHECKING:
    from ..results import AnalysisResult

SCHEMA_VERSION = "1.0"


def build_stats(result: "AnalysisResult", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    fps = result.video.fps or 30.0
    percentages = possession_percentages(result.team_possession)
    movement = summarize_movement(result.distances, result.speeds) if result.distances is not None else {}

    teams: dict[str, dict[str, Any]] = {}
    for team in (1, 2):
        team_shots = [s for s in result.shots if s.team == team]
        angles = [s.release_angle_deg for s in team_shots]
        contests = [s.contest_distance_m for s in team_shots if s.contest_distance_m is not None]
        teams[str(team)] = {
            "possession_pct": round(percentages[team], 2),
            "possession_frames": sum(1 for t in result.team_possession if t == team),
            "passes": sum(1 for e in result.events if e.type == PASS and e.team == team),
            "interceptions": sum(1 for e in result.events if e.type == INTERCEPTION and e.team == team),
            "rebounds": sum(1 for e in result.events if e.type == REBOUND and e.team == team),
            "shots": len(team_shots),
            "shots_made": sum(1 for s in team_shots if s.result == "made"),
            "avg_release_angle_deg": _mean(angles),
            "avg_contest_distance_m": _mean(contests),
            "players": sorted(p for p, t in result.team_assignment.items() if t == team),
        }

    frames_tracked = Counter(pid for frame in result.player_tracks for pid in frame)
    possession_frames = Counter(p for p in result.possession if p != NO_PLAYER)
    players: dict[str, dict[str, Any]] = {}
    for player_id in sorted(frames_tracked):
        moves = movement.get(player_id, {})
        players[str(player_id)] = {
            "team": result.team_assignment.get(player_id, 0),
            "frames_tracked": frames_tracked[player_id],
            "possession_frames": possession_frames.get(player_id, 0),
            "possession_s": round(possession_frames.get(player_id, 0) / fps, 2),
            "passes_made": sum(1 for e in result.events if e.type == PASS and e.from_player == player_id),
            "passes_received": sum(1 for e in result.events if e.type == PASS and e.to_player == player_id),
            "interceptions": sum(1 for e in result.events if e.type == INTERCEPTION and e.to_player == player_id),
            "turnovers": sum(1 for e in result.events if e.type == INTERCEPTION and e.from_player == player_id),
            "rebounds": sum(1 for e in result.events if e.type == REBOUND and e.to_player == player_id),
            "shots": sum(1 for s in result.shots if s.shooter_id == player_id),
            "distance_m": _round(moves.get("distance_m")),
            "max_speed_kmh": _round(moves.get("max_speed_kmh")),
            "avg_speed_kmh": _round(moves.get("avg_speed_kmh")),
        }

    ball_frames = sum(1 for b in result.ball_positions if b is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "fps": fps,
            "frames": result.num_frames,
            "duration_s": round(result.num_frames / fps, 2),
            "width": result.video.width,
            "height": result.video.height,
            "ball_visible_pct": round(100.0 * ball_frames / max(1, result.num_frames), 2),
            "court_mapping": result.has_court_mapping,
            "notes": list(result.notes),
            **(metadata or {}),
        },
        "teams": teams,
        "players": players,
        "events": [e.to_dict() for e in result.events],
        "shots": [s.to_dict() for s in result.shots],
    }


def write_json(stats: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=_json_default)
    return path


TRACKING_COLUMNS = [
    "frame", "time_s", "track_id", "team", "x1", "y1", "x2", "y2",
    "court_x_m", "court_y_m", "speed_kmh", "distance_m", "has_ball",
]


def write_tracking_csv(result: "AnalysisResult", path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = result.video.fps or 30.0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACKING_COLUMNS)
        for frame, players in enumerate(result.player_tracks):
            court = result.court_positions[frame] if result.court_positions else {}
            speeds = result.speeds[frame] if result.speeds else {}
            distances = result.distances[frame] if result.distances else {}
            for track_id in sorted(players):
                x1, y1, x2, y2 = players[track_id]
                cx, cy = court.get(track_id, (None, None))
                writer.writerow([
                    frame, round(frame / fps, 3), track_id, result.team_assignment.get(track_id, 0),
                    round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                    _round(cx), _round(cy), _round(speeds.get(track_id)), _round(distances.get(track_id)),
                    int(result.possession[frame] == track_id),
                ])
    return path


EVENT_COLUMNS = ["frame", "time_s", "type", "team", "from_player", "to_player", "detail"]


def write_events_csv(result: "AnalysisResult", path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = result.video.fps or 30.0
    rows = [
        [e.frame, round(e.frame / fps, 3), e.type, e.team, e.from_player, e.to_player, e.detail]
        for e in result.events
    ]
    for shot in result.shots:
        detail = f"angle={shot.release_angle_deg};apex_m={shot.apex_height_m};contest_m={shot.contest_distance_m};result={shot.result}"
        rows.append([shot.release_frame, round(shot.release_frame / fps, 3), "shot", shot.team, shot.shooter_id, None, detail])
    rows.sort(key=lambda row: row[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVENT_COLUMNS)
        writer.writerows(rows)
    return path


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


def _round(value, digits: int = 3):
    return None if value is None else round(float(value), digits)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")
