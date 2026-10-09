"""Player distance covered (meters) and speed (km/h) from court positions."""
from __future__ import annotations

from typing import Sequence

from ..utils.bbox import measure_distance

CourtPositions = dict[int, tuple[float, float]]


class SpeedDistanceCalculator:
    """Samples each player's court position every ``window_frames`` frames.

    Sampling over a short window instead of every frame keeps detection
    jitter from inflating the distance. Steps that would need a speed above
    ``max_speed_kmh`` are treated as tracking glitches and not counted.
    """

    def __init__(self, fps: float, window_frames: int = 5, max_speed_kmh: float = 40.0, smoothing_frames: int = 9):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.window_frames = max(1, window_frames)
        self.max_speed_kmh = max_speed_kmh
        self.smoothing_frames = max(1, smoothing_frames)

    def smooth(self, court_positions: Sequence[CourtPositions]) -> list[CourtPositions]:
        """Centered moving average of each player's position over ``smoothing_frames``.

        Box jitter of a pixel or two turns into centimeters of fake movement
        every frame, which adds up to meters over a clip. Averaging removes
        most of it while keeping real movement.
        """
        if self.smoothing_frames <= 1:
            return [dict(p) for p in court_positions]
        half = self.smoothing_frames // 2
        tracks: dict[int, dict[int, tuple[float, float]]] = {}
        for frame, positions in enumerate(court_positions):
            for track_id, xy in positions.items():
                tracks.setdefault(track_id, {})[frame] = xy
        smoothed: list[CourtPositions] = [dict() for _ in court_positions]
        for track_id, samples in tracks.items():
            for frame in samples:
                window = [samples[f] for f in range(frame - half, frame + half + 1) if f in samples]
                smoothed[frame][track_id] = (
                    sum(p[0] for p in window) / len(window),
                    sum(p[1] for p in window) / len(window),
                )
        return smoothed

    def calculate(self, court_positions: Sequence[CourtPositions]) -> tuple[list[dict[int, float]], list[dict[int, float]]]:
        """Return ``(distances, speeds)``: per frame, cumulative meters and current km/h for each visible player."""
        court_positions = self.smooth(court_positions)
        last_position: dict[int, tuple[float, float]] = {}
        last_frame: dict[int, int] = {}
        total: dict[int, float] = {}
        speed: dict[int, float] = {}
        distances: list[dict[int, float]] = []
        speeds: list[dict[int, float]] = []

        for frame, positions in enumerate(court_positions):
            frame_distances: dict[int, float] = {}
            frame_speeds: dict[int, float] = {}
            for track_id, position in positions.items():
                if track_id not in last_position:
                    last_position[track_id] = position
                    last_frame[track_id] = frame
                    total[track_id] = 0.0
                    speed[track_id] = 0.0
                elif frame - last_frame[track_id] >= self.window_frames:
                    step = measure_distance(last_position[track_id], position)
                    elapsed = (frame - last_frame[track_id]) / self.fps
                    step_speed = step / elapsed * 3.6
                    if step_speed <= self.max_speed_kmh:
                        total[track_id] += step
                        speed[track_id] = step_speed
                    last_position[track_id] = position
                    last_frame[track_id] = frame
                frame_distances[track_id] = total[track_id]
                frame_speeds[track_id] = speed[track_id]
            distances.append(frame_distances)
            speeds.append(frame_speeds)
        return distances, speeds


def summarize_movement(distances: Sequence[dict[int, float]], speeds: Sequence[dict[int, float]]) -> dict[int, dict[str, float]]:
    """Total distance, max speed, and average speed (over frames the player was mapped)."""
    summary: dict[int, dict[str, float]] = {}
    speed_samples: dict[int, list[float]] = {}
    for frame_distances, frame_speeds in zip(distances, speeds):
        for track_id, distance in frame_distances.items():
            entry = summary.setdefault(track_id, {"distance_m": 0.0, "max_speed_kmh": 0.0, "avg_speed_kmh": 0.0})
            entry["distance_m"] = max(entry["distance_m"], distance)
        for track_id, value in frame_speeds.items():
            speed_samples.setdefault(track_id, []).append(value)
            summary[track_id]["max_speed_kmh"] = max(summary[track_id]["max_speed_kmh"], value)
    for track_id, samples in speed_samples.items():
        moving = [s for s in samples if s > 0]
        summary[track_id]["avg_speed_kmh"] = sum(moving) / len(moving) if moving else 0.0
    return summary
