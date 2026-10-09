"""Synthetic basketball clip with exact ground truth.

Real broadcast footage has no labels for passes, shots, or distances, so the
end to end tests use this generator instead. It renders a court seen from a
broadcast-like angle (a known homography), six players in two jersey colors,
a referee, and a ball, and it scripts a short possession:

====== =========================================================
frames what happens
====== =========================================================
0-59   A (team 1) dribbles
60-71  A passes to B                         -> pass, team 1
72-129 B holds
130-141 B passes toward C, D intercepts     -> interception, team 2
142-189 D holds
190-201 D passes to E                        -> pass, team 2
202-239 E holds, then shoots
240-275 shot in the air (parabola)           -> shot, team 2
276-285 ball drops to A                      -> defensive rebound, team 1
286-329 A holds
====== =========================================================

Everything the pipeline should measure is recorded in ``ground_truth``.
Run ``python -m netvision.testing.synthetic out.mp4`` to write a clip.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..court.geometry import CourtGeometry
from ..court.rendering import CourtCanvas, render_court
from ..detection import Detection

WIDTH, HEIGHT, FPS = 1280, 720, 30.0
NUM_FRAMES = 330
PLAYER_HEIGHT_M = 2.0

# Court corners in meters -> image pixels (broadcast-style trapezoid)
_COURT_CORNERS_M = np.float32([[0, 0], [28, 0], [28, 15], [0, 15]])
_COURT_CORNERS_PX = np.float32([[140, 250], [1140, 250], [1265, 690], [15, 690]])

TEAM_JERSEYS = {1: (240, 240, 240), 2: (110, 40, 20), 0: (128, 128, 128)}  # BGR
FLOOR = (110, 170, 215)
STANDS = (50, 50, 50)
SKIN = (90, 130, 190)
BALL = (20, 110, 230)


@dataclass
class ScriptedPlayer:
    pid: int
    team: int
    waypoints: list[tuple[int, float, float]]  # (frame, x_m, y_m)

    def position(self, frame: int) -> tuple[float, float]:
        points = self.waypoints
        if frame <= points[0][0]:
            return points[0][1], points[0][2]
        for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
            if f0 <= frame <= f1:
                t = (frame - f0) / (f1 - f0) if f1 != f0 else 0.0
                return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        return points[-1][1], points[-1][2]

    def path_length(self, num_frames: int) -> float:
        total = 0.0
        previous = self.position(0)
        for frame in range(1, num_frames):
            current = self.position(frame)
            total += math.dist(previous, current)
            previous = current
        return total


def default_players() -> list[ScriptedPlayer]:
    return [
        ScriptedPlayer(1, 1, [(0, 15.0, 8.0), (60, 17.0, 9.0), (140, 17.5, 9.0), (255, 17.5, 9.0), (284, 24.5, 9.5), (329, 24.5, 9.5)]),  # A
        ScriptedPlayer(2, 1, [(0, 18.0, 4.0), (60, 19.0, 3.5), (130, 20.0, 4.0), (329, 20.5, 4.5)]),  # B
        ScriptedPlayer(3, 1, [(0, 20.0, 12.0), (130, 22.0, 12.0), (329, 22.0, 12.0)]),  # C
        ScriptedPlayer(4, 2, [(0, 16.0, 6.0), (141, 21.0, 8.0), (190, 22.0, 8.0), (240, 19.0, 6.0), (329, 19.0, 6.0)]),  # D
        ScriptedPlayer(5, 2, [(0, 19.0, 11.0), (200, 21.0, 10.3), (329, 21.0, 10.3)]),  # E
        ScriptedPlayer(6, 2, [(0, 24.0, 3.0), (329, 25.0, 2.5)]),  # F
        ScriptedPlayer(7, 0, [(0, 14.0, 14.4), (329, 19.0, 14.4)]),  # referee
    ]


@dataclass
class Segment:
    kind: str  # "hold", "pass", "shot", "drop"
    start: int
    end: int  # inclusive
    holder: Optional[int] = None
    target: Optional[int] = None


DEFAULT_SCRIPT = [
    Segment("hold", 0, 59, holder=1),
    Segment("pass", 60, 71, holder=1, target=2),
    Segment("hold", 72, 129, holder=2),
    Segment("pass", 130, 141, holder=2, target=4),
    Segment("hold", 142, 189, holder=4),
    Segment("pass", 190, 201, holder=4, target=5),
    Segment("hold", 202, 239, holder=5),
    Segment("shot", 240, 275, holder=5),
    Segment("drop", 276, 285, holder=5, target=1),
    Segment("hold", 286, 329, holder=1),
]

SHOT_VX = 8.0  # px per frame
SHOT_ANGLE_DEG = 50.0
SHOT_APEX_FRAMES = 18


@dataclass
class SyntheticClip:
    path: Path
    fps: float
    width: int
    height: int
    num_frames: int
    homography: np.ndarray  # court meters -> image pixels
    player_boxes: list[dict[int, list[float]]]
    ball_boxes: list[Optional[list[float]]]
    holders: list[int]
    keypoints: np.ndarray
    teams: dict[int, int]
    ground_truth: dict = field(default_factory=dict)


class SceneRenderer:
    def __init__(self, geometry: Optional[CourtGeometry] = None):
        self.geometry = geometry or CourtGeometry()
        self.homography = cv2.getPerspectiveTransform(_COURT_CORNERS_M, _COURT_CORNERS_PX)
        self.background = self._render_background()

    def project(self, points_m) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.homography).reshape(-1, 2)

    def _render_background(self) -> np.ndarray:
        canvas = CourtCanvas(self.geometry, px_per_m=40, margin_px=0)
        court = render_court(canvas, floor_color=FLOOR, line_color=(255, 255, 255), outside_color=FLOOR, line_thickness=4)
        scale = np.array([[1 / canvas.px_per_m, 0, 0], [0, 1 / canvas.px_per_m, 0], [0, 0, 1]])
        frame = np.full((HEIGHT, WIDTH, 3), STANDS, dtype=np.uint8)
        cv2.warpPerspective(court, self.homography @ scale, (WIDTH, HEIGHT), dst=frame,
                            borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_LINEAR)
        return frame

    def player_box(self, position_m) -> list[float]:
        x, y = position_m
        foot = self.project([(x, y)])[0]
        left, right = self.project([(x - 0.5, y), (x + 0.5, y)])
        px_per_m = float(np.linalg.norm(right - left))
        height = PLAYER_HEIGHT_M * px_per_m
        width = 0.45 * height
        return [float(foot[0] - width / 2), float(foot[1] - height), float(foot[0] + width / 2), float(foot[1])]

    @staticmethod
    def hand(box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return x2 - 0.15 * (x2 - x1), y1 + 0.5 * (y2 - y1)

    @staticmethod
    def ball_radius(box) -> float:
        return max(5.0, 0.07 * (box[3] - box[1]))

    def draw_player(self, frame, box, team) -> None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        jersey = TEAM_JERSEYS[team]
        shorts = tuple(int(c * 0.8) for c in jersey)
        cx = (x1 + x2) / 2
        cv2.rectangle(frame, (int(cx - 0.3 * w), int(y1 + 0.7 * h)), (int(cx - 0.08 * w), int(y2)), SKIN, -1)
        cv2.rectangle(frame, (int(cx + 0.08 * w), int(y1 + 0.7 * h)), (int(cx + 0.3 * w), int(y2)), SKIN, -1)
        cv2.rectangle(frame, (int(x1 + 0.12 * w), int(y1 + 0.55 * h)), (int(x2 - 0.12 * w), int(y1 + 0.72 * h)), shorts, -1)
        cv2.rectangle(frame, (int(x1 + 0.1 * w), int(y1 + 0.2 * h)), (int(x2 - 0.1 * w), int(y1 + 0.56 * h)), jersey, -1)
        cv2.circle(frame, (int(cx), int(y1 + 0.1 * h)), int(0.09 * h), SKIN, -1)

    @staticmethod
    def draw_ball(frame, center, radius) -> None:
        cv2.circle(frame, (int(round(center[0])), int(round(center[1]))), int(round(radius)), BALL, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(round(center[0])), int(round(center[1]))), int(round(radius)), (0, 0, 0), 1, cv2.LINE_AA)


def _ball_box(center, radius) -> list[float]:
    return [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]


def generate_clip(path: str | Path, players: Optional[Sequence[ScriptedPlayer]] = None,
                  script: Optional[Sequence[Segment]] = None, write_video: bool = True) -> SyntheticClip:
    players = list(players or default_players())
    script = list(script or DEFAULT_SCRIPT)
    renderer = SceneRenderer()
    by_id = {p.pid: p for p in players}
    num_frames = script[-1].end + 1

    boxes = [{p.pid: renderer.player_box(p.position(f)) for p in players} for f in range(num_frames)]
    ball_boxes: list[Optional[list[float]]] = [None] * num_frames
    holders = [-1] * num_frames
    shot_truth = None

    for segment in script:
        if segment.kind == "hold":
            for f in range(segment.start, segment.end + 1):
                box = boxes[f][segment.holder]
                hx, hy = renderer.hand(box)
                height = box[3] - box[1]
                # dribble in the middle of the hold, ball in hand for the last 10 frames
                if segment.start + 5 <= f <= segment.end - 10:
                    phase = (f - segment.start) % 16 / 16
                    hy = hy + (0.4 * height) * (1 - abs(2 * phase - 1))
                ball_boxes[f] = _ball_box((hx, hy), renderer.ball_radius(box))
                holders[f] = segment.holder
        elif segment.kind in ("pass", "drop"):
            start_box = boxes[segment.start - 1][segment.holder]
            end_box = boxes[segment.end + 1][segment.target]
            start = np.array(renderer.hand(start_box)) if segment.kind == "pass" else np.array(
                ball_center(ball_boxes[segment.start - 1]))
            end = np.array(renderer.hand(end_box))
            radius = renderer.ball_radius(start_box)
            steps = segment.end - segment.start + 2
            for i, f in enumerate(range(segment.start, segment.end + 1), start=1):
                center = start + (end - start) * (i / steps)
                ball_boxes[f] = _ball_box(center, radius)
        elif segment.kind == "shot":
            release_box = boxes[segment.start - 1][segment.holder]
            x0, y0 = renderer.hand(release_box)
            vx = SHOT_VX
            vy = vx * math.tan(math.radians(SHOT_ANGLE_DEG))
            gravity = vy / SHOT_APEX_FRAMES
            radius = renderer.ball_radius(release_box)
            for t, f in enumerate(range(segment.start, segment.end + 1), start=1):
                center = (x0 + vx * t, y0 - vy * t + 0.5 * gravity * t * t)
                ball_boxes[f] = _ball_box(center, radius)
            rise_px = vy * vy / (2 * gravity)
            shooter_height = release_box[3] - release_box[1]
            shooter_pos = by_id[segment.holder].position(segment.start - 1)
            shooter_team = by_id[segment.holder].team
            opponents = [
                (p.pid, math.dist(shooter_pos, p.position(segment.start - 1)))
                for p in players if p.team in (1, 2) and p.team != shooter_team
            ]
            defender, contest = min(opponents, key=lambda item: item[1])
            shot_truth = {
                "shooter": segment.holder,
                "team": shooter_team,
                "release_frame": segment.start,
                "release_angle_deg": SHOT_ANGLE_DEG,
                "apex_height_m": rise_px / shooter_height * PLAYER_HEIGHT_M,
                "apex_frame": segment.start - 1 + SHOT_APEX_FRAMES,
                "contest_distance_m": contest,
                "closest_defender": defender,
                "shooter_court_xy": shooter_pos,
            }

    keypoints = renderer.project(renderer.geometry.keypoints)
    events = []
    for previous, segment in zip(script, script[1:]):
        if segment.kind == "hold" and previous.kind in ("pass", "drop"):
            giver, receiver = previous.holder, segment.holder
            giver_team, receiver_team = by_id[giver].team, by_id[receiver].team
            if previous.kind == "drop":
                kind = "rebound"
            else:
                kind = "pass" if giver_team == receiver_team else "interception"
            events.append({"type": kind, "from": giver, "to": receiver, "team": receiver_team,
                           "frame": segment.start})

    teams = {p.pid: p.team for p in players}
    truth = {
        "fps": FPS,
        "num_frames": num_frames,
        "teams": teams,
        "events": events,
        "shot": shot_truth,
        "distances_m": {p.pid: p.path_length(num_frames) for p in players},
        "possession_frames": {pid: holders.count(pid) for pid in teams},
    }

    if write_video:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        try:
            for f in range(num_frames):
                frame = renderer.background.copy()
                # draw far players first so near players overlap them
                for pid in sorted(boxes[f], key=lambda p: boxes[f][p][3]):
                    renderer.draw_player(frame, boxes[f][pid], teams[pid])
                if ball_boxes[f] is not None:
                    box = ball_boxes[f]
                    renderer.draw_ball(frame, ball_center(box), (box[2] - box[0]) / 2)
                writer.write(frame)
        finally:
            writer.release()

    return SyntheticClip(Path(path), FPS, WIDTH, HEIGHT, num_frames, renderer.homography, boxes, ball_boxes,
                         holders, keypoints, teams, truth)


def ball_center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class GroundTruthDetector:
    """Fake detector that replays the clip's ground truth like a YOLO model would.

    Options add realistic noise: pixel jitter, frames where the ball is
    missed, and false ball detections far away from the real ball.
    """

    def __init__(self, clip: SyntheticClip, noise_px: float = 1.0, ball_dropouts: Sequence[int] = (),
                 false_balls: Optional[dict[int, list[float]]] = None, seed: int = 0,
                 player_class: str = "player", ball_class: str = "ball"):
        self.clip = clip
        self.noise_px = noise_px
        self.ball_dropouts = set(ball_dropouts)
        self.false_balls = false_balls or {}
        self.rng = np.random.default_rng(seed)
        self.player_class = player_class
        self.ball_class = ball_class
        self.frame = 0

    def _jitter(self, box) -> tuple[float, float, float, float]:
        noise = self.rng.normal(0, self.noise_px, 4) if self.noise_px else np.zeros(4)
        return tuple(float(v) for v in np.asarray(box) + noise)

    def detect(self, frames) -> list[list[Detection]]:
        output = []
        for _ in frames:
            f = self.frame
            detections = [Detection(self._jitter(b), 0.9, self.player_class) for b in self.clip.player_boxes[f].values()]
            ball = self.clip.ball_boxes[f]
            if ball is not None and f not in self.ball_dropouts:
                detections.append(Detection(self._jitter(ball), 0.6, self.ball_class))
            if f in self.false_balls:
                detections.append(Detection(tuple(self.false_balls[f]), 0.95, self.ball_class))
            output.append(detections)
            self.frame += 1
        return output


class GroundTruthKeypoints:
    """Fake court keypoint model. Can drop frames or corrupt one keypoint."""

    def __init__(self, clip: SyntheticClip, noise_px: float = 0.5, missing_frames: Sequence[int] = (),
                 corrupt: Optional[dict[int, tuple[int, tuple[float, float]]]] = None, seed: int = 1):
        self.clip = clip
        self.noise_px = noise_px
        self.missing_frames = set(missing_frames)
        self.corrupt = corrupt or {}
        self.rng = np.random.default_rng(seed)
        self.frame = 0

    def detect(self, frames) -> list[Optional[np.ndarray]]:
        output = []
        for _ in frames:
            f = self.frame
            if f in self.missing_frames:
                output.append(None)
            else:
                points = self.clip.keypoints + self.rng.normal(0, self.noise_px, self.clip.keypoints.shape)
                if f in self.corrupt:
                    index, xy = self.corrupt[f]
                    points[index] = xy
                output.append(points)
            self.frame += 1
        return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic NetVision test clip and its ground truth")
    parser.add_argument("output", help="Output video path, e.g. synthetic.mp4")
    args = parser.parse_args(argv)
    clip = generate_clip(args.output)
    truth_path = Path(args.output).with_suffix(".truth.json")
    truth_path.write_text(json.dumps(clip.ground_truth, indent=2, default=float))
    print(f"Wrote {clip.path} ({clip.num_frames} frames) and {truth_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
