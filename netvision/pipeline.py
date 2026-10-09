"""The NetVision pipeline: analyze pass, analysis stage, render pass, export."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import NetVisionConfig, pixel_scale
from .court import CourtGeometry, TacticalViewConverter
from .detection import (
    Detector,
    KeypointSource,
    StaticCourtCalibration,
    YoloDetector,
    YoloKeypointDetector,
    resolve_weights,
)
from .drawers import Drawer, default_drawers
from .events import PassInterceptionDetector
from .metrics import (
    ShotAnalyzer,
    SpeedDistanceCalculator,
    build_stats,
    write_events_csv,
    write_json,
    write_tracking_csv,
)
from .possession import BallAcquisitionDetector, team_possession
from .results import AnalysisResult, RawDetections
from .team_assigner import ColorJerseyClassifier, JerseyClassifier, TeamAssigner, build_jersey_classifier
from .trackers import BallTracker, PlayerTracker, RimDetector, interpolate_positions, remove_wrong_detections
from .utils import VideoInfo, create_writer, get_video_info, iter_batches, iter_frames, read_stub, save_stub, stub_key

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutputs:
    result: AnalysisResult
    video_path: Optional[Path]
    stats_path: Path
    tracking_csv_path: Path
    events_csv_path: Path
    timings: dict[str, float]


class NetVisionPipeline:
    def __init__(
        self,
        config: Optional[NetVisionConfig] = None,
        player_detector: Optional[Detector] = None,
        ball_detector: Optional[Detector] = None,
        rim_detector: Optional[Detector] = None,
        keypoint_source: Optional[KeypointSource] = None,
        team_classifier: Optional[JerseyClassifier] = None,
        drawers: Optional[Sequence[Drawer]] = None,
        description: Optional[dict] = None,
    ):
        if player_detector is None:
            raise ValueError("A player detector is required")
        self.config = config or NetVisionConfig()
        self.player_detector = player_detector
        self.ball_detector = ball_detector or player_detector
        self.rim_detector = rim_detector
        self.keypoint_source = keypoint_source
        self.team_classifier = team_classifier or ColorJerseyClassifier(
            self.config.team.outlier_factor, self.config.team.outlier_min_distance
        )
        self.geometry = CourtGeometry(length=self.config.court.length_m, width=self.config.court.width_m)
        self.drawers = list(drawers) if drawers is not None else default_drawers(
            self.config.output.draw_keypoints, self.config.output.draw_tactical_view, self.geometry,
            self.config.output.tactical_view_width_px,
        )
        self.description = description or {}

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, config: NetVisionConfig) -> "NetVisionPipeline":
        models = config.models
        min_conf = min(models.player_confidence, models.ball_confidence, models.rim_confidence)
        cache: dict[str, YoloDetector] = {}

        def detector_for(weights: str) -> YoloDetector:
            if weights not in cache:
                logger.info("Loading detector %s", weights)
                cache[weights] = YoloDetector(weights, confidence=min_conf, device=models.device)
            return cache[weights]

        player_weights = resolve_weights(models.player_model, models.fallback_model)
        ball_weights = resolve_weights(models.ball_model, player_weights)
        player_detector = detector_for(player_weights)
        ball_detector = detector_for(ball_weights)

        rim_detector = None
        rim_weights = resolve_weights(models.rim_model, None)
        if rim_weights:
            rim_detector = detector_for(rim_weights)
        else:
            for detector in (player_detector, ball_detector):
                if detector.has_class(models.rim_class_names):
                    rim_detector = detector
                    break

        keypoint_source = None
        keypoint_name = None
        if models.court_calibration:
            keypoint_source = StaticCourtCalibration.from_json(models.court_calibration)
            keypoint_name = f"calibration:{Path(models.court_calibration).name}"
        else:
            keypoint_weights = resolve_weights(models.court_keypoint_model, None)
            if keypoint_weights:
                keypoint_source = YoloKeypointDetector(keypoint_weights, models.keypoint_confidence, models.device)
                keypoint_name = keypoint_weights

        team = config.team
        classifier = build_jersey_classifier(
            team.classifier, team.clip_model, team.team1_label, team.team2_label, models.device,
            team.outlier_factor, team.outlier_min_distance,
        )
        description = {
            "player_model": player_weights,
            "ball_model": ball_weights,
            "rim_model": rim_detector.weights if rim_detector else None,
            "court_keypoints": keypoint_name,
            "team_classifier": type(classifier).__name__,
        }
        return cls(config, player_detector, ball_detector, rim_detector, keypoint_source, classifier, description=description)

    # ------------------------------------------------------------------ pass 1
    def detect(self, video_path: str | Path, info: Optional[VideoInfo] = None) -> RawDetections:
        info = info or get_video_info(video_path)
        cfg = self.config
        stub_path = None
        if cfg.output.use_stubs:
            stub_path = Path(cfg.output.stub_dir) / stub_key(video_path, sorted(self.description.items()), cfg.max_frames)
            cached = read_stub(stub_path)
            if isinstance(cached, RawDetections):
                logger.info("Loaded detections from stub %s", stub_path)
                return cached

        player_tracker = PlayerTracker(
            cfg.models.player_class_names, cfg.models.player_confidence, info.fps,
            cfg.tracking.lost_track_buffer, cfg.tracking.track_activation_threshold,
            cfg.tracking.minimum_matching_threshold, cfg.tracking.tracker,
        )
        ball_selector = BallTracker(cfg.models.ball_class_names, cfg.models.ball_confidence)
        rim_selector = RimDetector(cfg.models.rim_class_names, cfg.models.rim_confidence)
        team_assigner = TeamAssigner(self.team_classifier, cfg.team.sample_every_n_frames, cfg.team.max_samples_per_track)

        player_tracks, ball_positions = [], []
        rim_positions = [] if self.rim_detector is not None else None
        keypoints = [] if self.keypoint_source is not None else None
        frame_index = 0
        for batch in iter_batches(video_path, cfg.models.batch_size, cfg.max_frames):
            results: dict[int, list] = {}
            for detector in (self.player_detector, self.ball_detector, self.rim_detector):
                if detector is not None and id(detector) not in results:
                    results[id(detector)] = detector.detect(batch)
            batch_keypoints = self.keypoint_source.detect(batch) if self.keypoint_source is not None else None
            for offset, frame in enumerate(batch):
                players = player_tracker.update(results[id(self.player_detector)][offset])
                player_tracks.append(players)
                ball_positions.append(ball_selector.select(results[id(self.ball_detector)][offset]))
                if rim_positions is not None:
                    rim_positions.append(rim_selector.select(results[id(self.rim_detector)][offset]))
                if keypoints is not None:
                    keypoints.append(batch_keypoints[offset])
                team_assigner.observe(frame, frame_index, players)
                frame_index += 1
        raw = RawDetections(player_tracks, ball_positions, rim_positions, keypoints, team_assigner.state())
        if stub_path is not None:
            save_stub(stub_path, raw)
        return raw

    # ------------------------------------------------------------------ analysis
    def analyze(self, raw: RawDetections, info: VideoInfo) -> AnalysisResult:
        cfg = self.config
        scale = pixel_scale(info.height)
        fps = info.fps or 30.0
        notes: list[str] = []

        ball = remove_wrong_detections(raw.ball_positions, cfg.tracking.ball_max_px_per_frame * scale)
        ball = interpolate_positions(ball, cfg.tracking.ball_max_interpolation_gap)

        team_assigner = TeamAssigner(self.team_classifier)
        team_assigner.load_state(raw.team_crops)
        teams = team_assigner.finalize()

        homographies = court_positions = distances = speeds = None
        if raw.keypoints is not None:
            converter = TacticalViewConverter(
                self.geometry, cfg.court.keypoint_tolerance, cfg.court.min_keypoints,
                cfg.court.homography_carry_frames, cfg.court.off_court_margin_m,
            )
            homographies = converter.build_homographies(raw.keypoints)
            mapped = sum(h is not None for h in homographies)
            if mapped == 0:
                notes.append("Court keypoints were found in no frame, so metric stats are unavailable.")
            else:
                court_positions = converter.players_to_court(raw.player_tracks, homographies)
                calculator = SpeedDistanceCalculator(
                    fps, cfg.metrics.speed_window_frames, cfg.metrics.max_speed_kmh, cfg.metrics.position_smoothing_frames
                )
                distances, speeds = calculator.calculate(court_positions)
                if mapped < len(homographies):
                    notes.append(f"Court mapping available in {mapped} of {len(homographies)} frames.")
        else:
            notes.append("No court keypoint model or calibration file: tactical view, speed, distance and contest distance are off.")
        if raw.rim_positions is None:
            notes.append("No rim detector: shot results are reported as unknown.")

        # Possession candidates: players on a team, and on the court when a
        # court mapping exists for that frame (drops fans and the bench).
        on_court_teams = {pid for pid, team in teams.items() if team in (1, 2)}
        possession_players = []
        for frame_index, frame in enumerate(raw.player_tracks):
            candidates = {p: b for p, b in frame.items() if not on_court_teams or p in on_court_teams}
            if homographies is not None and homographies[frame_index] is not None and court_positions is not None:
                candidates = {p: b for p, b in candidates.items() if p in court_positions[frame_index]}
            possession_players.append(candidates)
        acquisition = BallAcquisitionDetector(
            cfg.possession.possession_threshold_px * scale, cfg.possession.min_possession_frames,
            cfg.possession.containment_threshold, cfg.possession.backfill_confirmed,
        )
        possession = acquisition.detect_ball_possession(possession_players, ball)

        shot_analyzer = ShotAnalyzer(
            fps, cfg.metrics.min_flight_frames, cfg.metrics.max_flight_seconds, cfg.metrics.min_rise_ratio,
            cfg.metrics.player_height_m, cfg.metrics.release_fit_frames, self.geometry, cfg.metrics.min_descent_ratio,
        )
        shots = shot_analyzer.analyze(possession, raw.player_tracks, ball, teams, court_positions, raw.rim_positions)
        events = PassInterceptionDetector().detect(possession, teams, {s.release_frame: s.shooter_id for s in shots})

        result = AnalysisResult(
            video=info,
            player_tracks=raw.player_tracks,
            ball_positions=ball,
            raw_ball_positions=raw.ball_positions,
            rim_positions=raw.rim_positions,
            keypoints=raw.keypoints,
            team_assignment=teams,
            possession=possession,
            team_possession=team_possession(possession, teams),
            events=events,
            shots=shots,
            homographies=homographies,
            court_positions=court_positions,
            distances=distances,
            speeds=speeds,
            notes=notes,
        )
        result.stats = build_stats(result, {"models": self.description})
        return result

    # ------------------------------------------------------------------ pass 2
    def render(self, video_path: str | Path, result: AnalysisResult, output_path: str | Path) -> Path:
        info = result.video
        for drawer in self.drawers:
            drawer.prepare(result)
        writer = create_writer(output_path, info.fps or 30.0, info.width, info.height)
        try:
            for index, frame in enumerate(iter_frames(video_path, result.num_frames)):
                for drawer in self.drawers:
                    frame = drawer.draw(frame, index, result)
                writer.write(frame)
        finally:
            writer.release()
        return Path(output_path)

    # ------------------------------------------------------------------ all together
    def run(self, video_path: str | Path, output_dir: Optional[str | Path] = None) -> PipelineOutputs:
        video_path = Path(video_path)
        output_dir = Path(output_dir or self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}

        start = time.perf_counter()
        info = get_video_info(video_path)
        if self.config.max_frames is not None:
            info = VideoInfo(info.fps, info.width, info.height, min(info.frame_count, self.config.max_frames))
        raw = self.detect(video_path, info)
        info = VideoInfo(info.fps, info.width, info.height, len(raw))
        timings["detect_s"] = time.perf_counter() - start

        start = time.perf_counter()
        result = self.analyze(raw, info)
        timings["analyze_s"] = time.perf_counter() - start

        video_out = None
        if self.config.output.write_video:
            start = time.perf_counter()
            video_out = self.render(video_path, result, output_dir / f"{video_path.stem}_annotated{self.config.output.video_extension}")
            timings["render_s"] = time.perf_counter() - start

        result.stats["metadata"]["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
        stats_path = write_json(result.stats, output_dir / f"{video_path.stem}_stats.json")
        tracking_path = write_tracking_csv(result, output_dir / f"{video_path.stem}_tracking.csv")
        events_path = write_events_csv(result, output_dir / f"{video_path.stem}_events.csv")
        return PipelineOutputs(result, video_out, stats_path, tracking_path, events_path, timings)
