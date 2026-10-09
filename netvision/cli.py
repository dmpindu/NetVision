"""Command line interface.

Example::

    python -m netvision input_videos/video_1.mp4 --output-dir output
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from . import __version__
from .config import NetVisionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netvision",
        description="NetVision: basketball computer vision analytics (players, ball, possession, passes, shots).",
    )
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("-o", "--output-dir", help="Folder for the annotated video, JSON and CSV files")
    parser.add_argument("-c", "--config", help="YAML config file (see configs/default.yaml)")
    parser.add_argument("--player-model", help="Player detector weights (.pt)")
    parser.add_argument("--ball-model", help="Ball detector weights (.pt)")
    parser.add_argument("--rim-model", help="Rim/hoop detector weights (.pt)")
    parser.add_argument("--court-model", help="Court keypoint pose model weights (.pt)")
    parser.add_argument("--court-calibration", help="JSON file with manual court keypoints for a fixed camera")
    parser.add_argument("--team-classifier", choices=["auto", "clip", "color"], help="How to assign teams")
    parser.add_argument("--team1-label", help='Zero-shot label for team 1, e.g. "white shirt"')
    parser.add_argument("--team2-label", help='Zero-shot label for team 2, e.g. "dark blue shirt"')
    parser.add_argument("--device", help="cpu, cuda, mps (default: auto)")
    parser.add_argument("--max-frames", type=int, help="Only process the first N frames")
    parser.add_argument("--use-stubs", action="store_true", help="Cache detections and reuse them on reruns")
    parser.add_argument("--stub-dir", help="Folder for cached detections")
    parser.add_argument("--no-video", action="store_true", help="Skip writing the annotated video")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def config_from_args(args: argparse.Namespace) -> NetVisionConfig:
    config = NetVisionConfig.from_yaml(args.config) if args.config else NetVisionConfig()
    overrides = {
        ("models", "player_model"): args.player_model,
        ("models", "ball_model"): args.ball_model,
        ("models", "rim_model"): args.rim_model,
        ("models", "court_keypoint_model"): args.court_model,
        ("models", "court_calibration"): args.court_calibration,
        ("models", "device"): args.device,
        ("team", "classifier"): args.team_classifier,
        ("team", "team1_label"): args.team1_label,
        ("team", "team2_label"): args.team2_label,
        ("output", "output_dir"): args.output_dir,
        ("output", "stub_dir"): args.stub_dir,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            setattr(getattr(config, section), key, value)
    if args.use_stubs:
        config.output.use_stubs = True
    if args.no_video:
        config.output.write_video = False
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    return config


def print_summary(stats: dict, outputs) -> None:
    meta = stats["metadata"]
    print()
    print(f"NetVision summary  ({meta['frames']} frames, {meta['duration_s']} s)")
    print("-" * 60)
    for team, values in stats["teams"].items():
        print(
            f"Team {team}: possession {values['possession_pct']:5.1f}% | passes {values['passes']} | "
            f"interceptions {values['interceptions']} | shots {values['shots']} | players {values['players']}"
        )
    for shot in stats["shots"]:
        contest = f"{shot['contest_distance_m']:.2f} m" if shot["contest_distance_m"] is not None else "n/a"
        print(
            f"Shot by #{shot['shooter_id']} (team {shot['team']}) at frame {shot['release_frame']}: "
            f"release {shot['release_angle_deg']:.1f} deg, arc {shot['apex_height_m']:.2f} m, contest {contest}, {shot['result']}"
        )
    for note in meta.get("notes", []):
        print(f"Note: {note}")
    print("-" * 60)
    if outputs.video_path:
        print(f"Video:    {outputs.video_path}")
    print(f"Stats:    {outputs.stats_path}")
    print(f"Tracking: {outputs.tracking_csv_path}")
    print(f"Events:   {outputs.events_csv_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = config_from_args(args)
        from .pipeline import NetVisionPipeline  # heavy imports after argument parsing

        pipeline = NetVisionPipeline.from_config(config)
        outputs = pipeline.run(args.video, config.output.output_dir)
    except (FileNotFoundError, ValueError, IOError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print_summary(outputs.result.stats, outputs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
