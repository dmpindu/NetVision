"""Video reading and writing.

Frames are streamed instead of loaded all at once, so long clips do not run
out of memory. The pipeline reads the video twice: once to analyze and once
to draw.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def _open(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"Could not open video: {path}")
    return capture


def get_video_info(path: str | Path) -> VideoInfo:
    capture = _open(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return VideoInfo(fps=float(fps), width=width, height=height, frame_count=count)


def iter_frames(path: str | Path, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield BGR frames one at a time."""
    capture = _open(path)
    try:
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
            index += 1
    finally:
        capture.release()


def iter_batches(path: str | Path, batch_size: int, max_frames: Optional[int] = None) -> Iterator[list[np.ndarray]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[np.ndarray] = []
    for frame in iter_frames(path, max_frames):
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_video(path: str | Path, max_frames: Optional[int] = None) -> list[np.ndarray]:
    return list(iter_frames(path, max_frames))


_FOURCC = {".mp4": "mp4v", ".avi": "XVID", ".mov": "mp4v", ".mkv": "XVID"}


def create_writer(path: str | Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = _FOURCC.get(path.suffix.lower())
    if fourcc is None:
        raise ValueError(f"Unsupported video extension: {path.suffix}")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open video writer for {path}")
    return writer


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: float = 30.0) -> None:
    if not frames:
        raise ValueError("No frames to save")
    height, width = frames[0].shape[:2]
    writer = create_writer(path, fps, width, height)
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
