"""Model wrappers.

Every model sits behind a tiny interface (``detect(frames)``) so the pipeline
does not care whether results come from YOLO, a calibration file, or a fake
used in tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str


class Detector(Protocol):
    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        ...


class KeypointSource(Protocol):
    def detect(self, frames: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Return one (K, 2) array per frame with NaN for missing points, or None."""
        ...


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def matches(class_name: str, wanted: Iterable[str]) -> bool:
    return normalize_name(class_name) in {normalize_name(w) for w in wanted}


def resolve_weights(path: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    """Use ``path`` when the file exists, otherwise the fallback (which may be None)."""
    if path and Path(path).exists():
        return path
    if path and fallback:
        logger.info("Model %s not found, using %s", path, fallback)
    return fallback


class YoloDetector:
    """Ultralytics YOLO object detector."""

    def __init__(self, weights: str, confidence: float = 0.1, device: Optional[str] = None, image_size: int = 1280):
        from ultralytics import YOLO  # imported lazily so tests do not need it

        self.weights = weights
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.image_size = image_size
        self.class_names: dict[int, str] = dict(self.model.names)

    def has_class(self, wanted: Iterable[str]) -> bool:
        return any(matches(name, wanted) for name in self.class_names.values())

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(
            list(frames), conf=self.confidence, device=self.device, imgsz=self.image_size, verbose=False
        )
        output: list[list[Detection]] = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls in zip(xyxy, confidences, class_ids):
                    frame_detections.append(
                        Detection(tuple(float(v) for v in box), float(conf), self.class_names.get(int(cls), str(cls)))
                    )
            output.append(frame_detections)
        return output


class YoloKeypointDetector:
    """Ultralytics YOLO pose model trained on court landmarks."""

    def __init__(self, weights: str, confidence: float = 0.5, device: Optional[str] = None, num_keypoints: int = 18):
        from ultralytics import YOLO

        self.weights = weights
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.num_keypoints = num_keypoints

    def detect(self, frames: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        results = self.model.predict(list(frames), conf=self.confidence, device=self.device, verbose=False)
        output: list[Optional[np.ndarray]] = []
        for result in results:
            keypoints = result.keypoints
            if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                output.append(None)
                continue
            xy = keypoints.xy[0].cpu().numpy().astype(float)
            conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(len(xy))
            points = np.full((self.num_keypoints, 2), np.nan)
            count = min(self.num_keypoints, len(xy))
            for i in range(count):
                x, y = xy[i]
                # YOLO reports undetected keypoints as (0, 0)
                if conf[i] >= self.confidence and not (x == 0 and y == 0):
                    points[i] = (x, y)
            output.append(points)
        return output


class StaticCourtCalibration:
    """Court keypoints typed in by hand for a fixed camera.

    JSON format: ``{"keypoints": {"0": [x, y], "7": [x, y], ...}}`` where the
    keys are court keypoint indices (see ``netvision.court.geometry``).
    """

    def __init__(self, keypoints: dict[int, Sequence[float]], num_keypoints: int = 18):
        if len(keypoints) < 4:
            raise ValueError("A calibration needs at least 4 keypoints")
        self.points = np.full((num_keypoints, 2), np.nan)
        for index, (x, y) in keypoints.items():
            index = int(index)
            if not 0 <= index < num_keypoints:
                raise ValueError(f"Keypoint index {index} is out of range")
            self.points[index] = (float(x), float(y))

    @classmethod
    def from_json(cls, path: str | Path) -> "StaticCourtCalibration":
        import json

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls({int(k): v for k, v in data["keypoints"].items()})

    def detect(self, frames: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        return [self.points.copy() for _ in frames]
