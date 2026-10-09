"""Assign each tracked player to a team from jersey appearance.

Two classifiers are available:

* ``ClipJerseyClassifier``: zero-shot image classification with the
  Hugging Face fashion-clip model, the approach from the tutorial. Each crop
  is scored against two text labels such as "white shirt" and
  "dark blue shirt".
* ``ColorJerseyClassifier``: no download needed. It finds the dominant jersey
  color of each track and splits the tracks into two groups with KMeans.
  Tracks that match neither team color (referees, fans) get team 0.

Both classifiers see several crops per track and decide once per track, so a
player cannot flip teams halfway through the clip.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from ..utils.bbox import clip_box

logger = logging.getLogger(__name__)

TEAM_UNKNOWN = 0


class JerseyClassifier(Protocol):
    def assign(self, crops_by_track: dict[int, list[np.ndarray]]) -> dict[int, int]:
        ...


def crop_player(frame: np.ndarray, bbox: Sequence[float], max_height: int = 128) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(bbox, width, height)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] > max_height:
        scale = max_height / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max_height), interpolation=cv2.INTER_AREA)
    return crop.copy()


def dominant_jersey_color(crop: np.ndarray) -> np.ndarray:
    """Return the jersey color of a player crop in Lab color space.

    Looks at the torso (rows 20-50 percent, middle half of the width), where
    the jersey fills most of the pixels even when the box also catches floor,
    crowd, or arms. Pixels are split into two clusters and the larger
    cluster is taken as the jersey color.
    """
    height, width = crop.shape[:2]
    torso = crop[int(0.2 * height): max(int(0.2 * height) + 1, int(0.5 * height)),
                 int(0.25 * width): max(int(0.25 * width) + 1, int(0.75 * width))]
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < 2:
        return lab.mean(axis=0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(lab, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=2)
    return centers[int(np.argmax(counts))]


class ColorJerseyClassifier:
    def __init__(self, outlier_factor: float = 3.0, outlier_min_distance: float = 25.0, seed: int = 0):
        self.outlier_factor = outlier_factor
        self.outlier_min_distance = outlier_min_distance
        self.seed = seed
        self.team_colors: Optional[np.ndarray] = None  # Lab, row 0 = team 1

    def track_color(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        colors = np.array([dominant_jersey_color(c) for c in crops])
        return np.median(colors, axis=0)

    def assign(self, crops_by_track: dict[int, list[np.ndarray]]) -> dict[int, int]:
        track_ids = [t for t, crops in crops_by_track.items() if crops]
        if not track_ids:
            return {}
        features = np.array([self.track_color(crops_by_track[t]) for t in track_ids], dtype=np.float32)
        if len(track_ids) == 1:
            self.team_colors = np.array([features[0], features[0]])
            return {track_ids[0]: 1}

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(self.seed)
        _, labels, centers = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        # Median of each group is robust to a referee pulling the mean.
        for k in range(2):
            members = features[labels == k]
            if len(members):
                centers[k] = np.median(members, axis=0)
        # Team 1 is the lighter jersey (higher L in Lab), team 2 the darker one.
        order = np.argsort(-centers[:, 0])
        cluster_to_team = {int(order[0]): 1, int(order[1]): 2}
        self.team_colors = centers[order]

        distances = np.linalg.norm(features - centers[labels], axis=1)
        median = float(np.median(distances)) if len(distances) else 0.0
        limit = max(self.outlier_factor * median, self.outlier_min_distance)

        teams = {}
        for track_id, label, distance in zip(track_ids, labels, distances):
            if self.outlier_factor > 0 and distance > limit:
                teams[track_id] = TEAM_UNKNOWN
            else:
                teams[track_id] = cluster_to_team[int(label)]
        return teams


class ClipJerseyClassifier:
    """Zero-shot jersey classifier using a CLIP model from Hugging Face."""

    def __init__(self, model_id: str, team1_label: str, team2_label: str, device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor  # heavy import, kept lazy

        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.labels = [team1_label, team2_label]
        self.device = device or "cpu"
        self.model.to(self.device)
        self.model.eval()

    def classify(self, crops: Sequence[np.ndarray], batch_size: int = 32) -> list[int]:
        import torch
        from PIL import Image

        teams: list[int] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch]
            inputs = self.processor(text=self.labels, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits_per_image
            probabilities = logits.softmax(dim=1).cpu().numpy()
            teams.extend(int(np.argmax(p)) + 1 for p in probabilities)
        return teams

    def assign(self, crops_by_track: dict[int, list[np.ndarray]]) -> dict[int, int]:
        teams = {}
        for track_id, crops in crops_by_track.items():
            if not crops:
                continue
            votes = Counter(self.classify(crops))
            teams[track_id] = votes.most_common(1)[0][0]
        return teams


def build_jersey_classifier(kind: str, model_id: str, team1_label: str, team2_label: str, device: Optional[str] = None,
                            outlier_factor: float = 3.0, outlier_min_distance: float = 25.0) -> JerseyClassifier:
    kind = kind.lower()
    if kind == "color":
        return ColorJerseyClassifier(outlier_factor, outlier_min_distance)
    if kind == "clip":
        return ClipJerseyClassifier(model_id, team1_label, team2_label, device)
    if kind == "auto":
        try:
            return ClipJerseyClassifier(model_id, team1_label, team2_label, device)
        except Exception as error:  # missing package, no network, etc.
            logger.warning("CLIP team classifier unavailable (%s). Using jersey color clustering.", error)
            return ColorJerseyClassifier(outlier_factor, outlier_min_distance)
    raise ValueError(f"Unknown team classifier: {kind}")


class TeamAssigner:
    """Collects player crops while the video streams, then decides teams per track."""

    def __init__(self, classifier: JerseyClassifier, sample_every_n_frames: int = 5, max_samples_per_track: int = 15):
        self.classifier = classifier
        self.sample_every_n_frames = max(1, sample_every_n_frames)
        self.max_samples_per_track = max_samples_per_track
        self.crops: dict[int, list[np.ndarray]] = defaultdict(list)

    def observe(self, frame: np.ndarray, frame_index: int, players: dict[int, list[float]]) -> None:
        for track_id, bbox in players.items():
            samples = self.crops[track_id]
            if len(samples) >= self.max_samples_per_track:
                continue
            # Always take the first sighting, then sample every N frames.
            if samples and frame_index % self.sample_every_n_frames != 0:
                continue
            crop = crop_player(frame, bbox)
            if crop is not None:
                samples.append(crop)

    def finalize(self) -> dict[int, int]:
        return self.classifier.assign(dict(self.crops))

    def state(self) -> dict[int, list[np.ndarray]]:
        return dict(self.crops)

    def load_state(self, crops: dict[int, list[np.ndarray]]) -> None:
        self.crops = defaultdict(list, crops)
