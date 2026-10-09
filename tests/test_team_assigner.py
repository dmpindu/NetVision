"""Unit tests for team assignment."""
import sys
import types

import cv2
import numpy as np
import pytest

from netvision.team_assigner import (
    TEAM_UNKNOWN,
    ColorJerseyClassifier,
    TeamAssigner,
    build_jersey_classifier,
    crop_player,
    dominant_jersey_color,
)
from netvision.team_assigner.team_assigner import ClipJerseyClassifier

WHITE = (240, 240, 240)
BLUE = (110, 40, 20)
GRAY = (128, 128, 128)
FLOOR = (110, 170, 215)


def player_crop(jersey, height=100, width=45):
    crop = np.full((height, width, 3), FLOOR, dtype=np.uint8)
    cv2.rectangle(crop, (int(0.1 * width), int(0.2 * height)), (int(0.9 * width), int(0.56 * height)), jersey, -1)
    cv2.circle(crop, (width // 2, int(0.1 * height)), int(0.09 * height), (90, 130, 190), -1)
    return crop


def to_lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def test_dominant_jersey_color_ignores_background():
    """TC-TEAM-01: the torso color is found even with floor pixels in the crop."""
    color = dominant_jersey_color(player_crop(BLUE))
    assert np.linalg.norm(color - to_lab(BLUE)) < 5


def test_color_classifier_splits_teams_lighter_is_team1():
    """TC-TEAM-02: white and dark blue tracks split into team 1 (lighter) and team 2."""
    crops = {1: [player_crop(WHITE)] * 3, 2: [player_crop(BLUE)] * 3, 3: [player_crop(WHITE)], 4: [player_crop(BLUE)]}
    teams = ColorJerseyClassifier().assign(crops)
    assert teams == {1: 1, 2: 2, 3: 1, 4: 2}


def test_color_classifier_flags_referee_as_unknown():
    """TC-TEAM-03: a track whose jersey matches neither team (referee) gets team 0."""
    crops = {i: [player_crop(WHITE)] for i in range(3)}
    crops.update({i: [player_crop(BLUE)] for i in range(3, 6)})
    crops[9] = [player_crop(GRAY)]
    teams = ColorJerseyClassifier().assign(crops)
    assert teams[9] == TEAM_UNKNOWN
    assert [teams[i] for i in range(6)] == [1, 1, 1, 2, 2, 2]


def test_color_classifier_edge_cases():
    """TC-TEAM-04: no tracks gives {}, one track gives team 1, empty crop lists are skipped."""
    assert ColorJerseyClassifier().assign({}) == {}
    assert ColorJerseyClassifier().assign({7: [player_crop(BLUE)], 8: []}) == {7: 1}


def test_team_assigner_sampling_limits():
    """TC-TEAM-05: crops are sampled every N frames up to a maximum per track."""
    frame = np.full((300, 300, 3), 100, dtype=np.uint8)
    assigner = TeamAssigner(ColorJerseyClassifier(), sample_every_n_frames=5, max_samples_per_track=3)
    for f in range(40):
        assigner.observe(frame, f, {1: [10, 10, 60, 120]})
    assert len(assigner.state()[1]) == 3
    assigner = TeamAssigner(ColorJerseyClassifier(), sample_every_n_frames=5, max_samples_per_track=10)
    for f in range(1, 11):
        assigner.observe(frame, f, {1: [10, 10, 60, 120]})
    assert len(assigner.state()[1]) == 3  # first sighting at f=1, then f=5 and f=10


def test_crop_player_handles_tiny_and_large_boxes():
    """TC-TEAM-06: tiny boxes are skipped and large crops are resized to 128 px high."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert crop_player(frame, (10, 10, 12, 12)) is None
    assert crop_player(frame, (0, 0, 100, 400)).shape[0] == 128


class _FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def softmax(self, dim):
        e = np.exp(self.data - self.data.max(axis=dim, keepdims=True))
        return _FakeTensor(e / e.sum(axis=dim, keepdims=True))

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def to(self, device):
        return self


def _install_fake_transformers(monkeypatch):
    """Fake CLIP that says 'white shirt' when the crop is bright."""

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def __call__(self, text, images, return_tensors, padding):
            brightness = [float(np.asarray(img).mean()) for img in images]
            return {"brightness": _FakeTensor(brightness)}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, brightness):
            b = brightness.data
            logits = np.stack([b / 10, (255 - b) / 10], axis=1)
            return types.SimpleNamespace(logits_per_image=_FakeTensor(logits))

    fake = types.ModuleType("transformers")
    fake.CLIPModel = FakeModel
    fake.CLIPProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake)

    import contextlib

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_clip_classifier_majority_vote(monkeypatch):
    """TC-TEAM-07: the CLIP classifier labels each crop and takes a majority vote per track."""
    _install_fake_transformers(monkeypatch)
    classifier = ClipJerseyClassifier("fake/model", "white shirt", "dark blue shirt")
    white, blue = np.full((40, 20, 3), 240, np.uint8), np.full((40, 20, 3), 30, np.uint8)
    assert classifier.classify([white, blue]) == [1, 2]
    teams = classifier.assign({1: [white, white, blue], 2: [blue, blue, white], 3: []})
    assert teams == {1: 1, 2: 2}


def test_build_classifier_auto_falls_back_to_color(monkeypatch):
    """TC-TEAM-08: 'auto' uses jersey colors when CLIP cannot be loaded; bad names raise."""
    broken = types.ModuleType("transformers")  # no CLIPModel attribute -> ImportError
    monkeypatch.setitem(sys.modules, "transformers", broken)
    assert isinstance(build_jersey_classifier("auto", "x", "a", "b"), ColorJerseyClassifier)
    assert isinstance(build_jersey_classifier("color", "x", "a", "b"), ColorJerseyClassifier)
    with pytest.raises(ValueError):
        build_jersey_classifier("magic", "x", "a", "b")
