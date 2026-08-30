"""Tests for affdet.extract.dinov2_classifier — relabel YOLO predictions using
a DINOv2-feature linear probe instead of the detector's own class head.

Diagnostic result (dinov2_probe_beats_yolo_classhead memory, 2026-07-22):
probe scores 98.76% vs YOLO's own class head 93.12% on identical predicted
crops (4750 held-out objects, full synthetic corpus) -- this module wires
that result into the live pipeline.
"""
from pathlib import Path

import numpy as np
import pytest

from affdet.extract.dinov2_classifier import (
    DINOv2Classifier,
    relabel_predictions,
)


class _FakeProbe:
    """Mimics sklearn's LogisticRegression.predict(X) -> array of labels."""

    def __init__(self, label_for_feature: dict[float, str]):
        self._label_for_feature = label_for_feature

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Each fake feature is a single repeated float; use its first entry
        # as the lookup key so tests can control predictions deterministically.
        return np.array([self._label_for_feature[row[0]] for row in X])


def test_relabel_overrides_cls_with_probe_prediction():
    clf = DINOv2Classifier(model=_FakeProbe({1.0: "teapot", 2.0: "vase"}))
    preds = [
        {"bbox": [0, 0, 10, 10], "cls": "vase", "conf": 0.9},
        {"bbox": [10, 10, 20, 20], "cls": "kettle", "conf": 0.8},
    ]
    features = np.array([
        [1.0] * 768,
        [2.0] * 768,
    ])

    out = relabel_predictions(preds, features, clf)

    assert out[0]["cls"] == "teapot"
    assert out[1]["cls"] == "vase"


def test_relabel_preserves_bbox_and_conf():
    clf = DINOv2Classifier(model=_FakeProbe({5.0: "mango"}))
    preds = [{"bbox": [1, 2, 3, 4], "cls": "potato", "conf": 0.42}]
    features = np.array([[5.0] * 768])

    out = relabel_predictions(preds, features, clf)

    assert out[0]["bbox"] == [1, 2, 3, 4]
    assert out[0]["conf"] == 0.42


def test_relabel_empty_predictions_returns_empty():
    clf = DINOv2Classifier(model=_FakeProbe({}))
    out = relabel_predictions([], np.zeros((0, 768)), clf)
    assert out == []


def test_relabel_mismatched_lengths_raises():
    clf = DINOv2Classifier(model=_FakeProbe({1.0: "mango"}))
    preds = [{"bbox": [0, 0, 1, 1], "cls": "x", "conf": 1.0},
             {"bbox": [0, 0, 1, 1], "cls": "x", "conf": 1.0}]
    features = np.array([[1.0] * 768])

    with pytest.raises(ValueError, match="length"):
        relabel_predictions(preds, features, clf)


def test_load_probe_roundtrip(tmp_path: Path):
    import pickle
    from affdet.extract.dinov2_classifier import load_probe

    fake = _FakeProbe({9.0: "cheese"})
    pkl_path = tmp_path / "probe.pkl"
    pkl_path.write_bytes(pickle.dumps(fake))

    clf = load_probe(pkl_path)

    assert clf.predict(np.array([[9.0] * 768])) == ["cheese"]
