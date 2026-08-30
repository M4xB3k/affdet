"""Relabel detector predictions with a DINOv2-feature linear probe.

Diagnostic result (2026-07-22, full synthetic corpus, 4750 held-out objects,
identical YOLO-predicted crops fed to both methods): a DINOv2 linear probe
classifies 98.76% correctly vs. the fine-tuned YOLO-World detector's own
class head at 93.12%. YOLO's errors cluster in shape-similar produce
(teapot->vase, pear->lemon, chili->cucumber); the probe all but eliminates
those. This module swaps the detector's class label for the probe's
prediction while keeping the detector's box (localisation) untouched --
the diagnostic showed the box/localisation problem and the classification
problem are separate, and only the second one improves this way.

The probe is injected (like DINOv2Extractor's backbone), so this stays
unit-testable without loading a real classifier; ``load_probe`` builds the
real one lazily from a persisted pickle (a fitted sklearn classifier, see
dinov2_vs_yolo_classhead.py which produced dinov2_probe_fullcorpus.pkl).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DINOv2Classifier:
    """Wraps a fitted classifier exposing sklearn's ``predict(X) -> labels``."""

    model: Any

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


def load_probe(path: Path | str) -> DINOv2Classifier:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return DINOv2Classifier(model=model)


def relabel_predictions(
    preds: list[dict[str, Any]],
    features: np.ndarray,
    classifier: DINOv2Classifier,
) -> list[dict[str, Any]]:
    """Return a copy of ``preds`` with each "cls" replaced by the classifier's
    prediction from the matching row of ``features`` (same order, one row per
    prediction -- callers extract DINOv2 features from each prediction's own
    bbox crop, so row i's feature corresponds to preds[i])."""
    if len(preds) != len(features):
        raise ValueError(
            f"length mismatch: {len(preds)} predictions vs "
            f"{len(features)} feature rows"
        )
    if not preds:
        return []

    labels = classifier.predict(features)
    return [
        {**p, "cls": str(label)}
        for p, label in zip(preds, labels)
    ]
