"""Tests for the single-photo end-to-end orchestrator: raw image -> YOLO boxes
-> synthetic Scene -> DINOv2 crop features -> AffordanceMLP -> Predictions.

Heavy deps (YOLO, DINOv2 backbone, trained MLP) are injected fakes, matching
the injection pattern used throughout (DINOv2Extractor.backbone,
yolo_boxes.Predictor, Inferencer.model_factory) so this stays unit-testable.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from affdet.extract.dinov2_classifier import DINOv2Classifier
from affdet.model.mlp import AffordanceMLP
from affdet.scripts.detect_and_infer import resolve_classifier_path, run_single_image


class _FakeBackbone:
    def parameters(self):
        yield torch.zeros(1)

    def forward_features(self, crops: torch.Tensor) -> dict:
        n = crops.shape[0]
        return {"x_norm_clstoken": torch.zeros(n, 768)}


def _fake_predictor(image_path: Path, scene) -> list[dict]:
    return [
        {"bbox": [5.0, 5.0, 40.0, 60.0], "cls": "knife"},
        {"bbox": [50.0, 10.0, 100.0, 70.0], "cls": "apple"},
    ]


def test_run_single_image_writes_predictions_and_returns_them(tmp_path):
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 150), color=(128, 128, 128)).save(img_path)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    out_dir = tmp_path / "out"

    preds = run_single_image(
        image_path=img_path,
        ckpt_dir=ckpt_dir,
        out_dir=out_dir,
        predictor=_fake_predictor,
        backbone=_FakeBackbone(),
    )

    assert preds.tier1
    for e in preds.tier1:
        assert 0.0 <= e.score <= 1.0
        assert e.verdict in (0, 1)
    assert (out_dir / "scene" / "scene_0001.json").exists()
    assert (out_dir / "features" / "scene_0001.npz").exists()
    assert (out_dir / "predictions.json").exists()


def test_run_single_image_all_pairs_scores_structurally_gated_pairs(tmp_path):
    """Two knives are never a candidate pair for any Tier-1 predicate under
    the default class gate, but all_pairs=True should still score them (the
    mode meant to survive a wrong detector class label)."""
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 150), color=(128, 128, 128)).save(img_path)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    two_knives = lambda image_path, scene: [
        {"bbox": [5.0, 5.0, 40.0, 60.0], "cls": "knife"},
        {"bbox": [50.0, 10.0, 100.0, 70.0], "cls": "knife"},
    ]

    gated = run_single_image(
        image_path=img_path, ckpt_dir=ckpt_dir, out_dir=tmp_path / "gated",
        predictor=two_knives, backbone=_FakeBackbone(),
    )
    assert gated.tier1 == []

    all_pairs = run_single_image(
        image_path=img_path, ckpt_dir=ckpt_dir, out_dir=tmp_path / "all_pairs",
        predictor=two_knives, backbone=_FakeBackbone(), all_pairs=True,
    )
    assert len(all_pairs.tier1) == 12  # 2 ordered pairs x 6 predicates


class _FakeVaseProbe:
    """Always predicts "vase" regardless of feature values -- enough to prove
    relabeling actually happened (the fake predictor below hands out "kettle",
    a class the class-based candidate gate treats differently)."""

    def predict(self, features):
        return np.array(["vase"] * len(features))


def test_run_single_image_with_classifier_overrides_detector_class(tmp_path):
    """Wiring point for the DINOv2-probe-vs-YOLO-classhead diagnostic result
    (probe 98.76% vs YOLO 93.12%, see dinov2_probe_beats_yolo_classhead memory):
    when a classifier is injected, each object's obj_class in the written
    Scene should come from the probe, not the detector's own "cls"."""
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 150), color=(128, 128, 128)).save(img_path)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    out_dir = tmp_path / "out"

    kettle_predictor = lambda image_path, scene: [
        {"bbox": [5.0, 5.0, 40.0, 60.0], "cls": "kettle", "conf": 0.9},
    ]

    run_single_image(
        image_path=img_path,
        ckpt_dir=ckpt_dir,
        out_dir=out_dir,
        predictor=kettle_predictor,
        backbone=_FakeBackbone(),
        classifier=DINOv2Classifier(model=_FakeVaseProbe()),
        all_pairs=True,
    )

    import json
    scene = json.loads((out_dir / "scene" / "scene_0001.json").read_text())
    assert scene["objects"][0]["obj_class"] == "vase"


def test_resolve_classifier_path_defaults_to_dinov2_probe_when_unset():
    """DINOv2 classification is now the default (per the supervisor-approved
    direction: remove class detection from YOLO, move it to DINOv2) -- an
    unset --dinov2-probe should resolve to the default probe path, not None."""
    resolved = resolve_classifier_path(dinov2_probe=None, no_dinov2_probe=False,
                                        default_path=Path("/default/probe.pkl"))
    assert resolved == Path("/default/probe.pkl")


def test_resolve_classifier_path_explicit_path_overrides_default():
    resolved = resolve_classifier_path(dinov2_probe=Path("/custom/probe.pkl"),
                                        no_dinov2_probe=False,
                                        default_path=Path("/default/probe.pkl"))
    assert resolved == Path("/custom/probe.pkl")


def test_resolve_classifier_path_no_dinov2_probe_flag_disables_it():
    """Escape hatch back to YOLO's own class head."""
    resolved = resolve_classifier_path(dinov2_probe=None, no_dinov2_probe=True,
                                        default_path=Path("/default/probe.pkl"))
    assert resolved is None


def test_resolve_classifier_path_no_dinov2_probe_wins_over_explicit_path():
    resolved = resolve_classifier_path(dinov2_probe=Path("/custom/probe.pkl"),
                                        no_dinov2_probe=True,
                                        default_path=Path("/default/probe.pkl"))
    assert resolved is None


def test_run_single_image_no_detections_gives_empty_predictions(tmp_path):
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 150), color=(0, 0, 0)).save(img_path)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    out_dir = tmp_path / "out"

    preds = run_single_image(
        image_path=img_path,
        ckpt_dir=ckpt_dir,
        out_dir=out_dir,
        predictor=lambda image_path, scene: [],
        backbone=_FakeBackbone(),
    )
    assert preds.tier1 == []
