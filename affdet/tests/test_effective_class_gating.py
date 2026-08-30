"""Cross-consumer regression test: dataset + inferencer must gate candidate
pairs on EFFECTIVE class (like the label deriver), not raw obj_class.

Without this, a recategorized peeler (scene obj_class="knife",
obj_id="knife_011", effective class "peeler") gets canPeel labels from the
deriver, but dataset/infer would gate on raw "knife" and build canCut/canPierce
columns instead — the peel labels are never trained/predicted.
"""
import json

import numpy as np
import torch

from affdet.schemas import (Scene, SceneObject, Camera, Labels, LabelEntry,
                            Features, Predictions)
from affdet.model.dataset import AffordanceDataset
from affdet.model.mlp import AffordanceMLP
from affdet.model.infer import Inferencer
from affdet.intersection_matrix import TIER1_PREDICATES

_PEEL_I = TIER1_PREDICATES.index("canPeel")
_CUT_I = TIER1_PREDICATES.index("canCut")


def _write_scene(scenes_dir, feats_dir):
    objs = [SceneObject(obj_id="knife_011", obj_class="knife", bbox=[0, 0, 10, 10]),
            SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0, 0, 10, 10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128, 96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (scenes_dir / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_011", "apple_1"],
             features=np.zeros((2, 768), dtype=np.float32)).save(feats_dir / "scene_0001.npz")


def _write_labels(labs_dir):
    # Deriver already gated on effective class "peeler" -> emits canPeel,
    # keyed by scene obj_id knife_011 (not by effective class).
    entry = LabelEntry(actor="knife_011", recipient="apple_1", predicate="canPeel",
                        verdict=1, confidence=0.9, source="t", score=0.8)
    (labs_dir / "scene_0001.json").write_text(
        Labels(scene_id="scene_0001", tier1=[entry]).model_dump_json())


def _write_resolved(path):
    path.write_text(json.dumps({"knife_011": {"effective_class": "peeler"}}))


def _mask_for(ds, pred_idx):
    for i in range(len(ds)):
        x, y, m = ds[i]
        if m[pred_idx] > 0:
            return True
    return False


def test_dataset_gates_on_effective_class_with_resolved_file(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene(scenes, feats)
    _write_labels(labs)
    resolved = tmp_path / "resolved_instances.json"
    _write_resolved(resolved)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0,
                            resolved_instances=resolved)

    assert _mask_for(ds, _PEEL_I) is True
    assert _mask_for(ds, _CUT_I) is False


def test_dataset_falls_back_to_raw_class_without_resolved_file(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene(scenes, feats)
    _write_labels(labs)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0)

    assert _mask_for(ds, _CUT_I) is True
    assert _mask_for(ds, _PEEL_I) is False


def test_inferencer_gates_on_effective_class_with_resolved_file(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    ckpt = tmp_path / "ckpt"; ckpt.mkdir()
    out = tmp_path / "preds"
    _write_scene(scenes, feats)
    _write_labels(labs)
    resolved = tmp_path / "resolved_instances.json"
    _write_resolved(resolved)
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt / "best.pt")

    Inferencer(resolved_instances=resolved).predict(scenes, feats, labs, ckpt, out)

    preds = Predictions.model_validate_json((out / "scene_0001.json").read_text())
    keyed = {(e.actor, e.recipient, e.predicate) for e in preds.tier1}
    assert ("knife_011", "apple_1", "canPeel") in keyed
    assert ("knife_011", "apple_1", "canCut") not in keyed
