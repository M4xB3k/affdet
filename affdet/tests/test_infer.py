import numpy as np
import torch
from pathlib import Path

from affdet.schemas import (Scene, SceneObject, Camera, Labels, LabelEntry,
                            Features, Predictions)
from affdet.model.mlp import AffordanceMLP
from affdet.model.infer import Inferencer


def _setup(tmp_path):
    scenes = tmp_path/"scenes"; scenes.mkdir()
    feats = tmp_path/"feats"; feats.mkdir()
    labs = tmp_path/"labs"; labs.mkdir()
    ckpt = tmp_path/"ckpt"; ckpt.mkdir()
    objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0,0,10,10]),
            SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0,0,10,10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128,96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                            intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
    (scenes/"scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1","apple_1"],
             features=np.zeros((2,768),dtype=np.float32)).save(feats/"scene_0001.npz")
    (labs/"scene_0001.json").write_text(Labels(scene_id="scene_0001", tier1=[
        LabelEntry(actor="knife_1", recipient="apple_1", predicate="canCut",
                   verdict=1, confidence=0.9, source="t", score=0.8)
    ]).model_dump_json())
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt/"best.pt")
    return scenes, feats, labs, ckpt


def test_writes_predictions_with_score_in_range(tmp_path):
    scenes, feats, labs, ckpt = _setup(tmp_path)
    out = tmp_path/"preds"
    Inferencer().predict(scenes, feats, labs, ckpt, out)
    preds = Predictions.model_validate_json((out/"scene_0001.json").read_text())
    assert preds.tier1
    for e in preds.tier1:
        assert 0.0 <= e.score <= 1.0
        assert e.verdict in (0, 1)
        assert (e.verdict == 1) == (e.score >= 0.5)


def _setup_two_knives(tmp_path):
    """A pair the intersection matrix structurally forbids for every
    predicate (canCut/canPierce need a food recipient; two knives are never a
    candidate pair for anything) — the case all_pairs is meant to still score,
    e.g. when a detector mislabels a real recipient as another utensil."""
    scenes = tmp_path/"scenes"; scenes.mkdir()
    feats = tmp_path/"feats"; feats.mkdir()
    labs = tmp_path/"labs"; labs.mkdir()
    ckpt = tmp_path/"ckpt"; ckpt.mkdir()
    objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0,0,10,10]),
            SceneObject(obj_id="knife_2", obj_class="knife", bbox=[0,0,10,10])]
    s = Scene(scene_id="scene_0002", image_path="x.png", image_size=[128,96],
              source="real", objects=objs,
              camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                            intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
    (scenes/"scene_0002.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1","knife_2"],
             features=np.zeros((2,768),dtype=np.float32)).save(feats/"scene_0002.npz")
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt/"best.pt")
    return scenes, feats, labs, ckpt


def test_default_mode_excludes_structurally_impossible_pairs(tmp_path):
    scenes, feats, labs, ckpt = _setup_two_knives(tmp_path)
    out = tmp_path/"preds"
    Inferencer().predict(scenes, feats, labs, ckpt, out)
    preds = Predictions.model_validate_json((out/"scene_0002.json").read_text())
    assert preds.tier1 == []


def _setup_many_scenes(tmp_path, n=10):
    """n independent scenes (knife/apple canCut) for split-filtering tests --
    enough scenes that a 70/15/15 split gives a non-trivial test subset."""
    scenes = tmp_path/"scenes"; scenes.mkdir()
    feats = tmp_path/"feats"; feats.mkdir()
    labs = tmp_path/"labs"; labs.mkdir()
    ckpt = tmp_path/"ckpt"; ckpt.mkdir()
    for i in range(n):
        sid = f"scene_{i:04d}"
        objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0,0,10,10]),
                SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0,0,10,10])]
        s = Scene(scene_id=sid, image_path="x.png", image_size=[128,96],
                  source="blender", objects=objs,
                  camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
        (scenes/f"{sid}.json").write_text(s.model_dump_json())
        Features(obj_ids=["knife_1","apple_1"],
                 features=np.zeros((2,768),dtype=np.float32)).save(feats/f"{sid}.npz")
        (labs/f"{sid}.json").write_text(Labels(scene_id=sid, tier1=[
            LabelEntry(actor="knife_1", recipient="apple_1", predicate="canCut",
                       verdict=1, confidence=0.9, source="t", score=0.8)
        ]).model_dump_json())
    torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt/"best.pt")
    return scenes, feats, labs, ckpt


def test_split_filter_scores_only_that_splits_scenes(tmp_path):
    """Inferencer(split=...) must match AffordanceDataset's own seeded 70/15/15
    scene split exactly, so scoring 'test' never touches a scene the model was
    trained or early-stopped on -- otherwise reported F1 is inflated by
    memorized training data, not a real held-out number."""
    from affdet.model.dataset import scene_split_map
    scenes, feats, labs, ckpt = _setup_many_scenes(tmp_path, n=10)

    out_test = tmp_path/"preds_test"
    Inferencer(split="test", seed=0).predict(scenes, feats, labs, ckpt, out_test)

    split_map = scene_split_map(scenes, seed=0)
    test_scene_ids = {sid for sid, sp in split_map.items() if sp == "test"}
    written_scene_ids = {p.stem for p in out_test.glob("scene_*.json")}

    assert written_scene_ids == test_scene_ids
    assert written_scene_ids  # split must be non-empty for n=10


def test_no_split_scores_all_scenes(tmp_path):
    """Backwards compatible: split=None (default) scores every scene, as
    before this change."""
    scenes, feats, labs, ckpt = _setup_many_scenes(tmp_path, n=10)
    out = tmp_path/"preds_all"
    Inferencer().predict(scenes, feats, labs, ckpt, out)
    written_scene_ids = {p.stem for p in out.glob("scene_*.json")}
    assert len(written_scene_ids) == 10


def test_all_pairs_scores_every_predicate_for_every_ordered_pair(tmp_path):
    scenes, feats, labs, ckpt = _setup_two_knives(tmp_path)
    out = tmp_path/"preds"
    Inferencer(all_pairs=True).predict(scenes, feats, labs, ckpt, out)
    preds = Predictions.model_validate_json((out/"scene_0002.json").read_text())
    # 2 ordered pairs (knife_1->knife_2, knife_2->knife_1) x 6 predicates
    assert len(preds.tier1) == 12
    pairs = {(e.actor, e.recipient, e.predicate) for e in preds.tier1}
    assert ("knife_1", "knife_2", "canCut") in pairs
    for e in preds.tier1:
        assert 0.0 <= e.score <= 1.0


class _StubClassifier:
    """Predicts a fixed label per row, in call order -- stands in for
    DINOv2Classifier without a real fitted model, matching how Inferencer's
    `model_factory` is stubbed above."""

    def __init__(self, labels):
        self._labels = list(labels)

    def predict(self, features):
        assert len(features) == len(self._labels)
        return list(self._labels)


def test_classifier_gate_uses_predicted_class_not_ground_truth(tmp_path):
    """gate_classifier swaps is_candidate_pair's input class from
    scene.obj_class (ground truth) to the classifier's own prediction on that
    object's DINOv2 features -- this is what lets a gated eval measure the
    REAL probe's error rate instead of assuming perfect labels (E1's gate is
    always fed ground truth; this is the probe-in-the-loop version)."""
    scenes, feats, labs, ckpt = _setup_two_knives(tmp_path)
    out = tmp_path / "preds"
    # Ground truth is knife/knife (never a candidate pair for anything).
    # The stub claims knife_1 is "knife" but knife_2 is "apple" -- with that
    # WRONG label, canCut(knife_1 -> knife_2) becomes structurally admissible
    # even though the true recipient is a knife.
    clf = _StubClassifier(["knife", "apple"])
    Inferencer(gate_classifier=clf).predict(scenes, feats, labs, ckpt, out)
    preds = Predictions.model_validate_json((out / "scene_0002.json").read_text())
    pairs = {(e.actor, e.recipient, e.predicate) for e in preds.tier1}
    assert ("knife_1", "knife_2", "canCut") in pairs


def test_no_gate_classifier_falls_back_to_ground_truth(tmp_path):
    """Backwards compatible: gate_classifier=None (default) gates on
    scene.obj_class exactly as before this change."""
    scenes, feats, labs, ckpt = _setup_two_knives(tmp_path)
    out = tmp_path / "preds"
    Inferencer().predict(scenes, feats, labs, ckpt, out)
    preds = Predictions.model_validate_json((out / "scene_0002.json").read_text())
    assert preds.tier1 == []
