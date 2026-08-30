"""Regression test: dataset + inferencer must exclude any scene that
contains a curation-dropped instance, ENTIRELY -- not just the pairs that
touch it.

Scene rendering does not consult curation.json's drop list (omni_scenes_all_
merged was found to contain 1,037/12,753 scenes with a dropped instance
placed in them anyway — a render-time bug, tracked separately). Earlier
behavior only excluded pairs touching the dropped instance, keeping the
other (valid) objects' pairs in that same scene. User explicitly requested
(2026-07-14) the stronger policy instead: if a scene contains ANY dropped
instance, the whole scene is untrustworthy and none of its pairs should
train, even pairs between two otherwise-valid objects.
AffordanceDataset/Inferencer filter using the same resolved_instances.json
already loaded for effective-class mapping, via the existing
load_dropped_set() helper (affdet/render/omni_pair_render.py already uses
this as "the single source of truth" for instance drops).
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

_CUT_I = TIER1_PREDICATES.index("canCut")


def _write_scene(scenes_dir, feats_dir):
    # cup_013 is curation-dropped but was placed in this scene anyway
    # (the render-time bug) alongside a normal knife/apple pair.
    objs = [SceneObject(obj_id="knife_001", obj_class="knife", bbox=[0, 0, 10, 10]),
            SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0, 0, 10, 10]),
            SceneObject(obj_id="cup_013", obj_class="cup", bbox=[0, 0, 10, 10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128, 96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (scenes_dir / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_001", "apple_1", "cup_013"],
             features=np.zeros((3, 768), dtype=np.float32)).save(feats_dir / "scene_0001.npz")


def _write_labels(labs_dir):
    entry = LabelEntry(actor="knife_001", recipient="apple_1", predicate="canCut",
                        verdict=1, confidence=0.9, source="t", score=0.8)
    (labs_dir / "scene_0001.json").write_text(
        Labels(scene_id="scene_0001", tier1=[entry]).model_dump_json())


def _write_resolved(path):
    path.write_text(json.dumps({
        "cup_013": {"effective_class": "cup", "dropped": True, "reason": "curation_drop"},
    }))


def test_dataset_excludes_whole_scene_containing_a_dropped_instance(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene(scenes, feats)
    _write_labels(labs)
    resolved = tmp_path / "resolved_instances.json"
    _write_resolved(resolved)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0,
                            resolved_instances=resolved)

    # cup_013 is dropped -> the WHOLE scene is excluded, including the
    # otherwise-valid knife_001<->apple_1 canCut pair that doesn't even
    # touch cup_013.
    assert len(ds) == 0


def test_dataset_keeps_scene_without_resolved_file(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene(scenes, feats)
    _write_labels(labs)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0)

    # No resolved_instances.json -> no drop info -> scene kept, both the
    # knife_001<->apple_1 canCut pair and the apple_1->cup_013 canContain
    # pair are present (contamination reproduced without the fix).
    assert len(ds) == 2


def test_dataset_keeps_scene_with_no_dropped_instances(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene(scenes, feats)
    _write_labels(labs)
    resolved = tmp_path / "resolved_instances.json"
    resolved.write_text(json.dumps({}))  # no drops at all

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0,
                            resolved_instances=resolved)

    assert len(ds) == 2


def test_inferencer_excludes_whole_scene_containing_a_dropped_instance(tmp_path):
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

    # Whole scene dropped -> no predictions file written for it at all.
    assert not (out / "scene_0001.json").exists()
