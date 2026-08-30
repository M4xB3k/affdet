"""Trainer(holdout_mode=...) — E1c/E1c' need Trainer to build its train/val
AffordanceDatasets with the pair-level instance holdout, same as it already
does for split_strategy. Without this, ExperimentConfig has nowhere to route
holdout_mode/holdout_frac even after they're added there.

Reuses the hub-corpus fixture from test_dataset_holdout_wiring.py: one mesh
(knife_001) present in every scene, so the scene-level split collapses but the
pair-level holdout must not.
"""
from pathlib import Path

import numpy as np

from affdet.schemas import Scene, SceneObject, Camera, Labels, LabelEntry, Features
from affdet.model.train import Trainer


def _corpus(tmp_path, n_scenes=24):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    for i in range(n_scenes):
        sid = f"scene_{i:04d}"
        ids = ["knife_001", f"bowl_{i:03d}", f"apple_{i:03d}"]
        objs = [SceneObject(obj_id=o, obj_class=o.rsplit("_", 1)[0],
                            bbox=[0, 0, 10, 10]) for o in ids]
        s = Scene(scene_id=sid, image_path="x.png", image_size=[128, 96],
                  source="blender", objects=objs,
                  camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0],
                                      [0, 0, 1, 0], [0, 0, 0, 1]],
                                intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        (scenes / f"{sid}.json").write_text(s.model_dump_json())
        Features(obj_ids=ids,
                 features=np.zeros((3, 768), dtype=np.float32)
                 ).save(feats / f"{sid}.npz")
        (labs / f"{sid}.json").write_text(Labels(
            scene_id=sid,
            tier1=[LabelEntry(actor="knife_001", recipient=f"apple_{i:03d}",
                              predicate="canCut", verdict=1, confidence=1.0,
                              source="test")],
        ).model_dump_json())
    return scenes, feats, labs


def test_fit_accepts_holdout_mode_on_a_hub_corpus(tmp_path):
    """Regression for the scene-level collapse: a scene-level split_strategy
    would put every scene in train and fail here (empty val). Trainer must
    route holdout_mode through to both its train and val AffordanceDatasets."""
    scenes, feats, labs = _corpus(tmp_path)
    ckpt = tmp_path / "ckpt"
    Trainer(epochs=1, seed=0, holdout_mode="strict", holdout_frac=0.30
           ).fit(scenes, feats, labs, ckpt)
    assert (ckpt / "best.pt").exists()


def test_fit_accepts_actor_only_holdout_mode(tmp_path):
    scenes, feats, labs = _corpus(tmp_path)
    ckpt = tmp_path / "ckpt"
    Trainer(epochs=1, seed=0, holdout_mode="actor_only", holdout_frac=0.30
           ).fit(scenes, feats, labs, ckpt)
    assert (ckpt / "best.pt").exists()


def test_holdout_mode_defaults_to_none_and_leaves_scene_split_unchanged(tmp_path):
    """Default behaviour must not move -- existing (non-holdout) runs must
    keep using split_strategy exactly as before."""
    assert Trainer().holdout_mode is None
    assert Trainer().holdout_frac == 0.20
