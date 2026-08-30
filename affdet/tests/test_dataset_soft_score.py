import numpy as np
from pathlib import Path

from affdet.schemas import (Scene, SceneObject, Camera, Labels, LabelEntry,
                            Features)
from affdet.model.dataset import AffordanceDataset


def _setup(tmp_path: Path, score, use_score: bool):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0,0,10,10]),
            SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0,0,10,10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128,96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                            intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
    (scenes / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1","apple_1"],
             features=np.zeros((2,768), dtype=np.float32)).save(feats/"scene_0001.npz")
    entry = LabelEntry(actor="knife_1", recipient="apple_1", predicate="canCut",
                       verdict=1, confidence=0.9, source="t",
                       score=(score if use_score else None))
    (labs / "scene_0001.json").write_text(
        Labels(scene_id="scene_0001", tier1=[entry]).model_dump_json())
    return scenes, feats, labs


def _cut_target(ds):
    from affdet.intersection_matrix import TIER1_PREDICATES
    ci = TIER1_PREDICATES.index("canCut")
    for i in range(len(ds)):
        x, y, m = ds[i]
        if m[ci] > 0:
            return float(y[ci])
    return None


def test_soft_score_used_as_target(tmp_path):
    scenes, feats, labs = _setup(tmp_path, 0.72, use_score=True)
    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0)
    assert abs(_cut_target(ds) - 0.72) < 1e-6


def test_verdict_fallback_when_score_none(tmp_path):
    scenes, feats, labs = _setup(tmp_path, None, use_score=False)
    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0)
    assert _cut_target(ds) == 1.0
