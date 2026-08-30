"""AffordanceDataset(holdout_mode=...) — the pair-level instance split, end to end.

`test_instance_holdout.py` covers the routing rule in isolation. This covers the
WIRING: that AffordanceDataset actually keeps every scene and filters per pair,
which is the part the scene-level implementation got wrong (it assigned whole
scenes and produced an empty test set on the real corpus).

The hub fixture reproduces that pathology: one mesh present in every scene.
"""
import numpy as np

from affdet.schemas import Scene, SceneObject, Camera, Labels, LabelEntry, Features
from affdet.model.dataset import AffordanceDataset, instance_holdout


def _corpus(tmp_path, n_scenes=24):
    """Every scene holds the same hub knife plus a scene-unique bowl and apple.

    Scene-level union-find sees one component here and cannot split it at all.
    """
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


def test_test_split_is_not_empty(tmp_path):
    """The regression. The scene-level instance split returns zero test rows on
    a hub corpus; the pair-level one must not."""
    scenes, feats, labs = _corpus(tmp_path)
    ds = AffordanceDataset(scenes, feats, labs, split="test", seed=0,
                           holdout_mode="strict", holdout_frac=0.30)
    assert len(ds) > 0


def test_train_and_test_are_both_populated(tmp_path):
    scenes, feats, labs = _corpus(tmp_path)
    tr = AffordanceDataset(scenes, feats, labs, split="train", seed=0,
                           holdout_mode="strict", holdout_frac=0.30)
    te = AffordanceDataset(scenes, feats, labs, split="test", seed=0,
                           holdout_mode="strict", holdout_frac=0.30)
    assert len(tr) > 0 and len(te) > 0


def test_actor_only_mode_populates_both_sides(tmp_path):
    scenes, feats, labs = _corpus(tmp_path)
    tr = AffordanceDataset(scenes, feats, labs, split="train", seed=0,
                           holdout_mode="actor_only", holdout_frac=0.30)
    te = AffordanceDataset(scenes, feats, labs, split="test", seed=0,
                           holdout_mode="actor_only", holdout_frac=0.30)
    assert len(tr) > 0 and len(te) > 0


def test_holdout_is_recorded_on_the_dataset(tmp_path):
    """The runner needs to persist which meshes were drawn (H2)."""
    scenes, feats, labs = _corpus(tmp_path)
    ds = AffordanceDataset(scenes, feats, labs, split="test", seed=0,
                           holdout_mode="strict", holdout_frac=0.30)
    assert ds._holdout is not None
    assert ds._holdout.test_meshes


def test_scene_strategy_is_unchanged_when_holdout_is_off(tmp_path):
    """Default behaviour must not move -- existing results stay reproducible."""
    scenes, feats, labs = _corpus(tmp_path)
    a = AffordanceDataset(scenes, feats, labs, split="train", seed=0)
    b = AffordanceDataset(scenes, feats, labs, split="train", seed=0,
                          split_strategy="scene")
    assert len(a) == len(b) > 0
    assert AffordanceDataset(scenes, feats, labs, split="train", seed=0
                             )._holdout is None
