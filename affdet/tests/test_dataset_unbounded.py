"""AffordanceDataset(unbounded=True) — train the model to actually reject
structurally impossible pairs (e.g. knife-canCut-knife) instead of never
seeing them.

Default mode excludes gated-out (actor, recipient, predicate) triples from
the loss entirely (mask=0), so the model has no signal there and produces
wild, overconfident guesses if ever asked about them at inference time (with
Inferencer(all_pairs=True)). Unbounded mode keeps every predicate in the mask
for every ordered pair, forcing y=0 (hard negative) wherever the pair isn't a
structural candidate and has no real label — the model now sees these as
explicit negatives during training.
"""
import numpy as np
import torch

from affdet.schemas import Scene, SceneObject, Camera, Labels, LabelEntry, Features
from affdet.model.dataset import AffordanceDataset
from affdet.intersection_matrix import TIER1_PREDICATES

_CUT_I = TIER1_PREDICATES.index("canCut")
_POUR_I = TIER1_PREDICATES.index("canPourInto")


def _write_two_knives(scenes_dir, feats_dir, labs_dir):
    objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0, 0, 10, 10]),
            SceneObject(obj_id="knife_2", obj_class="knife", bbox=[0, 0, 10, 10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128, 96],
              source="real", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (scenes_dir / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1", "knife_2"],
             features=np.zeros((2, 768), dtype=np.float32)).save(feats_dir / "scene_0001.npz")
    (labs_dir / "scene_0001.json").write_text(
        Labels(scene_id="scene_0001", tier1=[]).model_dump_json())


def test_default_mode_excludes_impossible_pair_from_mask_entirely(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_two_knives(scenes, feats, labs)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0)
    assert len(ds) == 0  # knife-knife has zero candidate predicates -> pair dropped entirely


def test_unbounded_mode_keeps_impossible_pair_as_hard_negative(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_two_knives(scenes, feats, labs)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0, unbounded=True)
    assert len(ds) == 2  # knife_1->knife_2 and knife_2->knife_1, each one pair-row

    for i in range(len(ds)):
        x, y, mask = ds[i]
        assert mask.sum() == len(TIER1_PREDICATES)  # every predicate now in-loss
        assert y[_CUT_I] == 0.0    # structurally impossible -> hard negative
        assert y[_POUR_I] == 0.0
    # candidate_masks stays the gate-only truth regardless of unbounded, so a
    # Trainer can still evaluate F1 on just the original candidate pairs.
    assert ds._candidate_masks.sum() == 0


def test_candidate_masks_equals_masks_when_not_unbounded(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0, 0, 10, 10]),
            SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0, 0, 10, 10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128, 96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (scenes / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1", "apple_1"],
             features=np.zeros((2, 768), dtype=np.float32)).save(feats / "scene_0001.npz")
    (labs / "scene_0001.json").write_text(Labels(scene_id="scene_0001", tier1=[]).model_dump_json())

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0)
    assert torch.equal(ds._masks, ds._candidate_masks)


def test_unbounded_mode_preserves_real_labels_where_they_exist(tmp_path):
    """A real positive label on a normally-gated-in pair must still come
    through as y=1 under unbounded mode, not get overwritten to 0."""
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    objs = [SceneObject(obj_id="knife_1", obj_class="knife", bbox=[0, 0, 10, 10]),
            SceneObject(obj_id="apple_1", obj_class="apple", bbox=[0, 0, 10, 10])]
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128, 96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (scenes / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1", "apple_1"],
             features=np.zeros((2, 768), dtype=np.float32)).save(feats / "scene_0001.npz")
    (labs / "scene_0001.json").write_text(Labels(scene_id="scene_0001", tier1=[
        LabelEntry(actor="knife_1", recipient="apple_1", predicate="canCut",
                   verdict=1, confidence=0.9, source="t", score=0.8),
    ]).model_dump_json())

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0, unbounded=True)
    assert len(ds) == 2
    x, y, mask = ds[0]  # knife_1 -> apple_1 (first pair emitted)
    assert mask.sum() == len(TIER1_PREDICATES)
    assert y[_CUT_I] == 0.8       # real soft-score label preserved
    assert y[_POUR_I] == 0.0      # apple can't canPourInto -> hard negative


def _write_scene_with_many_pairs(scenes_dir, feats_dir, labs_dir):
    """2 knives + 3 apples: knife->apple pairs are real candidates (canCut
    etc.); every other ordered pair (knife->knife, apple->apple, apple->knife)
    is structurally impossible. This scene has 6 candidate rows and 14
    impossible rows (20 ordered pairs total among 5 objects), enough headroom
    to test a subsample ratio cap."""
    objs = ([SceneObject(obj_id=f"knife_{i}", obj_class="knife", bbox=[0, 0, 10, 10]) for i in range(2)]
            + [SceneObject(obj_id=f"apple_{i}", obj_class="apple", bbox=[0, 0, 10, 10]) for i in range(3)])
    s = Scene(scene_id="scene_0001", image_path="x.png", image_size=[128, 96],
              source="real", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (scenes_dir / "scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=[o.obj_id for o in objs],
             features=np.zeros((5, 768), dtype=np.float32)).save(feats_dir / "scene_0001.npz")
    (labs_dir / "scene_0001.json").write_text(
        Labels(scene_id="scene_0001", tier1=[]).model_dump_json())


def test_unbounded_without_ratio_keeps_every_impossible_pair(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene_with_many_pairs(scenes, feats, labs)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0, unbounded=True)
    assert len(ds) == 20  # all 20 ordered pairs kept (6 candidate + 14 impossible)


def test_unbounded_neg_ratio_subsamples_impossible_pairs_per_scene(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene_with_many_pairs(scenes, feats, labs)

    ds = AffordanceDataset(scenes, feats, labs, split=None, seed=0,
                            unbounded=True, unbounded_neg_ratio=1)
    # 6 candidate rows kept in full; impossible rows capped at ratio(1) x 6 = 6.
    assert len(ds) == 12
    n_candidate = int((ds._candidate_masks.sum(dim=1) > 0).sum())
    n_impossible = len(ds) - n_candidate
    assert n_candidate == 6
    assert n_impossible == 6


def test_unbounded_neg_ratio_is_reproducible_for_same_seed(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    feats = tmp_path / "feats"; feats.mkdir()
    labs = tmp_path / "labs"; labs.mkdir()
    _write_scene_with_many_pairs(scenes, feats, labs)

    a = AffordanceDataset(scenes, feats, labs, split=None, seed=3,
                           unbounded=True, unbounded_neg_ratio=1)
    b = AffordanceDataset(scenes, feats, labs, split=None, seed=3,
                           unbounded=True, unbounded_neg_ratio=1)
    assert torch.equal(a._masks, b._masks)
    assert torch.equal(a._xs, b._xs)
