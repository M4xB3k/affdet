"""H5 — split strategies for the generalization ladder (E1).

The corpus splits by SCENE, which controls only the weakest generalization
claim. Measured on the real corpus: 30.6% of test (actor, recipient,
predicate) triples also occur in train, and 100.00% of test labels involve a
mesh instance that was also seen in train -- so the reported metric contains
no instance-novel evidence at all.

Three strategies let the same model be evaluated at three levels, and the
DEGRADATION across them is the finding:
  scene     -- unseen arrangement  (status quo)
  pair      -- unseen (actor, recipient) combination
  instance  -- unseen mesh instance ("obj_id", e.g. bowl_016)
"""
import json

import pytest

from affdet.model.dataset import (
    scene_split_map, pair_split_map, instance_split_map, build_split_map,
)


def _write_scene(d, scene_id, objects):
    p = d / scene_id
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{scene_id}.json").write_text(json.dumps({
        "scene_id": scene_id,
        "image_path": f"{scene_id}.png",
        "width": 640, "height": 480,
        "objects": [{"obj_id": o, "obj_class": o.rsplit("_", 1)[0],
                     "bbox": [0, 0, 10, 10], "mask_path": f"{o}_mask.png"}
                    for o in objects],
    }))


@pytest.fixture
def corpus(tmp_path):
    """30 scenes drawn from 10 disjoint mesh cliques, each mesh reused across
    3 scenes.

    Mirrors the real corpus in the property that matters -- instances recur
    across scenes, so a scene-level split leaks them -- while keeping the
    scene/mesh graph separable. A pool small enough that every mesh co-occurs
    with every other collapses into a single connected component that cannot
    be split at all, which is a real constraint of instance-disjoint
    splitting, not a bug (see the corpus-level feasibility analysis: the
    binding limit is actor-instance poverty).
    """
    d = tmp_path / "scenes"
    for clique in range(10):
        for rep in range(3):
            _write_scene(d, f"scene_{clique * 3 + rep:04d}",
                         [f"knife_{clique:03d}", f"apple_{clique:03d}"])
    return d


def test_instance_split_is_actually_instance_disjoint(corpus):
    """The whole point: no mesh may appear on both sides of the split."""
    m = instance_split_map(corpus, seed=0)
    by_split = {}
    for scene_id, sp in m.items():
        objs = json.loads(
            (corpus / scene_id / f"{scene_id}.json").read_text())["objects"]
        by_split.setdefault(sp, set()).update(o["obj_id"] for o in objs)
    train, test = by_split.get("train", set()), by_split.get("test", set())
    assert train and test, "split produced an empty side"
    assert not (train & test), f"instances leaked across split: {train & test}"


def test_scene_split_is_not_instance_disjoint(corpus):
    """Documents the status quo the ladder is measured against: the scene
    split reuses meshes on both sides."""
    m = scene_split_map(corpus, seed=0)
    by_split = {}
    for scene_id, sp in m.items():
        objs = json.loads(
            (corpus / scene_id / f"{scene_id}.json").read_text())["objects"]
        by_split.setdefault(sp, set()).update(o["obj_id"] for o in objs)
    assert by_split["train"] & by_split["test"], (
        "expected the scene split to leak instances -- if this fails the "
        "corpus fixture no longer reuses meshes and the test is meaningless")


def test_pair_split_is_pair_disjoint(corpus):
    """No unordered mesh pair may co-occur on both sides."""
    m = pair_split_map(corpus, seed=0)
    by_split = {}
    for scene_id, sp in m.items():
        objs = json.loads(
            (corpus / scene_id / f"{scene_id}.json").read_text())["objects"]
        ids = sorted(o["obj_id"] for o in objs)
        pairs = {(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]}
        by_split.setdefault(sp, set()).update(pairs)
    train, test = by_split.get("train", set()), by_split.get("test", set())
    assert train and test
    assert not (train & test), f"pairs leaked: {train & test}"


def test_split_maps_are_deterministic(corpus):
    for fn in (scene_split_map, pair_split_map, instance_split_map):
        assert fn(corpus, seed=0) == fn(corpus, seed=0)


def test_split_seed_changes_assignment(corpus):
    assert instance_split_map(corpus, seed=0) != instance_split_map(corpus, seed=7)


def test_build_split_map_dispatches_by_strategy(corpus):
    assert build_split_map(corpus, "scene", 0) == scene_split_map(corpus, 0)
    assert build_split_map(corpus, "pair", 0) == pair_split_map(corpus, 0)
    assert build_split_map(corpus, "instance", 0) == instance_split_map(corpus, 0)


def test_build_split_map_rejects_unknown_strategy(corpus):
    with pytest.raises(ValueError, match="unknown split strategy"):
        build_split_map(corpus, "nonsense", 0)


def test_every_scene_is_assigned(corpus):
    """No scene may be silently dropped -- a scene absent from the map would
    vanish from training and evaluation alike."""
    n = len(list(corpus.glob("scene_*/scene_*.json")))
    for fn in (scene_split_map, pair_split_map, instance_split_map):
        m = fn(corpus, seed=0)
        assert len(m) == n, f"{fn.__name__} assigned {len(m)}/{n} scenes"
        assert set(m.values()) <= {"train", "val", "test"}
