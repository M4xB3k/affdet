"""H5 fix — instance holdout routed at the PAIR level, not the scene level.

The scene-level union-find split (`instance_split_map`) is exactly disjoint but
degenerates on the real corpus: `fork_003` appears in 1,478 of 12,753 scenes,
`fork_001` in 1,460, `fork_002` in 1,387, and with only 3 fork meshes nearly
every scene contains one. Union-find chains the whole corpus into a single
connected component, so all 12,753 scenes land in "train" and test is EMPTY.

The unit of evaluation is the PAIR, not the scene. So instance holdout is
expressed as a set of held-out mesh instances plus a per-pair routing rule:
a scene may contribute some pairs to train and others to test.

Two holdout modes (see tools/docs/briefs/experiments_findings.md §3):
  strict      E1c  -- BOTH meshes novel; the "unseen instance" claim
  actor_only  E1c' -- actor mesh novel; the "unseen tool" claim
"""
import json

import pytest

from affdet.model.dataset import (
    instance_holdout, pair_split_for, InstanceHoldout,
)


def _scene(d, scene_id, objects):
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
def hub_corpus(tmp_path):
    """The real corpus's pathology in miniature: `fork_001` is in EVERY scene,
    so any scene-level instance-disjoint split must collapse to one component.
    """
    for i in range(20):
        _scene(tmp_path, f"scene_{i:04d}",
               ["fork_001", f"bowl_{i:03d}", f"apple_{i:03d}"])
    return tmp_path


def test_holdout_is_nonempty_on_a_hub_corpus(hub_corpus):
    """The regression this whole module exists for: a hub mesh must not be
    able to starve the test side."""
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    assert ho.test_meshes, "no meshes held out for test"
    assert ho.val_meshes, "no meshes held out for val"


def test_strict_routes_only_fully_novel_pairs_to_test(hub_corpus):
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    for a in ("bowl_000", "apple_000", "fork_001"):
        for b in ("bowl_001", "apple_001", "fork_001"):
            if a == b:
                continue
            got = pair_split_for(a, b, ho, mode="strict")
            if got == "test":
                assert a in ho.test_meshes and b in ho.test_meshes


def test_strict_train_never_touches_a_held_out_mesh(hub_corpus):
    """Train must be clean: if EITHER mesh is held out, the pair cannot train.
    Otherwise the model sees the "novel" mesh and the claim is void."""
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    held = ho.test_meshes | ho.val_meshes
    for a in sorted(ho.all_meshes):
        for b in sorted(ho.all_meshes):
            if a == b:
                continue
            if pair_split_for(a, b, ho, mode="strict") == "train":
                assert a not in held and b not in held


def test_actor_only_mode_ignores_recipient_novelty(hub_corpus):
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    a = sorted(ho.test_meshes)[0]
    seen = sorted(ho.all_meshes - ho.test_meshes - ho.val_meshes)[0]
    assert pair_split_for(a, seen, ho, mode="actor_only") == "test"
    # Actor seen => not a test pair, whatever the recipient is.
    assert pair_split_for(seen, a, ho, mode="actor_only") != "test"


def test_actor_only_train_never_uses_a_held_out_actor(hub_corpus):
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    held = ho.test_meshes | ho.val_meshes
    for a in sorted(ho.all_meshes):
        for b in sorted(ho.all_meshes):
            if a == b:
                continue
            if pair_split_for(a, b, ho, mode="actor_only") == "train":
                assert a not in held


def test_val_and_test_meshes_are_disjoint(hub_corpus):
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    assert not (ho.test_meshes & ho.val_meshes)


def test_holdout_is_deterministic_for_a_seed(hub_corpus):
    a = instance_holdout(hub_corpus, seed=3, frac=0.2)
    b = instance_holdout(hub_corpus, seed=3, frac=0.2)
    assert a.test_meshes == b.test_meshes and a.val_meshes == b.val_meshes


def test_different_seeds_give_different_holdouts(hub_corpus):
    """Draw variance is the point: for low-instance predicates it dominates
    training-seed variance, so each seed must resample WHICH meshes are held
    out (findings doc §3)."""
    seeds = {frozenset(instance_holdout(hub_corpus, seed=s, frac=0.2).test_meshes)
             for s in range(5)}
    assert len(seeds) > 1


def test_holdout_fraction_is_respected(hub_corpus):
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    n = len(ho.all_meshes)
    assert 0.5 * 0.2 * n <= len(ho.test_meshes) <= 1.6 * 0.2 * n


def test_larger_fraction_holds_out_more(hub_corpus):
    small = instance_holdout(hub_corpus, seed=0, frac=0.15)
    big = instance_holdout(hub_corpus, seed=0, frac=0.30)
    assert len(big.test_meshes) > len(small.test_meshes)


def test_unknown_mode_raises(hub_corpus):
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    with pytest.raises(ValueError):
        pair_split_for("bowl_000", "apple_000", ho, mode="nonsense")


def test_small_categories_still_get_held_out(hub_corpus):
    """Stratifying by category is what stops a big category (bread has 156
    meshes) from soaking up the whole draw and leaving actor categories with
    nothing novel. fork has 3 meshes and pan 4 on the real corpus."""
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    cats = {m.rsplit("_", 1)[0] for m in ho.test_meshes}
    assert len(cats) > 1, "holdout drew from only one category"


def test_single_instance_category_is_never_held_out(tmp_path):
    """A category with one mesh cannot contribute a novel instance without
    disappearing from train entirely -- it must stay in train."""
    for i in range(6):
        _scene(tmp_path, f"scene_{i:04d}", ["solo_001", f"bowl_{i:03d}"])
    ho = instance_holdout(tmp_path, seed=0, frac=0.2)
    assert "solo_001" not in ho.test_meshes
    assert "solo_001" not in ho.val_meshes


def test_holdout_round_trips_through_dict(hub_corpus):
    """The ledger has to record WHICH meshes were held out, or a run is not
    reproducible (H2)."""
    ho = instance_holdout(hub_corpus, seed=0, frac=0.2)
    back = InstanceHoldout.from_dict(ho.to_dict())
    assert back.test_meshes == ho.test_meshes
    assert back.val_meshes == ho.val_meshes
    assert back.all_meshes == ho.all_meshes
