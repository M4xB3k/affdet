"""instance_holdout must stratify on EFFECTIVE class, not the id prefix.

The corpus has 27 `knife_*` meshes, of which exactly four (knife_011/014/018/
021) are recategorised as peelers -- and those four are canPeel's only actors.
Stratifying by id prefix draws k=5 of the 27 knives at frac=0.20, so the
hypergeometric chance that no peeler lands in test is comb(23,5)/comb(27,5) =
41.7% per seed. Observed on the v4 ladder: seed 0's test knives were
002/003/024/025/030 -- no peeler -- so canPeel had ZERO test pairs and its F1
was a placeholder, not a score. Only 19.8% of 3-seed runs get all three seeds
populated, and 7.2% lose the predicate outright.

Stratifying on effective class puts the four peelers in their own stratum, so
`max(1, ...)` guarantees one in test and one in val on every seed.
"""
from pathlib import Path

import pytest

from affdet.model.dataset import instance_holdout

PEELERS = {"knife_011", "knife_014", "knife_018", "knife_021"}


def _corpus(tmp_path: Path, n_knives: int = 27) -> Path:
    """Minimal scene tree mirroring the real knife/peeler imbalance."""
    import json

    scenes = tmp_path / "scenes"
    knives = [f"knife_{i:03d}" for i in range(1, n_knives + 1)]
    foods = [f"apple_{i:03d}" for i in range(1, 9)]
    for i, (k, f) in enumerate(
            [(k, f) for k in knives for f in foods]):
        d = scenes / "worker_0" / f"scene_{i:05d}"
        d.mkdir(parents=True)
        (d / f"scene_{i:05d}.json").write_text(json.dumps(
            {"objects": [{"obj_id": k}, {"obj_id": f}]}))
    return scenes


def _eff_map() -> dict[str, str]:
    """knife_011/014/018/021 are peelers; every other knife stays a knife."""
    return {m: "peeler" for m in PEELERS}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_every_seed_holds_out_at_least_one_peeler(tmp_path, seed):
    scenes = _corpus(tmp_path)
    h = instance_holdout(scenes, seed=seed, frac=0.20,
                         effective_classes=_eff_map())
    assert PEELERS & set(h.test_meshes), (
        f"seed {seed}: no peeler in test -- canPeel would have zero test pairs")
    assert PEELERS & set(h.val_meshes), (
        f"seed {seed}: no peeler in val -- nothing to calibrate canPeel on")


def test_peelers_are_not_in_both_arms(tmp_path):
    h = instance_holdout(_corpus(tmp_path), seed=0, frac=0.20,
                         effective_classes=_eff_map())
    assert not (h.test_meshes & h.val_meshes)


def test_plain_knives_still_held_out(tmp_path):
    """Splitting the stratum must not starve the knife side of the split."""
    h = instance_holdout(_corpus(tmp_path), seed=0, frac=0.20,
                         effective_classes=_eff_map())
    plain = {m for m in h.test_meshes
             if m.startswith("knife_") and m not in PEELERS}
    assert plain, "no ordinary knife held out -- canCut loses its test arm"


def test_without_effective_classes_behaviour_is_unchanged(tmp_path):
    """Omitting the map must reproduce the old id-prefix stratification, so
    existing runs and their committed numbers stay reproducible."""
    scenes = _corpus(tmp_path)
    before = instance_holdout(scenes, seed=0, frac=0.20)
    after = instance_holdout(scenes, seed=0, frac=0.20, effective_classes=None)
    assert before.test_meshes == after.test_meshes
    assert before.val_meshes == after.val_meshes
