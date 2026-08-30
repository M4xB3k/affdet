"""Tests for the coverage-aware sampling layer (affdet.render.omni_coverage).

Pure-Python: no Blender, no meshes. Verifies the label index buckets pos/neg
correctly, the tracker clamps targets to availability, and weighted selection
reaches rare affordances instead of starving them when total targets exceed the
scene budget.
"""
import json

import numpy as np
import pytest

from affdet.render.omni_coverage import (
    AffordancePairIndex, CoverageTracker, POS_MIN, NEG_MAX, POURINTO,
)


@pytest.fixture
def label_file(tmp_path):
    rows = []
    # canContain: many, mostly negative
    for i in range(30):
        rows.append({"actor": f"bowl_{i:03d}", "food": "apple", "affordance": "canContain",
                     "score": 8 if i < 10 else 1})
    # canPeel: rare
    for i, food in enumerate(["apple", "carrot", "cucumber", "mango"]):
        rows.append({"actor": "peeler_011", "food": food, "affordance": "canPeel",
                     "score": 9 if i < 3 else 1})
    # a few mid scores that must be skipped
    rows.append({"actor": "knife_001", "food": "apple", "affordance": "canCut", "score": 5})
    rows.append({"actor": "knife_001", "food": "banana", "affordance": "canCut", "score": 9})
    # a duplicate row (label file has these) — must be deduped
    rows.append({"actor": "knife_001", "food": "banana", "affordance": "canCut", "score": 9})
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_index_buckets_and_dedup(label_file):
    idx = AffordancePairIndex.from_jsonl(label_file)
    # mid score (5) skipped; duplicate canCut pos deduped to 1
    assert idx.available("canCut", "pos") == 1
    assert idx.available("canCut", "neg") == 0
    assert idx.available("canContain", "pos") == 10
    assert idx.available("canContain", "neg") == 20
    assert idx.available("canPeel", "pos") == 3
    assert idx.available("canPeel", "neg") == 1


def test_target_clamped_to_availability(label_file):
    idx = AffordancePairIndex.from_jsonl(label_file)
    trk = CoverageTracker(index=idx, n_scenes=1000)
    # canPeel pos target can't exceed its 3 available pairs
    assert trk.target["canPeel"]["pos"] == 3
    assert trk.target["canPeel"]["neg"] == 1
    # canPourInto is composition-driven, not clamped by the label index
    assert trk.target[POURINTO]["pos"] > 3


def test_weighted_selection_reaches_rare_affordance(label_file):
    idx = AffordancePairIndex.from_jsonl(label_file)
    trk = CoverageTracker(index=idx, n_scenes=500)
    rng = np.random.RandomState(0)
    # Simulate selection: each pick records one pair for that (aff, bucket).
    for _ in range(500):
        t = trk.weighted_target(rng)
        if t is None:
            break
        trk.record_pair(*t)
    # canPeel (rare) must have been reached, not starved by the big buckets.
    assert trk.pair_counts["canPeel"]["pos"] > 0


def test_weighted_target_none_when_all_met(label_file):
    idx = AffordancePairIndex.from_jsonl(label_file)
    trk = CoverageTracker(index=idx, n_scenes=1)
    rng = np.random.RandomState(0)
    # Drive every bucket to its target.
    for _ in range(10000):
        t = trk.weighted_target(rng)
        if t is None:
            break
        trk.record_pair(*t)
    assert trk.weighted_target(rng) is None
