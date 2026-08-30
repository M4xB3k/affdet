"""Converting volunteered list-all output into a training label corpus.

The training corpus (`labels_omni_v4.jsonl`) is keyed on
(actor, recipient, predicate) with NO scene id -- labels are instance-pair
level. list-all output is scene-scoped, so the same instance pair can be
reported in several scenes at different sizes and poses, and those must be
reconciled into one score.

Two things must not go wrong:

1. **Structurally impossible pairs must be filtered.** list-all volunteers ~50 %
   junk; training on it would teach the MLP about pairs the gate will never
   present at inference.
2. **Multi-pass votes must be reconciled before cross-scene aggregation**, not
   after, or a pair seen in three scenes with three passes each would weight
   scenes by how often the VLM happened to mention them.
"""
import pytest

from affdet.eval.listall_labels import (
    build_label_corpus,
    reconcile_scene_scores,
)


def _rec(scene_id, scores):
    return {"scene_id": scene_id,
            "scores": [{"actor": a, "recipient": r, "predicate": p,
                        "score": s} for a, r, p, s in scores]}


CLS = {"knife_1": "knife", "apple_1": "apple", "bowl_1": "bowl",
       "banana_1": "banana"}


# --------------------------------------------------------------------------
# Per-pass reconciliation
# --------------------------------------------------------------------------

def test_reconcile_takes_median_across_passes_within_a_scene():
    passes = [
        [_rec("s1", [("knife_1", "apple_1", "canCut", 8)])],
        [_rec("s1", [("knife_1", "apple_1", "canCut", 2)])],
        [_rec("s1", [("knife_1", "apple_1", "canCut", 7)])],
    ]
    got = reconcile_scene_scores(passes)
    assert got[("s1", "knife_1", "apple_1", "canCut")] == 7.0


def test_reconcile_keeps_a_pair_seen_in_only_one_pass():
    passes = [
        [_rec("s1", [("knife_1", "apple_1", "canCut", 8)])],
        [_rec("s1", [])],
        [_rec("s1", [])],
    ]
    got = reconcile_scene_scores(passes)
    assert got[("s1", "knife_1", "apple_1", "canCut")] == 8.0


def test_reconcile_keeps_scenes_separate():
    """The same pair in two scenes is two observations, not one."""
    passes = [[
        _rec("s1", [("knife_1", "apple_1", "canCut", 8)]),
        _rec("s2", [("knife_1", "apple_1", "canCut", 3)]),
    ]]
    got = reconcile_scene_scores(passes)
    assert got[("s1", "knife_1", "apple_1", "canCut")] == 8.0
    assert got[("s2", "knife_1", "apple_1", "canCut")] == 3.0


# --------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------

def test_build_filters_structurally_impossible_pairs():
    """apple cutting a knife is never a candidate; it must not reach training."""
    passes = [[_rec("s1", [
        ("knife_1", "apple_1", "canCut", 8),
        ("apple_1", "knife_1", "canCut", 6),
    ])]]
    rows = build_label_corpus(passes, CLS)
    keys = {(r["actor"], r["recipient"], r["predicate"]) for r in rows}
    assert ("knife_1", "apple_1", "canCut") in keys
    assert ("apple_1", "knife_1", "canCut") not in keys


def test_build_aggregates_the_same_pair_across_scenes():
    """One row per (actor, recipient, predicate), matching labels_omni_v4."""
    passes = [[
        _rec("s1", [("knife_1", "apple_1", "canCut", 9)]),
        _rec("s2", [("knife_1", "apple_1", "canCut", 5)]),
        _rec("s3", [("knife_1", "apple_1", "canCut", 7)]),
    ]]
    rows = build_label_corpus(passes, CLS)
    assert len(rows) == 1
    assert rows[0]["score"] == 7          # median of 9, 5, 7


def test_build_records_how_many_scenes_backed_each_label():
    passes = [[
        _rec("s1", [("knife_1", "apple_1", "canCut", 9)]),
        _rec("s2", [("knife_1", "apple_1", "canCut", 5)]),
    ]]
    rows = build_label_corpus(passes, CLS)
    assert rows[0]["n_scenes"] == 2


def test_build_emits_the_v4_row_shape():
    passes = [[_rec("s1", [("knife_1", "apple_1", "canCut", 8)])]]
    row = build_label_corpus(passes, CLS)[0]
    for k in ("actor", "recipient", "predicate", "score"):
        assert k in row
    assert isinstance(row["score"], (int, float))


def test_build_drops_pairs_with_unknown_classes():
    """An obj_id absent from the class map cannot be gate-checked, so it is
    dropped rather than let through unchecked."""
    passes = [[_rec("s1", [("ghost_9", "apple_1", "canCut", 8)])]]
    assert build_label_corpus(passes, CLS) == []


def test_build_is_deterministic_in_row_order():
    passes = [[_rec("s1", [
        ("knife_1", "apple_1", "canCut", 8),
        ("bowl_1", "banana_1", "canContain", 9),
    ])]]
    a = build_label_corpus(passes, CLS)
    b = build_label_corpus(passes, CLS)
    assert [(r["actor"], r["recipient"], r["predicate"]) for r in a] == \
           [(r["actor"], r["recipient"], r["predicate"]) for r in b]


def test_build_on_empty_input_is_empty():
    assert build_label_corpus([], CLS) == []


def test_scores_stay_on_the_0_10_scale():
    """v4 labels are 0-10 integers; the distillation path rescales. Emitting a
    0-1 value here would silently shrink every target by 10x."""
    passes = [[_rec("s1", [("knife_1", "apple_1", "canCut", 8)])]]
    row = build_label_corpus(passes, CLS)[0]
    assert 0 <= row["score"] <= 10


def test_min_scenes_filter_drops_thinly_supported_pairs():
    passes = [[
        _rec("s1", [("knife_1", "apple_1", "canCut", 8)]),
        _rec("s2", [("bowl_1", "banana_1", "canContain", 9)]),
        _rec("s3", [("bowl_1", "banana_1", "canContain", 8)]),
    ]]
    rows = build_label_corpus(passes, CLS, min_scenes=2)
    keys = {(r["actor"], r["recipient"], r["predicate"]) for r in rows}
    assert ("bowl_1", "banana_1", "canContain") in keys
    assert ("knife_1", "apple_1", "canCut") not in keys


def test_negative_min_scenes_raises():
    with pytest.raises(ValueError):
        build_label_corpus([], CLS, min_scenes=0)
