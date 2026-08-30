"""Balanced scene sampling for the list-all training experiment.

The draw must be balanced across (affordance, polarity) slots and, above all,
must never touch the E7 evaluation corpus. A leak there would silently make the
retrained model's E7 score meaningless, and the failure would be invisible in
the metrics -- the model would simply look better than it is.
"""
import pytest

from affdet.eval.listall_sample import balanced_sample, slot_counts


def _scene(sid, slots, n_objects=3):
    return {"scene_id": sid, "realised_affordances": slots,
            "n_objects": n_objects}


def _corpus(n=60):
    """Scenes cycling through three slots, so balance is checkable."""
    slots = [[["canCut", "pos"]], [["canCut", "neg"]], [["canContain", "pos"]]]
    return [_scene(f"scene_v1_{i:04d}", slots[i % 3]) for i in range(n)]


def test_sample_returns_requested_size():
    got = balanced_sample(_corpus(60), n=30, seed=0)
    assert len(got) == 30


def test_sample_is_deterministic_under_seed():
    a = balanced_sample(_corpus(60), n=30, seed=7)
    b = balanced_sample(_corpus(60), n=30, seed=7)
    assert [s["scene_id"] for s in a] == [s["scene_id"] for s in b]


def test_different_seeds_give_different_draws():
    a = balanced_sample(_corpus(60), n=30, seed=1)
    b = balanced_sample(_corpus(60), n=30, seed=2)
    assert [s["scene_id"] for s in a] != [s["scene_id"] for s in b]


def test_sample_never_repeats_a_scene():
    got = balanced_sample(_corpus(60), n=45, seed=0)
    ids = [s["scene_id"] for s in got]
    assert len(ids) == len(set(ids))


def test_sample_balances_across_slots():
    """Each (affordance, polarity) slot should be represented roughly equally,
    not in the proportion it happens to occur in the corpus."""
    got = balanced_sample(_corpus(90), n=30, seed=0)
    counts = slot_counts(got)
    assert set(counts) == {("canCut", "pos"), ("canCut", "neg"),
                           ("canContain", "pos")}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi - lo <= 1, counts


def test_excluded_scene_ids_are_never_drawn():
    """The E7 eval corpus must not leak into training. This is the test that
    matters most: a leak is undetectable downstream."""
    corpus = _corpus(60)
    banned = {s["scene_id"] for s in corpus[:30]}
    got = balanced_sample(corpus, n=25, seed=0, exclude=banned)
    assert banned.isdisjoint({s["scene_id"] for s in got})


def test_requesting_more_than_available_returns_everything():
    got = balanced_sample(_corpus(10), n=999, seed=0)
    assert len(got) == 10


def test_requesting_more_than_available_after_exclusion():
    corpus = _corpus(10)
    banned = {s["scene_id"] for s in corpus[:6]}
    got = balanced_sample(corpus, n=999, seed=0, exclude=banned)
    assert len(got) == 4


def test_scene_with_no_realised_affordances_is_skipped():
    """A scene with nothing realised carries no training signal for any slot,
    so it should not consume a draw."""
    corpus = _corpus(30) + [_scene("scene_v1_9999", [])]
    got = balanced_sample(corpus, n=31, seed=0)
    assert "scene_v1_9999" not in {s["scene_id"] for s in got}


def test_multi_slot_scene_counts_toward_every_slot_it_realises():
    corpus = [_scene("scene_v1_0001", [["canCut", "pos"], ["canPeel", "neg"]])]
    counts = slot_counts(corpus)
    assert counts[("canCut", "pos")] == 1
    assert counts[("canPeel", "neg")] == 1


def test_negative_n_raises():
    with pytest.raises(ValueError):
        balanced_sample(_corpus(10), n=-1, seed=0)
