"""Classification scoring + zero-shot prompt vocabulary (E5).

Two things are pinned here:

1. ``prompt_vocabulary`` — the taxonomy uses snake_case class IDs
   (``red_wine_glass``). Fine-tuned detectors treat a class as a learned
   index and never read the string, but zero-shot detectors read it as
   English text, so the underscore form would measure their tolerance for
   underscores rather than their detection ability. The scored label must
   still be the canonical ID.

2. ``score_classification`` — the probe-vs-class-head comparison. Both
   methods must be scored over the SAME objects, so that the difference
   between them is the classifier and nothing else.
"""
import pytest

from affdet.eval.classification import (
    canonical_from_prompt,
    prompt_vocabulary,
    score_classification,
)


# --------------------------------------------------------------------------
# Zero-shot prompt vocabulary
# --------------------------------------------------------------------------

def test_prompt_vocabulary_despaces_snake_case():
    assert prompt_vocabulary(["red_wine_glass"]) == ["red wine glass"]


def test_prompt_vocabulary_leaves_plain_names_alone():
    assert prompt_vocabulary(["apple", "banana"]) == ["apple", "banana"]


def test_prompt_vocabulary_preserves_order():
    """Class order is the index mapping for a detector's output, so it must
    survive the rewrite."""
    classes = ["apple", "hot_dog", "banana", "sweet_potato"]
    assert prompt_vocabulary(classes) == [
        "apple", "hot dog", "banana", "sweet potato"]


def test_prompt_maps_back_to_canonical_id():
    """Scoring happens against the canonical class ID, never the prompt
    string -- otherwise the zero-shot rows would be scored against a
    different label space than the fine-tuned ones."""
    classes = ["red_wine_glass", "apple"]
    assert canonical_from_prompt("red wine glass", classes) == "red_wine_glass"
    assert canonical_from_prompt("apple", classes) == "apple"


def test_unknown_prompt_maps_to_none():
    """An open-vocabulary detector can emit text outside the taxonomy; that
    is a wrong answer, not a crash, and must not be silently coerced into a
    valid class."""
    assert canonical_from_prompt("spaceship", ["apple"]) is None


# --------------------------------------------------------------------------
# Probe vs class head
# --------------------------------------------------------------------------

def test_perfect_classifier_scores_one():
    r = score_classification(
        true=["knife", "bowl"],
        probe=["knife", "bowl"],
        classhead=["knife", "bowl"],
    )
    assert r["probe"]["accuracy"] == 1.0
    assert r["classhead"]["accuracy"] == 1.0
    assert r["n"] == 2


def test_probe_beating_classhead_is_reported_as_a_delta():
    r = score_classification(
        true=["knife", "bowl", "apple", "pear"],
        probe=["knife", "bowl", "apple", "pear"],
        classhead=["knife", "bowl", "apple", "lemon"],
    )
    assert r["probe"]["accuracy"] == 1.0
    assert r["classhead"]["accuracy"] == 0.75
    assert r["delta_accuracy"] == pytest.approx(0.25)


def test_mismatched_lengths_are_rejected():
    """Both methods must be scored on the identical object set -- a length
    mismatch means they are not, and silently zipping would compare
    different objects to each other."""
    with pytest.raises(ValueError, match="same objects"):
        score_classification(true=["knife", "bowl"], probe=["knife"],
                             classhead=["knife", "bowl"])


def test_per_class_accuracy_breakdown():
    r = score_classification(
        true=["knife", "knife", "bowl"],
        probe=["knife", "bowl", "bowl"],
        classhead=["knife", "knife", "knife"],
    )
    assert r["probe"]["per_class"]["knife"]["acc"] == pytest.approx(0.5)
    assert r["probe"]["per_class"]["bowl"]["acc"] == 1.0
    assert r["classhead"]["per_class"]["knife"]["acc"] == 1.0
    assert r["classhead"]["per_class"]["bowl"]["acc"] == 0.0


def test_confusions_are_recorded_for_error_analysis():
    """The interesting result is WHICH classes get confused (the shape-
    similar produce failure), so the pairs must survive scoring."""
    r = score_classification(
        true=["pear", "pear"],
        probe=["pear", "pear"],
        classhead=["lemon", "lemon"],
    )
    top = r["classhead"]["top_confusions"][0]
    assert top["true"] == "pear"
    assert top["pred"] == "lemon"
    assert top["count"] == 2
    assert r["probe"]["top_confusions"] == []


def test_none_prediction_counts_as_wrong_not_skipped():
    """A zero-shot detector emitting an out-of-vocabulary label got it
    wrong; dropping those rows would inflate its accuracy."""
    r = score_classification(
        true=["knife", "bowl"],
        probe=["knife", None],
        classhead=["knife", "bowl"],
    )
    assert r["probe"]["accuracy"] == 0.5
    assert r["n"] == 2


def test_empty_input_does_not_crash():
    r = score_classification(true=[], probe=[], classhead=[])
    assert r["n"] == 0
    assert r["probe"]["accuracy"] == 0.0
