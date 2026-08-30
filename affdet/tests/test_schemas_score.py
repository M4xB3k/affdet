from affdet.schemas import LabelEntry
import pytest


def test_label_entry_accepts_soft_score():
    e = LabelEntry(actor="knife_1", recipient="apple_1", predicate="canCut",
                   verdict=1, confidence=0.9, source="vlm", score=0.78)
    assert e.score == pytest.approx(0.78)


def test_label_entry_score_defaults_none():
    e = LabelEntry(actor="knife_1", recipient="apple_1", predicate="canCut",
                   verdict=1, confidence=0.9, source="vlm")
    assert e.score is None


def test_label_entry_score_out_of_range_rejected():
    with pytest.raises(Exception):
        LabelEntry(actor="a", recipient="b", predicate="canCut", verdict=1,
                   confidence=0.9, source="vlm", score=1.5)
