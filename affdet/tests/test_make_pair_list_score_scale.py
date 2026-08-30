"""real_run()'s band logic (border/pos/neg) must match the ACTUAL scale
load_labels() returns: normalized 0-1 (score/9, per affdet.schemas.LabelEntry
and affdet/label/vlm_resolver.py), not the raw 0-9 VLM prompt scale.

VLM_THRESHOLD=5 and the border band 4<=score<=6 were written against the raw
scale (matching the OLDER labels_omni.jsonl corpus, which stored raw scores
like 9, 8, 8). The real corpus (labels_all_merged_v3) stores already-
normalized 0-1 scores, so with the raw-scale constants every single label
falls below threshold and gets banded "neg" regardless of its true verdict --
verified directly: a verdict=1, score=1.0 pair was banded "neg" because
1.0 < 5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.make_pair_list import score_band


def test_high_normalized_score_is_positive():
    """score=1.0 (verdict=1, VLM said yes) must NOT be banded 'neg'."""
    assert score_band(1.0) == "pos"


def test_low_normalized_score_is_negative():
    assert score_band(0.0) == "neg"
    assert score_band(0.2222222222222222) == "neg"


def test_mid_normalized_score_is_border():
    """Raw band was 4<=score<=6 out of 9 -> normalized ~0.444-0.667."""
    assert score_band(0.5) == "border"


def test_band_boundaries_are_on_the_normalized_scale():
    assert score_band(4 / 9) == "border"
    assert score_band(4 / 9 - 0.01) == "neg"
    assert score_band(6 / 9) == "border"
    assert score_band(6 / 9 + 0.01) == "pos"
