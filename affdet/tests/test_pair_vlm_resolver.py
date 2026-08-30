"""Tests for PairVlmResolver: resolves scene (actor_instance, recipient_instance,
predicate) triples to soft targets using the NEW pair-labeling VLM corpus
(labels_omni.jsonl schema: {actor, recipient, predicate, score} where actor
and recipient are both specific instance ids, e.g. "bowl_001"/"apple_017" --
NOT the old vlm_resolver.py schema of {actor, food (a class name), affordance}).

Lookup key is (actor_instance, recipient_CATEGORY, PLACEMENT, predicate),
averaged across every labeled recipient instance sharing that category+
placement (validated 2026-07-13, 92.2% coverage on omni_scenes_v3_merged --
see check_coverage_tmp.py on the cluster). Placement (whole/piece/opened)
comes from curation.json's per-category `placement` block: labeling only a
sample of instances per category+placement is enough signal for every other
instance in that same bucket, since VLM judgments track visual state
(whole apple vs cut piece) more than exact instance identity.

canPourInto remains rule-based (unchanged from vlm_resolver.py): it has no
real rows in labels_omni.jsonl to look up. canPierce no longer does -- it is
VLM-judged as of 72d0ef6 and resolves through the ordinary score lookup.
"""
import pytest

from affdet.label.pair_vlm_resolver import PairVlmResolver


def _curation():
    return {
        "apple": {"placement": {"whole": ["001", "002"], "piece": ["017"]}},
        "bowl": {},
    }


def _resolved():
    return {
        "knife_001": {"effective_class": "knife", "dropped": False,
                      "pourable_from": None, "pourable_into": None},
        "knife_011": {"effective_class": "peeler", "dropped": False,
                      "pourable_from": None, "pourable_into": None},
        "pan_001": {"effective_class": "pan", "dropped": True,
                    "pourable_from": None, "pourable_into": None},
        "teapot_014": {"effective_class": "teapot", "dropped": False,
                       "pourable_from": True, "pourable_into": True},
        "bottle_001": {"effective_class": "bottle", "dropped": False,
                       "pourable_from": False, "pourable_into": False},
        "bowl_003": {"effective_class": "bowl", "dropped": False,
                     "pourable_from": False, "pourable_into": True},
    }


def _rows():
    return [
        {"actor": "knife_001", "recipient": "apple_001", "predicate": "canCut", "score": 9},
        {"actor": "knife_001", "recipient": "apple_002", "predicate": "canCut", "score": 7},
        {"actor": "knife_001", "recipient": "apple_017", "predicate": "canCut", "score": 3},
        {"actor": "peeler_011", "recipient": "apple_001", "predicate": "canPeel", "score": 6},
    ]


def _res():
    return PairVlmResolver.from_dicts(_rows(), _resolved(), _curation())


def test_averages_scores_across_same_category_same_placement_instances():
    # apple_001 and apple_002 are both "whole" -> the bucket-wide average IS
    # the teacher signal (not a fallback for missing rows): mean(9, 7) = 8
    # -> 8/9, regardless of which "whole" apple is queried.
    assert _res().soft_target("knife_001", "knife", "apple",
                              "canCut", "apple_001") == pytest.approx(8 / 9)


def test_same_bucket_average_reused_for_sibling_instance():
    # apple_002 gets the identical bucket average as apple_001 above -- but
    # must NOT pull in apple_017 (a different placement bucket, "piece").
    assert _res().soft_target("knife_001", "knife", "apple",
                              "canCut", "apple_002") == pytest.approx(8 / 9)


def test_piece_placement_uses_separate_bucket():
    assert _res().soft_target("knife_001", "knife", "apple",
                              "canCut", "apple_017") == pytest.approx(3 / 9)


def test_unresolved_placement_returns_none():
    # apple_099 not listed under any placement in curation -> no bucket.
    assert _res().soft_target("knife_001", "knife", "apple",
                              "canCut", "apple_099") is None


def test_dropped_actor_returns_none():
    assert _res().soft_target("pan_001", "pan", "apple",
                              "canSupport", "apple_001") is None


def test_effective_class_remaps_peeler_for_lookup():
    # scene actor is knife_011 but score rows are keyed peeler_011.
    assert _res().soft_target("knife_011", "knife", "apple",
                              "canPeel", "apple_001") == pytest.approx(6 / 9)


def test_no_matching_rows_returns_none():
    assert _res().soft_target("knife_001", "knife", "apple",
                              "canPierce_typo", "apple_001") is None


def test_canpourinto_stays_rule_based_not_looked_up_in_vlm_rows():
    # teapot_014 pours from AND bowl_003 receives into -> 1.0, purely from
    # resolved_instances flags, no labels_omni.jsonl row needed.
    assert _res().soft_target("teapot_014", "teapot", "bowl",
                              "canPourInto", "bowl_003") == 1.0


def test_canpourinto_actor_cannot_pour_returns_zero():
    assert _res().soft_target("bottle_001", "bottle", "bowl",
                              "canPourInto", "bowl_003") == 0.0


def test_canpierce_is_looked_up_not_hardcoded():
    # canPierce moved from a hardcoded rule to real VLM judgment (72d0ef6):
    # tip sharpness is a visual property, so the score table decides, exactly
    # like canCut. A blunt-tipped knife must be able to score LOW.
    rows = _rows() + [
        {"actor": "knife_001", "recipient": "apple_001", "predicate": "canPierce", "score": 2},
    ]
    r = PairVlmResolver.from_dicts(rows, _resolved(), _curation())
    assert r.soft_target("knife_001", "knife", "apple",
                         "canPierce", "apple_001") == pytest.approx(2 / 9)


def test_canpierce_without_a_row_returns_none_not_one():
    # The old rule returned 1.0 for every non-corn pair, which is what gave
    # canPierce zero negatives corpus-wide and a degenerate F1 of 1.000.
    # Absent a VLM row the answer is "no evidence", not "true".
    assert _res().soft_target("knife_001", "knife", "apple",
                              "canPierce", "apple_001") is None


def test_canpierce_corn_carveout_is_gone():
    # corn is no longer special-cased; if the VLM scored it, that score wins.
    rows = _rows() + [
        {"actor": "knife_001", "recipient": "corn_001", "predicate": "canPierce", "score": 8},
    ]
    curation = dict(_curation())
    curation["corn"] = {"placement": {"whole": ["001"]}}
    r = PairVlmResolver.from_dicts(rows, _resolved(), curation)
    assert r.soft_target("knife_001", "knife", "corn",
                         "canPierce", "corn_001") == pytest.approx(8 / 9)


def test_score_of_ten_is_clamped_to_one():
    # The prompt text advertises a 0-10 range while the scale the resolver
    # divides by is 9, so a compliant model returning 10 would normalise to
    # 1.11 and hand the trainer a target outside [0, 1]. No 10 appears in the
    # labelled corpus, so no published number is affected -- this pins the
    # behaviour so a future prompt or model change cannot reintroduce it.
    rows = _rows() + [
        {"actor": "knife_001", "recipient": "apple_001", "predicate": "canPierce", "score": 10},
    ]
    r = PairVlmResolver.from_dicts(rows, _resolved(), _curation())
    assert r.soft_target("knife_001", "knife", "apple",
                         "canPierce", "apple_001") == pytest.approx(1.0)


def test_negative_score_is_clamped_to_zero():
    rows = _rows() + [
        {"actor": "knife_001", "recipient": "apple_001", "predicate": "canPierce", "score": -3},
    ]
    r = PairVlmResolver.from_dicts(rows, _resolved(), _curation())
    assert r.soft_target("knife_001", "knife", "apple",
                         "canPierce", "apple_001") == pytest.approx(0.0)
