import pytest
from affdet.label.vlm_resolver import VlmScoreResolver


def _pf(cls, dropped=False, pf=None, pi=None):
    return {"effective_class": cls, "dropped": dropped,
            "pourable_from": pf, "pourable_into": pi}


def _resolved():
    return {
        "knife_001": _pf("knife"),
        "knife_011": _pf("peeler"),
        "pan_001":   _pf("pan", dropped=True),
        # pour vessels with the new two-directional flags:
        "teapot_014": _pf("teapot", pf=True,  pi=True),   # spout + open top
        "teapot_023": _pf("teapot", pf=True,  pi=False),  # spout, lidded
        "kettle_001": _pf("kettle", pf=True,  pi=False),
        "bottle_001": _pf("bottle", pf=False, pi=False),
        "bowl_003":   _pf("bowl",   pf=False, pi=True),
    }


def _rows():
    return [
        {"actor": "knife_001", "food": "apple", "affordance": "canCut", "score": 9},
        {"actor": "knife_001", "food": "apple", "affordance": "canCut", "score": 9},
        {"actor": "peeler_011", "food": "apple", "affordance": "canPeel", "score": 7},
    ]


def _res():
    return VlmScoreResolver.from_dicts(_rows(), _resolved())


def test_hit_returns_score_over_nine():
    assert _res().soft_target("knife_001", "knife", "apple", "canCut") == pytest.approx(1.0)


def test_missing_actor_returns_none():
    assert _res().soft_target("knife_099", "knife", "apple", "canCut") is None


def test_dropped_instance_returns_none():
    assert _res().soft_target("pan_001", "pan", "apple", "canSupport") is None


def test_effective_class_remaps_peeler():
    assert _res().effective_class("knife_011", "knife") == "peeler"


def test_peeler_scores_looked_up_under_peeler_key():
    # scene id is knife_011, but the score row is keyed peeler_011
    assert _res().soft_target("knife_011", "knife", "apple", "canPeel") == pytest.approx(7/9)


def test_duplicate_rows_averaged():
    # both knife_001/apple/canCut rows are 9 -> mean 9 -> 1.0
    assert _res().soft_target("knife_001", "knife", "apple", "canCut") == pytest.approx(1.0)


# --- relational canPourInto: actor.pourable_from AND recipient.pourable_into ---

def test_pour_from_and_into_returns_one():
    # teapot_014 pours (from), bowl_003 receives (into) -> 1.0
    assert _res().soft_target("teapot_014", "teapot", "bowl",
                              "canPourInto", "bowl_003") == 1.0


def test_pour_actor_cannot_pour_from_returns_zero():
    # bottle can't pour out, even into a valid receiver
    assert _res().soft_target("bottle_001", "bottle", "bowl",
                              "canPourInto", "bowl_003") == 0.0


def test_pour_recipient_cannot_receive_returns_zero():
    # kettle pours (from) but a lidded teapot_023 can't be poured INTO
    assert _res().soft_target("kettle_001", "kettle", "teapot",
                              "canPourInto", "teapot_023") == 0.0


def test_pour_teapot_is_valid_recipient():
    # teapot as RECIPIENT: kettle -> teapot_014 (open top) -> 1.0
    assert _res().soft_target("kettle_001", "kettle", "teapot",
                              "canPourInto", "teapot_014") == 1.0


def test_pour_unknown_recipient_returns_none():
    # recipient not in resolved table -> no teacher signal
    assert _res().soft_target("teapot_014", "teapot", "bowl",
                              "canPourInto", "bowl_099") is None
