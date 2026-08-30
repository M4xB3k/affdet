"""Tests for the manual pair-labelling tool (affdet/label/manual_tool.py).

The load-bearing test in this file is the blindness denylist: the labeller must
never see the VLM's label or the model's prediction, or the human-vs-teacher
comparison (experiment E3) measures anchoring instead of agreement. That is
enforced structurally -- the tool refuses to load a pair list carrying those
fields -- and `test_blindness_*` below is what keeps it enforced.
"""
from __future__ import annotations

import json

import pytest

from affdet.label.manual_tool import (
    BLIND_FIELD_PATTERNS,
    LABELLED_PREDICATES,
    BlindnessViolation,
    PairListError,
    compute_pair_id,
    definition_for,
    grid_image_name,
    load_pair_list,
    load_session,
    make_record,
    append_record,
    remaining_items,
    _payload,
)


def _pair(**over):
    base = {
        "pair_id": "p0001",
        "actor": {"obj_id": "knife_011", "obj_class": "knife"},
        "recipient": {"obj_id": "apple_003", "obj_class": "apple"},
        "predicate": "canCut",
        "pair_dir": "knife_011__apple_003",
    }
    base.update(over)
    return base


def _write_list(tmp_path, pairs, version="test-v1", **extra):
    doc = {"pair_list_version": version, "pairs": pairs}
    doc.update(extra)
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(doc))
    return p


# --------------------------------------------------------------------------
# Blindness -- the single most important requirement in the brief
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "score", "vlm_score", "vlm_label", "label",
    "prediction", "pred", "pred_score", "model_score", "rationale",
])
def test_blindness_rejects_leaking_field(tmp_path, field):
    """A pair list carrying the teacher's answer must fail loudly at load time,
    not be silently ignored -- a field the tool merely declines to render is
    still one accidental template edit away from being displayed."""
    path = _write_list(tmp_path, [_pair(**{field: 7})])
    with pytest.raises(BlindnessViolation) as exc:
        load_pair_list(path)
    assert field in str(exc.value)


def test_blindness_checks_nested_objects(tmp_path):
    path = _write_list(tmp_path, [
        _pair(actor={"obj_id": "knife_011", "obj_class": "knife", "vlm_score": 9}),
    ])
    with pytest.raises(BlindnessViolation):
        load_pair_list(path)


def test_blindness_is_case_insensitive(tmp_path):
    path = _write_list(tmp_path, [_pair(VLM_Score=3)])
    with pytest.raises(BlindnessViolation):
        load_pair_list(path)


def test_blindness_allows_innocuous_lookalike_fields(tmp_path):
    """`predicate` contains "pred" as a substring; over-eager matching would
    reject every valid pair list ever written."""
    items = load_pair_list(_write_list(tmp_path, [_pair()]))
    assert items[0]["predicate"] == "canCut"


def test_blind_patterns_cover_label_corpus_field_names():
    """labels_omni.jsonl records are {actor, recipient, predicate, score,
    rationale} -- if someone pipes that straight into a pair list, both
    answer-bearing keys must trip the guard."""
    assert any(p in BLIND_FIELD_PATTERNS for p in ("score",))
    assert any(p in BLIND_FIELD_PATTERNS for p in ("rationale",))


# --------------------------------------------------------------------------
# Pair list validation
# --------------------------------------------------------------------------

def test_rejects_rule_based_predicates(tmp_path):
    """canPourInto is rule-based, never VLM-judged, so a human label on it
    cannot measure teacher quality (see the protocol doc §3). canPierce
    moved OFF this list 2026-08-03 -- it's a real VLM judgment now (a
    blunt-tip novelty knife showed the old hardcoded rule was wrong), so a
    human label on it is meaningful again."""
    path = _write_list(tmp_path, [_pair(predicate="canPourInto")])
    with pytest.raises(PairListError) as exc:
        load_pair_list(path)
    assert "canPourInto" in str(exc.value)


def test_labelled_predicates_are_the_five_vlm_judged_ones():
    # 2026-08-03: canPierce joined canCut/canPeel/canContain/canSupport --
    # only canPourInto remains rule-based (see vlm.py's VLM_PREDICATES).
    assert set(LABELLED_PREDICATES) == {
        "canCut", "canPeel", "canContain", "canSupport", "canPierce"}


def test_rejects_unknown_predicate(tmp_path):
    path = _write_list(tmp_path, [_pair(predicate="canTeleport")])
    with pytest.raises(PairListError):
        load_pair_list(path)


def test_rejects_missing_required_field(tmp_path):
    bad = _pair()
    del bad["recipient"]
    with pytest.raises(PairListError):
        load_pair_list(_write_list(tmp_path, [bad]))


def test_rejects_duplicate_pair_ids(tmp_path):
    path = _write_list(tmp_path, [_pair(), _pair()])
    with pytest.raises(PairListError):
        load_pair_list(path)


def test_rejects_empty_pair_list(tmp_path):
    with pytest.raises(PairListError):
        load_pair_list(_write_list(tmp_path, []))


def test_pair_list_version_is_required(tmp_path):
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps({"pairs": [_pair()]}))
    with pytest.raises(PairListError):
        load_pair_list(p)


# --------------------------------------------------------------------------
# pair_id stability -- the join key against VLM labels and model predictions
# --------------------------------------------------------------------------

def test_pair_id_is_stable_across_calls():
    a = compute_pair_id("knife_011", "apple_003", "canCut", "v1")
    b = compute_pair_id("knife_011", "apple_003", "canCut", "v1")
    assert a == b


def test_pair_id_is_order_sensitive():
    """Pairs are ORDERED; swapping actor and recipient is a different question
    and must never collide on the join key."""
    a = compute_pair_id("knife_011", "apple_003", "canCut", "v1")
    b = compute_pair_id("apple_003", "knife_011", "canCut", "v1")
    assert a != b


def test_pair_id_varies_with_predicate_and_version():
    base = compute_pair_id("knife_011", "apple_003", "canCut", "v1")
    assert base != compute_pair_id("knife_011", "apple_003", "canPeel", "v1")
    assert base != compute_pair_id("knife_011", "apple_003", "canCut", "v2")


# --------------------------------------------------------------------------
# Definitions -- imported live from prompts.py so protocol drift is impossible
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pred", ["canCut", "canPeel", "canContain", "canSupport"])
def test_definition_matches_prompts_module_verbatim(pred):
    """The human must be answering the SAME question the VLM was asked. If the
    calibration text is ever copy-pasted instead of imported, this fails."""
    from affdet.label.prompts import _CALIBRATION, AFFORDANCE_QUESTION

    text = definition_for(pred)
    assert _CALIBRATION[pred] in text
    assert AFFORDANCE_QUESTION[pred] in text


def test_definition_rejects_rule_based_predicate():
    with pytest.raises(KeyError):
        definition_for("canPourInto")


# --------------------------------------------------------------------------
# Which image the labeller sees -- must match what the VLM was shown
# --------------------------------------------------------------------------

def test_cancut_uses_the_cancut_composite():
    """canCut was labelled from a bespoke composite (2 paired angles + 2 solo
    close-up tiles) added during the knife-type fix, not the standard grid."""
    assert grid_image_name("canCut") == "cancut_grid.png"


@pytest.mark.parametrize("pred", ["canPeel", "canContain", "canSupport"])
def test_other_predicates_use_the_standard_grid(pred):
    assert grid_image_name(pred) == "grid.png"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

def test_record_carries_full_provenance():
    rec = make_record(_pair(), "yes", confidence=2, confidence_explicit=True,
                      note="", elapsed_ms=4200, pair_list_version="v1",
                      session_id="s1", labeller="max")
    for key in ("pair_id", "pair_list_version", "actor", "recipient",
                "predicate", "verdict", "confidence", "confidence_explicit",
                "timestamp_utc", "elapsed_ms", "session_id", "labeller",
                "tool_version", "schema_version"):
        assert key in rec


def test_record_never_carries_a_blind_field():
    # 2026-08-03: this used to substring-match `"score"` against the whole
    # record, which would now false-positive on the legitimate
    # `"human_score"` key (the labeller's OWN graded judgment, not a leaked
    # VLM/model value -- see test_human_score_field_does_not_trip_the_
    # blindness_denylist). Checks EXACT keys the teacher/model would use
    # instead, so a real leak is still caught precisely.
    rec = make_record(_pair(), "yes", confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max")
    assert "vlm_score" not in rec
    assert "model_score" not in rec
    assert "prediction" not in rec
    assert "rationale" not in rec
    flat = json.dumps(rec).lower()
    assert "vlm" not in flat and "prediction" not in flat


def test_confidence_defaults_to_three_but_is_flagged_inexplicit():
    """A skipped confidence and a deliberate 3 must stay distinguishable, or the
    field silently overstates certainty across the whole session."""
    rec = make_record(_pair(), "yes", confidence=None, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max")
    assert rec["confidence"] == 3
    assert rec["confidence_explicit"] is False


def test_fast_answer_is_flagged_not_blocked():
    rec = make_record(_pair(), "yes", confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=200, pair_list_version="v1",
                      session_id="s1", labeller="max")
    assert "fast_answer" in rec["flags"]


def test_normal_speed_answer_is_not_flagged():
    rec = make_record(_pair(), "yes", confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=5000, pair_list_version="v1",
                      session_id="s1", labeller="max")
    assert "fast_answer" not in rec["flags"]


@pytest.mark.parametrize("verdict", ["yes", "no", "unsure"])
def test_accepts_the_three_verdicts(verdict):
    rec = make_record(_pair(), verdict, confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max")
    assert rec["verdict"] == verdict


def test_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        make_record(_pair(), "maybe", confidence=3, confidence_explicit=False,
                    note="", elapsed_ms=1000, pair_list_version="v1",
                    session_id="s1", labeller="max")


# --------------------------------------------------------------------------
# Graded (0-9) scoring -- 2026-08-03: E3's binary yes/no couldn't distinguish
# "structurally capable, however poorly" (e.g. a box cutter CAN cut a melon)
# from "a good fit for this job" -- the model's own score is graded quality-
# of-fit, matching the VLM's training label, so a binary human verdict was
# comparing two different questions. Graded 0-9 scoring (same scale as the
# VLM prompt) lets a poor-but-real fit land at e.g. 2-3 instead of forcing a
# "yes" a 0.5/5-9 threshold then contradicts.
# --------------------------------------------------------------------------

def test_human_score_derives_yes_verdict_at_or_above_five():
    rec = make_record(_pair(), None, confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max", human_score=5)
    assert rec["human_score"] == 5
    assert rec["verdict"] == "yes"


def test_human_score_derives_no_verdict_below_five():
    rec = make_record(_pair(), None, confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max", human_score=2)
    assert rec["human_score"] == 2
    assert rec["verdict"] == "no"


def test_human_score_zero_is_a_valid_score_not_treated_as_missing():
    # 0 is falsy in Python -- a naive `if human_score:` check would silently
    # treat a genuine 0 (structurally impossible) the same as "not provided".
    rec = make_record(_pair(), None, confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max", human_score=0)
    assert rec["human_score"] == 0
    assert rec["verdict"] == "no"


def test_human_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        make_record(_pair(), None, confidence=3, confidence_explicit=False,
                    note="", elapsed_ms=1000, pair_list_version="v1",
                    session_id="s1", labeller="max", human_score=10)
    with pytest.raises(ValueError):
        make_record(_pair(), None, confidence=3, confidence_explicit=False,
                    note="", elapsed_ms=1000, pair_list_version="v1",
                    session_id="s1", labeller="max", human_score=-1)


def test_human_score_and_verdict_are_mutually_exclusive():
    # Exactly one of (legacy verdict) or (human_score) must drive a record --
    # supplying both is ambiguous about which one is authoritative.
    with pytest.raises(ValueError):
        make_record(_pair(), "yes", confidence=3, confidence_explicit=False,
                    note="", elapsed_ms=1000, pair_list_version="v1",
                    session_id="s1", labeller="max", human_score=7)


def test_legacy_verdict_only_calls_still_work_without_human_score():
    # Backward compatibility: existing yes/no/unsure callers (and any
    # in-progress non-graded session) must be unaffected.
    rec = make_record(_pair(), "yes", confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max")
    assert rec["verdict"] == "yes"
    assert rec["human_score"] is None


def test_payload_defaults_to_non_graded():
    payload = _payload([_pair()], done=0, total=1)
    assert payload["graded"] is False


def test_payload_can_request_graded_mode():
    payload = _payload([_pair()], done=0, total=1, graded=True)
    assert payload["graded"] is True


def test_human_score_field_does_not_trip_the_blindness_denylist():
    # human_score is the LABELLER'S OWN graded judgment, never the VLM's or
    # model's score -- it must coexist with the blind-field guard, not be
    # blocked by it. See test_record_never_carries_a_blind_field below for
    # what's still actually forbidden (vlm_score, prediction, etc).
    rec = make_record(_pair(), None, confidence=3, confidence_explicit=False,
                      note="", elapsed_ms=1000, pair_list_version="v1",
                      session_id="s1", labeller="max", human_score=4)
    assert "human_score" in rec


# --------------------------------------------------------------------------
# Durability and resume -- the user will not redo a session lost to a crash
# --------------------------------------------------------------------------

def _rec(pair_id, verdict="yes", **over):
    r = make_record(_pair(pair_id=pair_id), verdict, confidence=3,
                    confidence_explicit=False, note="", elapsed_ms=1000,
                    pair_list_version="v1", session_id="s1", labeller="max")
    r.update(over)
    return r


def test_append_then_load_roundtrips(tmp_path):
    out = tmp_path / "session.jsonl"
    append_record(out, _rec("p1"))
    append_record(out, _rec("p2", "no"))
    recs = load_session(out)
    assert [r["pair_id"] for r in recs] == ["p1", "p2"]
    assert recs[1]["verdict"] == "no"


def test_append_is_durable_one_line_per_record(tmp_path):
    out = tmp_path / "session.jsonl"
    append_record(out, _rec("p1"))
    append_record(out, _rec("p2"))
    assert len([ln for ln in out.read_text().splitlines() if ln.strip()]) == 2


def test_load_session_of_missing_file_is_empty(tmp_path):
    assert load_session(tmp_path / "nope.jsonl") == []


def test_load_session_tolerates_truncated_final_line(tmp_path):
    """A crash mid-write leaves a partial line; the completed work before it
    must still be recoverable."""
    out = tmp_path / "session.jsonl"
    append_record(out, _rec("p1"))
    with out.open("a") as fh:
        fh.write('{"pair_id": "p2", "verdi')
    assert [r["pair_id"] for r in load_session(out)] == ["p1"]


def test_remaining_skips_completed_items(tmp_path):
    items = [_pair(pair_id="p1"), _pair(pair_id="p2"), _pair(pair_id="p3")]
    out = tmp_path / "session.jsonl"
    append_record(out, _rec("p1"))
    assert [i["pair_id"] for i in remaining_items(items, load_session(out))] == ["p2", "p3"]


def test_remaining_is_everything_when_nothing_done(tmp_path):
    items = [_pair(pair_id="p1"), _pair(pair_id="p2")]
    assert len(remaining_items(items, [])) == 2


def test_revision_supersedes_but_preserves_the_original(tmp_path):
    """Going back with `B` must not rewrite history -- the original answer is
    kept so a change of mind is auditable."""
    out = tmp_path / "session.jsonl"
    append_record(out, _rec("p1", "yes"))
    append_record(out, _rec("p1", "no", revision_of="p1"))
    recs = load_session(out)
    assert len(recs) == 2
    assert [r["verdict"] for r in recs] == ["yes", "no"]


def test_generated_pairs_are_structurally_valid():
    """Regression: the dry-run generator once offered a peeler for canCut using
    hand-rolled class rules. Candidacy must come from the intersection matrix,
    the same authority the renderer and the VLM labeller used."""
    from affdet.intersection_matrix import is_candidate_pair

    assert is_candidate_pair("knife", "apple", "canCut")
    assert not is_candidate_pair("peeler", "orange", "canCut")
    assert is_candidate_pair("peeler", "orange", "canPeel")
    assert not is_candidate_pair("knife", "apple", "canContain")


def test_revised_item_is_not_offered_again(tmp_path):
    items = [_pair(pair_id="p1"), _pair(pair_id="p2")]
    out = tmp_path / "session.jsonl"
    append_record(out, _rec("p1", "yes"))
    append_record(out, _rec("p1", "no", revision_of="p1"))
    assert [i["pair_id"] for i in remaining_items(items, load_session(out))] == ["p2"]
