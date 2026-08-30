"""E7 list-all arm — prompt construction and response parsing.

The list-all arm exists to answer one question: when the VLM is NOT handed the
candidate list, does it volunteer the pairs a human labelled positive, or does
it quietly omit them? Everything these tests defend follows from that.

The two failure modes worth writing tests against:

1. **The prompt leaks the candidate list.** The arm would then measure nothing
   at all -- it would be arm G+ with extra steps. Several tests below assert
   that no pair, count, or per-pair enumeration reaches the prompt.
2. **The calibration text drifts from the pair labeller's.** The comparison
   against G+ is only meaningful if the physics being judged is word-for-word
   the same; a paraphrase would make the arms differ in two ways at once.
"""
import pytest

from affdet.label.listall_prompts import (
    build_listall_system_prompt,
    build_listall_user_prompt,
    parse_listall_response,
)
from affdet.label.prompts import _CALIBRATION, _novelty_block

PREDICATES = ["canContain", "canSupport", "canCut", "canPeel", "canPierce"]
VOCAB = ["apple", "bowl", "dish", "knife", "lemon", "peeler"]


# --------------------------------------------------------------------------
# System prompt: calibration reuse
# --------------------------------------------------------------------------

def test_system_prompt_reuses_calibration_verbatim():
    """Not paraphrased. If this drifts, a list-all vs G+ gap stops being
    attributable to the missing candidate list."""
    prompt = build_listall_system_prompt(PREDICATES)
    for p in PREDICATES:
        block = _CALIBRATION[p].replace("{novelty_block}", _novelty_block())
        assert block in prompt, f"{p} calibration is not present verbatim"


def test_system_prompt_substitutes_novelty_block():
    """The placeholder must be filled, not shipped literally.

    scene_prompts.py omitted this .replace() and sent the token
    "{novelty_block}" to the VLM on every G+/G/E call. See
    data/eval/E7_LISTALL_RESULTS.md.
    """
    prompt = build_listall_system_prompt(["canCut"])
    assert "{novelty_block}" not in prompt
    assert "WATCH FOR" in prompt


def test_system_prompt_only_includes_requested_predicates():
    prompt = build_listall_system_prompt(["canPeel"])
    assert "canPeel calibration" in prompt
    assert "canCut calibration" not in prompt


def test_system_prompt_carries_the_scoring_scale():
    prompt = build_listall_system_prompt(PREDICATES)
    assert "0-10" in prompt or "0–10" in prompt


def test_system_prompt_rejects_unknown_predicate():
    with pytest.raises(ValueError):
        build_listall_system_prompt(["canTeleport"])


# --------------------------------------------------------------------------
# The defining property: no candidate list anywhere
# --------------------------------------------------------------------------

def test_user_prompt_does_not_enumerate_candidate_pairs():
    """The whole experiment. The VLM must choose what to report."""
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=4)
    assert "->" not in prompt
    assert "Score exactly the following" not in prompt
    assert "structurally possible" not in prompt


def test_user_prompt_does_not_reveal_how_many_judgements_are_expected():
    """Even a bare count is a hint: told 'there are 7', the VLM pads to 7."""
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=4)
    assert "judgement(s)" not in prompt
    for n in range(1, 40):
        assert f"exactly {n}" not in prompt


def test_user_prompt_asks_for_every_affordance_that_holds():
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=4).lower()
    assert "every" in prompt or "all" in prompt


def test_user_prompt_includes_vocabulary_and_predicate_names():
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=4)
    for term in VOCAB:
        assert term in prompt
    for p in PREDICATES:
        assert p in prompt


def test_user_prompt_states_the_number_of_boxes_not_the_pairs():
    """Box ids are how identity travels; that is scaffolding, not the answer.
    Saying 'boxes [1]..[4] exist' does not say which pairs afford anything."""
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=4)
    assert "4" in prompt


def test_user_prompt_requests_score_per_reported_affordance():
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=4)
    assert "score" in prompt.lower()
    assert "actor_box" in prompt and "recipient_box" in prompt


def test_user_prompt_permits_reporting_nothing():
    """A scene with no affordance must be expressible, or the VLM invents one
    to satisfy the format -- manufacturing a hallucination we would then
    charge it for."""
    prompt = build_listall_user_prompt(VOCAB, PREDICATES, n_objects=2).lower()
    assert "empty" in prompt or "no affordance" in prompt


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _box_map():
    return {1: "knife_031", 2: "lemon_031", 3: "bowl_017"}


def test_parse_maps_box_numbers_back_to_obj_ids():
    raw = """{"affordances": [
        {"actor_box": 1, "recipient_box": 2, "predicate": "canCut", "score": 8},
        {"actor_box": 3, "recipient_box": 2, "predicate": "canContain", "score": 9}
    ]}"""
    got = parse_listall_response(raw, _box_map())
    assert got == {
        ("knife_031", "lemon_031", "canCut"): 8.0,
        ("bowl_017", "lemon_031", "canContain"): 9.0,
    }


def test_parse_strips_markdown_fence():
    raw = '```json\n{"affordances": []}\n```'
    assert parse_listall_response(raw, _box_map()) == {}


def test_parse_keeps_rows_referencing_unknown_boxes_out():
    """A box number that was never drawn cannot be joined to an object. It is
    dropped here and counted as unresolvable by the caller -- never guessed."""
    raw = """{"affordances": [
        {"actor_box": 9, "recipient_box": 2, "predicate": "canCut", "score": 8}
    ]}"""
    assert parse_listall_response(raw, _box_map()) == {}


def test_parse_drops_unknown_predicates():
    raw = """{"affordances": [
        {"actor_box": 1, "recipient_box": 2, "predicate": "canFly", "score": 8}
    ]}"""
    assert parse_listall_response(raw, _box_map()) == {}


def test_parse_drops_out_of_range_and_nonnumeric_scores():
    raw = """{"affordances": [
        {"actor_box": 1, "recipient_box": 2, "predicate": "canCut", "score": 44},
        {"actor_box": 1, "recipient_box": 2, "predicate": "canPierce", "score": "high"}
    ]}"""
    assert parse_listall_response(raw, _box_map()) == {}


def test_parse_drops_self_pairs():
    raw = """{"affordances": [
        {"actor_box": 1, "recipient_box": 1, "predicate": "canCut", "score": 8}
    ]}"""
    assert parse_listall_response(raw, _box_map()) == {}


def test_parse_accepts_bare_list():
    raw = """[{"actor_box": 1, "recipient_box": 2, "predicate": "canCut", "score": 7}]"""
    assert parse_listall_response(raw, _box_map()) == {
        ("knife_031", "lemon_031", "canCut"): 7.0}


def test_parse_last_write_wins_on_duplicate_triples():
    raw = """{"affordances": [
        {"actor_box": 1, "recipient_box": 2, "predicate": "canCut", "score": 3},
        {"actor_box": 1, "recipient_box": 2, "predicate": "canCut", "score": 8}
    ]}"""
    got = parse_listall_response(raw, _box_map())
    assert got[("knife_031", "lemon_031", "canCut")] == 8.0


def test_parse_raises_on_unparseable_text():
    with pytest.raises(ValueError):
        parse_listall_response("I could not find any affordances.", _box_map())


def test_parse_empty_affordances_is_not_an_error():
    """Distinct from a parse failure: 'nothing holds here' is a real answer
    and must not be retried as if the call had failed."""
    assert parse_listall_response('{"affordances": []}', _box_map()) == {}


# --------------------------------------------------------------------------
# Regression: the placeholder defect the existing arms shipped with
# --------------------------------------------------------------------------

def test_scene_arms_also_substitute_the_novelty_block():
    """Guards the 2026-08-18 fix to scene_prompts.py.

    All three committed E7 scene arms sent the VLM the literal string
    "{novelty_block}" in the middle of the canCut calibration, because
    build_scene_system_prompt inserted _CALIBRATION[p] without the
    substitution prompts.py performs. The warning about novelty/toy knives was
    therefore never delivered on any E7 call -- plausibly a contributor to the
    canCut/canPierce error documented in data/eval/E7_LABEL_QUALITY.md.
    """
    from affdet.label.scene_prompts import build_scene_system_prompt
    for arm in ("gplus", "g", "e"):
        prompt = build_scene_system_prompt(["canCut"], arm)
        assert "{novelty_block}" not in prompt
        assert "WATCH FOR" in prompt


def test_listall_and_gplus_share_identical_calibration_text():
    """The two prompts must differ ONLY in the candidate list. If the
    calibration diverges, a list-all-vs-G+ gap is no longer attributable to
    the missing enumeration."""
    from affdet.label.scene_prompts import build_scene_system_prompt
    gplus = build_scene_system_prompt(PREDICATES, "gplus")
    listall = build_listall_system_prompt(PREDICATES)
    for p in PREDICATES:
        block = _CALIBRATION[p].replace("{novelty_block}", _novelty_block())
        assert block in gplus and block in listall
