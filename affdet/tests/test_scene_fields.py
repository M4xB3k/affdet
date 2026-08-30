"""Restoring the committed-judgement fields to the enumerated scene arms.

`prompts.py` requires the pairwise labeller to emit an extra field for
canCut ("type"), canContain ("enclosure") and canSupport ("size_note") before
the score, on the recorded grounds that articulating the judgement measurably
stabilises it. The scene adaptation in `scene_prompts.py` dropped all three --
an omission, not a decision.

E8's `typed` prompt variant re-added the same mechanism to the list-all arm and
was the best-performing variant tested (agreement 0.796 -> 0.833, MAE 2.102 ->
1.802, zero cap violations), so the loss is worth measuring rather than
assuming benign.

The flag is opt-in: with fields=False the prompt must be byte-identical to the
one that produced every committed E7 number.
"""
import pytest

from affdet.label.scene_prompts import (
    build_scene_system_prompt, build_scene_user_prompt,
)


def _q(n=1):
    return [{"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut",
             "actor_box": 1, "recipient_box": 2,
             "actor_label": "knife", "recipient_label": "apple"}
            for _ in range(n)]


VOCAB = ["knife", "apple", "bowl"]


@pytest.mark.parametrize("arm", ["gplus", "g", "e"])
def test_default_is_byte_identical_to_the_shipped_prompt(arm):
    """Guards every committed E7 number against silent drift."""
    assert (build_scene_user_prompt(_q(), VOCAB, arm)
            == build_scene_user_prompt(_q(), VOCAB, arm, fields=False))


def test_fields_flag_changes_the_output_spec():
    off = build_scene_user_prompt(_q(), VOCAB, "gplus", fields=False)
    on = build_scene_user_prompt(_q(), VOCAB, "gplus", fields=True)
    assert on != off
    assert '"type"' in on and '"type"' not in off


def test_fields_requests_type_for_cancut():
    p = build_scene_user_prompt(_q(), VOCAB, "gplus", fields=True)
    assert '"type"' in p
    for t in ("kitchen knife", "box cutter"):
        assert t in p


def test_fields_requests_enclosure_and_size_note():
    qs = [{"actor": "bowl_1", "recipient": "apple_1",
           "predicate": p, "actor_box": 1, "recipient_box": 2,
           "actor_label": "bowl", "recipient_label": "apple"}
          for p in ("canContain", "canSupport")]
    p = build_scene_user_prompt(qs, VOCAB, "gplus", fields=True)
    assert "enclosure" in p and "size_note" in p


def test_fields_only_mentions_predicates_actually_asked_about():
    """A canPeel-only call must not carry canCut's knife-type language."""
    qs = [{"actor": "peeler_1", "recipient": "apple_1", "predicate": "canPeel",
           "actor_box": 1, "recipient_box": 2,
           "actor_label": "peeler", "recipient_label": "apple"}]
    p = build_scene_user_prompt(qs, VOCAB, "gplus", fields=True)
    assert "box cutter" not in p
    assert "enclosure" not in p


def test_fields_preserves_the_enumerated_candidate_list():
    """The whole point of the enumerated arms -- must survive the flag."""
    p = build_scene_user_prompt(_q(2), VOCAB, "gplus", fields=True)
    assert "Score exactly the following" in p
    assert "->" in p


def test_fields_works_for_arm_e_naming_style():
    p = build_scene_user_prompt(_q(), VOCAB, "e", fields=True)
    assert '"actor"' in p and "actor_box" not in p


def test_system_prompt_is_unaffected_by_the_flag():
    """Only the output spec changes; calibration text is untouched."""
    assert "canCut calibration" in build_scene_system_prompt(["canCut"], "gplus")
