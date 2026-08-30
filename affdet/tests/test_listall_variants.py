"""Prompt variants for the E8 criticality experiment.

The property every variant must preserve: no candidate list. A variant that
leaked one would silently turn the experiment into arm G+ and the comparison
against the baseline would measure nothing.
"""
import pytest

from affdet.label.listall_variants import (
    VARIANTS, build_variant_system_prompt, build_variant_user_prompt,
)
from affdet.label.listall_prompts import build_listall_system_prompt

PREDS = ["canContain", "canSupport", "canCut", "canPeel", "canPierce"]
VOCAB = ["apple", "bowl", "knife", "fork"]


def test_base_variant_is_byte_identical_to_the_shipped_prompt():
    """Otherwise 'base' is not a control."""
    assert (build_variant_system_prompt(PREDS, "base")
            == build_listall_system_prompt(PREDS))


@pytest.mark.parametrize("v", VARIANTS)
def test_no_variant_leaks_a_candidate_list(v):
    p = build_variant_user_prompt(VOCAB, PREDS, 4, v)
    assert "->" not in p
    assert "Score exactly the following" not in p
    assert "structurally possible" not in p


@pytest.mark.parametrize("v", VARIANTS)
def test_every_variant_keeps_the_calibration_blocks(v):
    s = build_variant_system_prompt(PREDS, v)
    assert "canCut calibration" in s
    assert "canContain definition" in s


@pytest.mark.parametrize("v", VARIANTS)
def test_no_variant_ships_the_novelty_placeholder(v):
    assert "{novelty_block}" not in build_variant_system_prompt(PREDS, v)


def test_strict_names_concrete_caps_not_vague_exhortation():
    s = build_variant_system_prompt(PREDS, "strict")
    for term in ("box cutter", "folding", "novelty", "pointed"):
        assert term in s.lower()


def test_typed_requires_the_classification_fields():
    s = build_variant_system_prompt(PREDS, "typed")
    u = build_variant_user_prompt(VOCAB, PREDS, 4, "typed")
    assert "knife_type" in s and "knife_type" in u
    assert "tip_shape" in s and "tip_shape" in u


def test_reasoned_requires_a_why_field():
    u = build_variant_user_prompt(VOCAB, PREDS, 4, "reasoned")
    assert '"why"' in u


def test_base_does_not_request_extra_fields():
    u = build_variant_user_prompt(VOCAB, PREDS, 4, "base")
    assert "knife_type" not in u and '"why"' not in u


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_variant_system_prompt(PREDS, "nope")
    with pytest.raises(ValueError):
        build_variant_user_prompt(VOCAB, PREDS, 4, "nope")


@pytest.mark.parametrize("v", VARIANTS)
def test_user_prompt_still_states_object_count_and_vocabulary(v):
    u = build_variant_user_prompt(VOCAB, PREDS, 4, v)
    assert "4" in u
    for t in VOCAB:
        assert t in u
