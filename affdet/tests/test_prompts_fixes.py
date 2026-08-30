"""Regression tests for the canCut/canSupport prompt fixes (2026-07-14).

canCut: validated against knife_029 (box-cutter-shaped mesh, mislabeled a
real kitchen knife under the old prompt, scored 8-9 consistently) and
knife_001 (genuine chef's knife, correctly 9 under both prompts). The old
prompt had no notion of "knife type" at all, so any sharp-looking blade
scored high regardless of whether it was a real kitchen knife, a folding
knife, a box cutter, or a decorative prop. Fixed prompt requires classifying
type FIRST and capping the score by type; also fixed a v1 bug where a
folded/closed blade was scored 0 outright (judging pose, not capability) --
v2 judges the object as if opened.

canSupport: validated against dish_012 x watermelon_001 (a small saucer next
to a whole watermelon scored 9/9 -- the VLM's own calibration example says
"food clearly larger than the dish -> 3" but wasn't reliably applying it).
Fixed prompt makes the size-comparison check mandatory and first, before any
category-based reasoning, and adds explicit pose-invariance for tipped/
upside-down containers (a render artifact, not the object's real capability).

See also: /scratch/maksym.bek/affdet/{typed_prompt_v2_levels,cansupport_fixed}.log
on the cluster for the full validation sweep data (10 runs x 4 reasoning
levels per fix).
"""
from affdet.label.prompts import build_system_prompt


def test_cancut_prompt_requires_type_classification():
    prompt = build_system_prompt(["canCut"])
    assert "kitchen knife" in prompt
    assert "box cutter" in prompt
    assert "folding" in prompt.lower() or "pocket" in prompt.lower()


def test_cancut_prompt_judges_capability_not_pose():
    prompt = build_system_prompt(["canCut"])
    # must NOT contain the v1 bug's blanket "closed -> 0" rule
    assert "as if opened" in prompt.lower() or "as if it were opened" in prompt.lower()


def test_cansupport_prompt_requires_size_check_first():
    prompt = build_system_prompt(["canSupport"])
    assert "size" in prompt.lower()
    # the dish/watermelon failure mode: must explicitly call out size
    # mismatch overriding category-based "purpose-built" reasoning
    assert "overhang" in prompt.lower()


def test_cansupport_prompt_judges_upright_not_tipped_pose():
    prompt = build_system_prompt(["canSupport"])
    assert "upright" in prompt.lower()
    assert "tipped" in prompt.lower() or "upside" in prompt.lower() or "on its side" in prompt.lower()


def test_cancut_prompt_requests_type_field_in_output():
    # 2026-07-14 second pass: forcing the model to output its own type
    # classification (not just reason about it internally) is required --
    # validated live that knife_006/008/022/029 (all box cutters) score
    # correctly (2-3) ONLY when the type field is requested; without it they
    # default to a lazy "5" regardless of true type. Same pattern as
    # canSupport's size_note.
    prompt = build_system_prompt(["canCut"])
    assert '"type"' in prompt


def test_unrelated_predicate_prompt_unaffected():
    # canPeel calibration text should still be present and untouched when
    # only canPeel is requested -- these fixes must not leak into or break
    # other predicates' calibration blocks.
    prompt = build_system_prompt(["canPeel"])
    assert "vegetable peeler" in prompt


def test_cancontain_prompt_requests_enclosure_field_in_output():
    # 2026-08-01: E3 (36-pair human gold set) found bowl_011 x broccolini_001
    # scored 0.89 ("yes") for a long stalk sticking out of a shallow bowl --
    # the model has walls/depth calibration text but nothing forces it to
    # articulate the enclosure judgment before scoring, unlike canSupport's
    # size_note and canCut's type field (both of which measurably improved
    # reliability the same way). Same fix pattern, applied to the one
    # predicate that never got it.
    prompt = build_system_prompt(["canContain"])
    assert '"enclosure"' in prompt or '"walls_note"' in prompt


def test_cancontain_prompt_output_field_unaffected_by_other_predicates():
    prompt = build_system_prompt(["canPeel"])
    assert '"enclosure"' not in prompt and '"walls_note"' not in prompt


def test_cancontain_prompt_treats_shallow_dish_as_real_containment():
    # 2026-08-01, second pass: the user's human gold labels judge a shallow
    # dish WITH a rim as genuine containment (loosened on purpose, see E3
    # canContain relabel) -- the original prompt's "shallow dish = very low
    # containment regardless" rule conflicted with that judgment on 4/6
    # remaining E3 misses (dish_005/010/017/018), all cases where the human
    # said "yes, contains" and the VLM (correctly per the OLD prompt) said
    # low/no. Loosened so only a truly FLAT plate (no rim) scores low.
    prompt = build_system_prompt(["canContain"])
    assert "flat plate" in prompt.lower() or "flat plate" in prompt
    assert "shallow dish" in prompt.lower()
    # the old blanket rule text must be gone, not just supplemented
    assert "very low walls even if food fits" not in prompt
    assert "kitchen knife" not in prompt  # canCut block must not leak in
