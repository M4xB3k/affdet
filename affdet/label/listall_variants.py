"""Prompt variants for the E8 list-all criticality experiment.

The baseline list-all prompt labels canCut 94.6 % positive and canPierce 97.1 %
positive on the training corpus. Human judgement on comparable objects is
**54.9 %** and **79.6 %**. The VLM is not merely stricter or looser than the
human -- it is wrong in a consistent direction, and the missing negatives are
what starve the training corpus and drive threshold calibration to 0.05.

Four variants, each isolating one hypothesis about WHY:

  base      the shipped list-all prompt. Control.

  strict    stronger, more explicit caps in the calibration text. Tests
            "the model did not weight the existing rules hard enough".
            Note the baseline text ALREADY caps box cutters at 2-3 and toy
            knives at 0-2, so if wording alone were the fix this should have
            worked already -- which is why the next two exist.

  typed     forces an explicit `knife_type` / `tip_shape` field in the output
            BEFORE the score. `prompts.py` already does this for the paired
            labeller (the `"type"` field on canCut) on the grounds that
            articulating the judgement measurably stabilises it. The list-all
            prompt dropped it. Tests "the model needs to commit to a
            classification before scoring".

  reasoned  asks for a one-clause rationale per judgement. Distinct from the
            `reasoning_effort` API knob, which buys hidden deliberation the
            model never has to make explicit; this forces the justification
            into the output where it constrains the score. Tests "making the
            model say why changes what it says".

Every variant keeps the calibration blocks and the no-candidate-list property
of the baseline, so a difference is attributable to the manipulation.
"""
from __future__ import annotations

from affdet.label.listall_prompts import (
    _LISTALL_INTRO, _SCORING, build_listall_system_prompt,
)
from affdet.label.prompts import AFFORDANCE_QUESTION

VARIANTS = ("base", "strict", "typed", "reasoned")

#: Appended to the system prompt. Deliberately concrete: the baseline text is
#: already emphatic, so vaguer exhortation ("be critical") would test nothing.
_STRICT_BLOCK = """\
CRITICAL SCORING DISCIPLINE — read before scoring canCut or canPierce.

You are systematically TOO GENEROUS on these two predicates. Most tools in
these scenes are NOT well suited to the food they sit beside. Before giving
any score of 6 or above, you must be able to name the specific feature that
earns it.

canCut — the DEFAULT is a low score. Award 8-9 ONLY to an unmistakable real
kitchen knife (chef's, paring, santoku, bread, cleaver) with a clear exposed
blade and a proper handle, against a food it can actually slice.
  - box cutter / utility / retractable blade  -> 2-3, never higher
  - folding or pocket knife                   -> 5-6, never higher
  - novelty, toy, decorative, plastic         -> 0-2, never higher
  - butter knife, spreader, blunt table knife -> 1-3, never higher
  - a real kitchen knife far too small for the food -> 4-6
If you cannot positively identify the object as a real kitchen knife, you must
not score above 5.

canPierce — the DEFAULT is a low score. Piercing requires a SHARP POINTED TIP
that concentrates force. Judge the tip, not the blade.
  - clearly pointed tip (chef's knife, paring knife, fork tines) -> 7-9
  - rounded, squared, blunt or snapped-off tip                   -> 0-3
  - bread knife, cleaver, santoku (blunt or squared tips)        -> 0-3
  - butter knife, spreader                                       -> 0-1
  - food far too hard or too large for the tip to enter          -> cap at 4
A knife that cuts well does NOT automatically pierce well. These are
independent judgements and you must score them independently."""

_TYPED_BLOCK = """\
MANDATORY CLASSIFICATION BEFORE SCORING.

For every canCut judgement you report, you must first state the actor's type
in a "knife_type" field, chosen from exactly:
  "kitchen knife" | "folding knife" | "box cutter" | "novelty/toy" |
  "butter knife" | "other"
The type constrains the score and you may not exceed its cap:
  kitchen knife -> up to 9   folding knife -> up to 6   box cutter -> up to 3
  novelty/toy   -> up to 2   butter knife  -> up to 3   other -> up to 4

For every canPierce judgement you must first state the actor's tip in a
"tip_shape" field, chosen from exactly:
  "pointed" | "rounded" | "squared" | "blunt" | "tines"
The tip constrains the score:
  pointed -> up to 9   tines -> up to 9   rounded -> up to 3
  squared -> up to 3   blunt -> up to 1

State the field FIRST, then the score consistent with it. Do not state a type
and then contradict it with the score."""

_REASONED_BLOCK = """\
JUSTIFY EVERY JUDGEMENT.

For each affordance you report, include a "why" field: one short clause naming
the specific visible feature that determines the score — the blade type, the
tip shape, the size relationship, the rim. Write the justification FIRST and
let it determine the number.

If you cannot name a concrete visible feature that supports a high score, the
score is not high. "It is a knife and that is food" is not a justification."""


def build_variant_system_prompt(predicates: list[str], variant: str) -> str:
    """System prompt for one variant. `base` is the unmodified list-all prompt."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected {VARIANTS}")
    base = build_listall_system_prompt(predicates)
    extra = {"base": "", "strict": _STRICT_BLOCK,
             "typed": _TYPED_BLOCK, "reasoned": _REASONED_BLOCK}[variant]
    return base if not extra else base + "\n\n" + extra


def build_variant_user_prompt(vocabulary: list[str], predicates: list[str],
                              n_objects: int, variant: str) -> str:
    """User turn for one variant.

    Carries no candidate list, exactly as the baseline: the variants change how
    the model is asked to JUDGE, never what it is asked to judge, so the
    volunteered-candidate property under test is preserved.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected {VARIANTS}")

    lines = [
        "Object categories that may appear in this scene (the complete "
        "vocabulary):",
        "  " + ", ".join(sorted(vocabulary)),
        "",
        f"This scene contains {n_objects} boxed objects, numbered [1] to "
        f"[{n_objects}].",
        "",
        "The affordances to consider:",
    ]
    lines += [f"  {p}: {AFFORDANCE_QUESTION[p]}"
              for p in sorted(predicates) if p in AFFORDANCE_QUESTION]
    lines += [
        "",
        "Examine the scene and report EVERY affordance relationship you judge "
        "to hold between the objects in it. Consider each object as a "
        "possible actor acting upon each other object as recipient; "
        "affordances are directional, so [1] acting on [2] is a different "
        "claim from [2] acting on [1]. Report each relationship you find, "
        "with a score.",
        "",
        "Report only the relationships you actually judge to hold. Do not pad "
        "the list, and do not omit a relationship that holds merely because it "
        "seems obvious.",
        "",
        "Respond with a single JSON object with one key, \"affordances\", "
        "whose value is a list. Each entry is:",
    ]
    if variant == "typed":
        lines += [
            '  {"actor_box": <int>, "recipient_box": <int>, '
            '"predicate": "<name>", "knife_type": "<type, canCut only>", '
            '"tip_shape": "<shape, canPierce only>", "score": <0-10>}',
            "Omit knife_type/tip_shape for predicates they do not apply to.",
        ]
    elif variant == "reasoned":
        lines += [
            '  {"actor_box": <int>, "recipient_box": <int>, '
            '"predicate": "<name>", "why": "<one short clause>", '
            '"score": <0-10>}',
        ]
    else:
        lines += [
            '  {"actor_box": <int>, "recipient_box": <int>, '
            '"predicate": "<name>", "score": <0-10>}',
        ]
    lines += [
        "",
        "If no affordance holds anywhere in this scene, return an empty list: "
        '{"affordances": []}.',
        "",
        "Return the JSON only — no commentary outside it.",
    ]
    return "\n".join(lines)
