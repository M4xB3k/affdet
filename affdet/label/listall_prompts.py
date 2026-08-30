"""Prompt builders for the E7 **list-all** arm.

The thesis exposé promised three VLM querying strategies as baselines,
following the object-affordance-mapping section of AutoGPT+P (Birr et al.,
RSS 2024): list-all, yes-no, and hybrid. This module implements the first in
its true form.

What makes it "true" list-all
-----------------------------
`scene_prompts.py` also makes one call per scene, but it *enumerates every
question*: "score exactly the following N judgements", each "already known to
be structurally possible". The candidate set is handed over, so the VLM only
has to grade a list somebody else built.

Here it is not. The VLM sees the image, the class vocabulary and the six
predicate definitions, and must decide **which pairs to report at all**. That
single difference is the experiment: when the VLM volunteers its own
candidates, does it omit pairs a human labelled positive?

The question matters because omissions make recall unmeasurable — a pair the
VLM never mentions cannot be distinguished from one it judged absent. That is
the known weakness of list-all in AutoGPT+P, and it is the mirror of a result
already established on the student side: `data/eval/E7_STUDENT_RESULTS.md` §2
shows that without the structural candidate gate the student's balanced
accuracy collapses to 0.501, i.e. chance. The hypothesis under test is that
enumeration is not a convenience for either judge but the thing that makes the
task measurable.

Prompt parity with arm G+
-------------------------
To keep a list-all-vs-G+ gap attributable to the missing candidate list, every
other element is held fixed: the same numbered-and-named boxed image, the same
vocabulary, and the calibration blocks reused **verbatim** from `prompts.py`.

One deliberate difference. `scene_prompts.build_scene_system_prompt` inserts
`_CALIBRATION[p]` without substituting the `{novelty_block}` placeholder, so
arms G+/G/E sent the VLM the literal token ``{novelty_block}`` where the
novelty/toy-knife warning belongs — the warning was never delivered on any E7
call, and the model saw template debris mid-prompt. This module substitutes it,
as `prompts.build_system_prompt` does. The fix is applied to the existing arms
too and they are re-run, so all four arms again share one prompt; see
`data/eval/E7_LISTALL_RESULTS.md` for what that supersedes.
"""
from __future__ import annotations

import json
import re

from affdet.label.prompts import _CALIBRATION, AFFORDANCE_QUESTION, _novelty_block

#: The predicates this arm can be asked for. canPourInto is absent for the
#: same reason it is absent from the E7 eval set: the human session never
#: judged it on these scenes.
LISTALL_PREDICATES = ("canContain", "canSupport", "canCut", "canPeel",
                      "canPierce")

_LISTALL_INTRO = """\
You are an expert at judging physical affordances between kitchen objects.

You will be shown ONE photograph of a real scene: several objects placed
together on a table. This is a single real arrangement, not a grid of
variations — every object you see is present at its true relative size.

Each object is outlined with a coloured box and labelled with a number and its
category, e.g. "[3] knife". Refer to objects ONLY by their number.

Judge each affordance using what is ACTUALLY VISIBLE — the real relative size,
shape and proportions of these specific object instances — not a generic
assumption about what a "typical" object of that category looks like. If this
particular container is unusually small, or this particular food unusually
large, judge from that, not from a category stereotype.

YOU decide which affordances are worth reporting. You are not given a list of
pairs to grade: survey the objects in the scene yourself and report every
affordance relationship you judge to hold between them."""

_SCORING = """\
Score 0-10 for each affordance you report:
  0    = physically impossible for these specific instances
  1-3  = extremely poor — technically possible for the class pairing, but this
         specific combination is so badly sized or shaped it barely works
  4-5  = poor — works but with real difficulty
  6-7  = acceptable — functional but not purpose-built for this exact pairing
  8-9  = good to excellent — the natural, well-fitted use of this actor on
         this recipient, given their actual visible sizes
  10   = perfect

IMPORTANT: a well-fitted, correctly-sized pairing doing its natural job scores
8-9. Reserve 5-6 for "technically works, but a poor match". Do NOT anchor 5 as
"average" — anchor 8-9 as "this specific pairing works well"."""


def build_listall_system_prompt(predicates: list[str]) -> str:
    """System prompt for one list-all call.

    Carries the same per-predicate calibration blocks as the pair labeller and
    as arm G+, so the physics being judged is identical and only the absence of
    the candidate list differs.
    """
    unknown = [p for p in predicates if p not in _CALIBRATION]
    if unknown:
        raise ValueError(f"unknown predicate(s): {unknown}; "
                         f"expected from {sorted(_CALIBRATION)}")
    novelty = _novelty_block()
    blocks = [_CALIBRATION[p].replace("{novelty_block}", novelty)
              for p in sorted(predicates)]
    return "\n\n".join([_LISTALL_INTRO, _SCORING, *blocks])


def build_listall_user_prompt(vocabulary: list[str], predicates: list[str],
                              n_objects: int) -> str:
    """User turn for one list-all call.

    Deliberately carries NO pair list and NO expected count. `n_objects` is the
    number of boxes drawn — identity scaffolding, since the VLM must know which
    ids are addressable — but it says nothing about which pairs afford
    anything, nor how many rows to return. Telling the model "there are 7
    judgements" would invite it to pad to 7 and would destroy the very quantity
    this arm measures.
    """
    unknown = [p for p in predicates if p not in AFFORDANCE_QUESTION]
    if unknown:
        raise ValueError(f"unknown predicate(s): {unknown}")

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
        '  {"actor_box": <int>, "recipient_box": <int>, '
        '"predicate": "<name>", "score": <0-10>}',
        "",
        "If no affordance holds anywhere in this scene, return an empty list: "
        '{"affordances": []}.',
        "",
        "Return the scores only — no rationale, no extra keys, no commentary "
        "outside the JSON.",
    ]
    return "\n".join(lines)


def parse_listall_response(content: str, box_of: dict[int, str],
                          predicates: tuple[str, ...] = LISTALL_PREDICATES
                          ) -> dict[tuple[str, str, str], float]:
    """Map a volunteered table back onto object ids.

    Returns {(actor_obj_id, recipient_obj_id, predicate): score}. `box_of` maps
    the drawn box number to the obj_id it outlines.

    Rows that cannot be resolved — a box number never drawn, an invented
    predicate, an out-of-range or non-numeric score, or an object acting on
    itself — are DROPPED rather than repaired. A row we cannot join is not
    evidence, and coercing it would put a fabricated score into the results.
    The caller counts what was dropped (see `listall_score.hallucination_report`);
    dropping is not the same as forgiving.

    An empty list is a valid answer ("nothing holds here"), not a parse
    failure, and must not be retried as if the call had errored.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response was not JSON: {e}") from e

    rows = data.get("affordances", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"expected a list of affordances, got {type(rows)}")

    out: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            a_box = int(row["actor_box"])
            r_box = int(row["recipient_box"])
            predicate = row["predicate"]
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if a_box == r_box:
            continue
        if predicate not in predicates:
            continue
        if not 0.0 <= score <= 10.0:
            continue
        actor, recipient = box_of.get(a_box), box_of.get(r_box)
        if actor is None or recipient is None:
            continue
        out[(actor, recipient, predicate)] = score
    return out
