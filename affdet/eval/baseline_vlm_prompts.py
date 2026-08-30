"""Prompts for the whole-photo VLM baseline (end-to-end: raw photo in, directed
affordance triples out, no help from our own detector/classifier/MLP).

Distinct from affdet/label/prompts.py, which scores ONE actor/recipient PAIR
per call against a paired composed-scene render (used to generate training
labels). This module instead asks a single VLM call to (a) find every object
in a real photo itself and (b) judge all 6 Tier-1 predicates for every
ordered pair among the objects it found — mirroring what our full pipeline
does end-to-end, so the comparison is fair on BOTH quality and speed.

canPierce and canPourInto are rule-based in the training-label pipeline
(vlm.py excludes them from VLM_PREDICATES: they come from curation metadata
like "this category always pierces" or "this scan's lid is glued shut", not
a visual judgment). A real photo has none of that hidden metadata, so the
baseline asks the VLM to judge them for real too — this measures whether a
VLM can approximate, from vision alone, rules our pipeline gets for free.
"""
from __future__ import annotations

import json
from pathlib import Path

CATEGORY_VOCAB: list[str] = sorted(
    k for k in json.loads(
        (Path(__file__).resolve().parents[2] / "scripts" / "omni" / "curation.json").read_text()
    ) if not k.startswith("_")
)

TIER1_PREDICATES: list[str] = [
    "canCut", "canPierce", "canPeel", "canContain", "canSupport", "canPourInto",
]

_INTRO = """\
You are an expert at judging physical affordances between kitchen objects in \
a real photograph.

You will be shown ONE photo of a real kitchen tabletop scene containing \
several objects. Your job has two parts:

1. Identify every distinct object in the photo. For each object, give it a \
short instance id (e.g. "knife_1", "apple_1" — use "_2" etc. if there are \
multiple objects of the same category) and a category name chosen from the \
fixed vocabulary below. If an object doesn't cleanly match any category, \
pick the closest one — do not invent a new category name.

2. For EVERY ordered pair of distinct objects you found (actor, recipient — \
order matters, "actor acts on recipient"), judge whether each of the 6 \
affordances below holds, based on what is ACTUALLY VISIBLE — the real \
relative size, shape, and proportions of these specific objects in this \
photo — not a generic assumption about what a "typical" object of that \
category can do.

Allowed categories: {categories}\
"""

_PREDICATE_DEFS = """\
The 6 affordances to judge for every ordered pair:

canCut — Can the actor CUT the recipient (slicing, chopping)?
canPierce — Can the actor PIERCE the recipient (stabbing, skewering)?
canPeel — Can the actor PEEL the recipient (removing skin/rind)?
canContain — Can the actor CONTAIN the recipient (recipient fits inside, \
enclosed by walls on at least 3 sides)?
canSupport — Can the actor SUPPORT the recipient (recipient rests stably on \
a flat surface, upright)?
canPourInto — Can the actor POUR INTO the recipient (actor has a real \
visible spout/pour lip, recipient has a real opening to receive it)?\
"""

_CALIBRATION = """\
Calibration notes — read carefully, these are the most common mistakes:

canCut: first identify what TYPE of "knife" or blade this is. A real \
kitchen knife (chef's/paring/utility, proper handle + substantial blade) can \
cut food well. A folding/pocket knife, box cutter, or decorative/toy knife \
is NOT a good food-cutting tool even if technically bladed — judge the \
capability of the type, not just "is there a blade."

canPierce: requires a real POINT (knife tip, fork tine, skewer tip) that \
could penetrate the recipient's surface. A butter knife, spoon, or blunt \
tool does not pierce even if it is otherwise knife-shaped.

canPeel: only a purpose-built peeling tool (vegetable peeler) or a real \
kitchen knife scores well here. Thick/hard rinds (pineapple, watermelon) cap \
the score low even for a good peeler.

canContain: the recipient must physically fit THROUGH the actor's opening \
AND sit enclosed by walls on at least 3 sides. A flat plate does NOT \
contain — food just rests on top with nothing surrounding it, that is \
canSupport instead, not canContain. Check the opening size against the \
recipient's actual size in the photo, not a category assumption.

canSupport: the recipient must rest stably on a flat, wide-enough surface \
without sliding, rolling, or overhanging the edge. A cup, mug, or glass — \
narrow rim, no flat top — is essentially NEVER a canSupport actor for \
anything, regardless of the food's size. Size mismatch (recipient clearly \
bigger than the actor's surface) overrides any category fit: a small dish \
under a large object always scores this false, however well-suited the \
dish would otherwise be.

canPourInto: the actor needs a REAL VISIBLE spout, pour lip, or narrow \
directed opening — not just "is a container." A plain-rimmed mug, bowl, or \
cup has NO spout and should score this FALSE even though it can hold \
liquid — pouring from it would slosh over the whole rim, not go in a \
controlled stream. Only score TRUE when you can actually see a spout-like \
feature on the actor.\
"""

_OUTRO = """\
Respond with a single JSON object, no other text, in exactly this shape:

{{
  "objects": [{{"id": "knife_1", "category": "knife"}}, ...],
  "affordances": [
    {{"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut", "verdict": 1}},
    ...
  ]
}}

Include an entry in "affordances" for EVERY ordered pair of distinct objects \
you found, crossed with all 6 predicates above (verdict is 1 for true, 0 \
for false — judge every one, do not omit predicates you think are obviously \
false). Do not include any object paired with itself.\
"""


def build_baseline_system_prompt() -> str:
    return "\n\n".join([
        _INTRO.format(categories=", ".join(CATEGORY_VOCAB)),
        _PREDICATE_DEFS,
        _CALIBRATION,
        _OUTRO,
    ])


def build_baseline_user_prompt() -> str:
    return (
        "This photo shows a real kitchen tabletop scene. Find every object, "
        "then judge all 6 affordances for every ordered pair among the "
        "objects you found, as described in the system prompt."
    )


def parse_baseline_response(raw: str) -> dict:
    """Strip an optional ```json ... ``` fence, then parse."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)
