"""Prompt builders for the E7 scene-level VLM arm.

Distinct from `prompts.py`, which labels ONE pair from a composed two-object
render ("You will be shown ONE image: a real composed scene with two objects").
E7 asks a different question: a real multi-object scene, every object boxed and
numbered, one call, a full table of judgements back.

The calibration text itself is reused verbatim from `prompts.py` — the scoring
rubric, the knife-type caps, the container/support size rules are the same
physics and must not drift between the two settings, or a student-vs-VLM
comparison would be measuring prompt differences.

Three arms differ only in what the VLM is told about the boxes:

  gplus   boxes numbered AND named ("[3] knife"). The student's gate runs on
          gold classes here, so naming the boxes is what makes the two sides
          symmetric.
  g       boxes numbered, NOT named. The student's gate runs on the DINOv2
          probe's prediction, so the VLM must likewise infer what each object
          is from pixels.
  e       no boxes at all, raw scene image. Both sides must find and identify
          the objects themselves.

Every arm receives the class VOCABULARY. Withholding it would not be
neutrality: the taxonomy is baked into the student three times over (probe
output space, gate lookup table, training labels), and an unconstrained VLM
invents its own category names, which makes the two outputs unjoinable.
"""
from __future__ import annotations

from affdet.label.prompts import (  # noqa: F401
    _CALIBRATION, _INTRO, _novelty_block,
)

#: Arms. See module docstring.
ARMS = ("gplus", "g", "e")

_SCENE_INTRO = """\
You are an expert at judging physical affordances between kitchen objects.

You will be shown ONE photograph of a real scene: several objects placed
together on a table. This is a single real arrangement, not a grid of
variations — every object you see is present at its true relative size.

Judge each requested affordance using what is ACTUALLY VISIBLE — the real
relative size, shape and proportions of these specific object instances — not
a generic assumption about what a "typical" object of that category looks
like. If this particular container is unusually small, or this particular food
unusually large, judge from that, not from a category stereotype."""

_BOXED_NAMED = """\
Each object is outlined with a coloured box and labelled with a number and its
category, e.g. "[3] knife". Refer to objects ONLY by their number."""

_BOXED_UNNAMED = """\
Each object is outlined with a coloured box and labelled with a number, e.g.
"[3]". The categories are NOT given — work out what each object is from the
image. Refer to objects ONLY by their number."""

_UNBOXED = """\
The objects are not outlined or labelled. Identify them yourself and refer to
each by its category name from the vocabulary below. If two objects share a
category, disambiguate by position (e.g. "knife on the left")."""

_SCORING = """\
Score 0-10 for each affordance you are asked about:
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


def build_scene_system_prompt(predicates: list[str], arm: str = "gplus") -> str:
    """System prompt for one scene-level call.

    Carries the same per-predicate calibration blocks as the pair labeller, so
    the physics being judged is identical and only the presentation differs.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    reference = {"gplus": _BOXED_NAMED, "g": _BOXED_UNNAMED,
                 "e": _UNBOXED}[arm]
    # The canCut block carries a "{novelty_block}" placeholder that must be
    # substituted, exactly as prompts.build_system_prompt does. Until
    # 2026-08-18 this call inserted the raw block, so arms G+/G/E sent the VLM
    # the literal token "{novelty_block}" instead of the novelty/toy-knife
    # warning -- see data/eval/E7_LISTALL_RESULTS.md for what the fix
    # supersedes.
    novelty = _novelty_block()
    blocks = [_CALIBRATION[p].replace("{novelty_block}", novelty)
              for p in sorted(predicates) if p in _CALIBRATION]
    return "\n\n".join([_SCENE_INTRO, reference, _SCORING, *blocks])


def _fmt_question(q: dict, arm: str) -> str:
    """One row of the requested-judgements list, in the arm's referring style."""
    if arm == "e":
        return (f"  {q['actor_label']} -> {q['recipient_label']}"
                f"  : {q['predicate']}")
    return (f"  [{q['actor_box']}] -> [{q['recipient_box']}]"
            f"  : {q['predicate']}")


#: Extra output fields the PAIRWISE labeller requires but the scene adaptation
#: originally dropped. `prompts.py` keeps them on the recorded grounds that
#: articulating the judgement explicitly measurably stabilises it; E8's `typed`
#: variant re-added the same mechanism to the list-all arm and was the best
#: variant tested. Opt-in, so every committed E7 number stays reproducible.
_FIELD_SPEC = {
    "canCut": ('"type"', '  - canCut carries "type", exactly one of: "kitchen '
               'knife", "folding/pocket knife", "box cutter", '
               '"decorative/toy" -- the classification you used to cap the '
               'score. Without stating it, scores drift toward an uncommitted '
               'middle value instead of the type\'s actual cap.'),
    "canContain": ('"enclosure"', '  - canContain carries "enclosure", a short '
                   'phrase on what you checked: whether the food fits through '
                   'the opening AND whether the walls enclose its full length '
                   '(e.g. "fits opening but sticks out above the rim").'),
    "canSupport": ('"size_note"', '  - canSupport carries "size_note", a short '
                   'phrase on the relative size you observed (e.g. "food much '
                   'larger than container").'),
}


def build_scene_user_prompt(questions: list[dict], vocabulary: list[str],
                            arm: str = "gplus", fields: bool = False) -> str:
    """User turn listing every judgement wanted for this scene.

    `questions` carries per-row `actor_box`/`recipient_box` (int, for the boxed
    arms) or `actor_label`/`recipient_label` (str, for arm "e"), plus
    `predicate`. Asking for the whole table in one call — rather than one call
    per pair — matches the shape the student produces and keeps the VLM's cost
    per scene comparable.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    lines = [
        "Object categories that may appear in this scene (the complete "
        "vocabulary):",
        "  " + ", ".join(sorted(vocabulary)),
        "",
        f"Score exactly the following {len(questions)} judgement(s). Each is "
        "already known to be structurally possible for the object classes "
        "involved; your job is to judge whether it holds for THESE SPECIFIC "
        "INSTANCES at their visible sizes.",
        "",
    ]
    lines += [_fmt_question(q, arm) for q in questions]
    asked = sorted({q["predicate"] for q in questions})
    extra = [_FIELD_SPEC[p] for p in asked if p in _FIELD_SPEC] if fields else []

    lines += [
        "",
        "Respond with a single JSON object with one key, \"judgements\", whose "
        "value is a list with EXACTLY one entry per requested judgement, in "
        "the same order. Each entry is:",
    ]
    ident = ('"actor": "<name as given>", "recipient": "<name as given>"'
             if arm == "e" else
             '"actor_box": <int>, "recipient_box": <int>')
    if extra:
        # One template per predicate: a single line listing every extra key at
        # once would be invalid JSON and would ask for fields the predicate
        # does not take.
        lines.append(f'  {{{ident}, "predicate": "<name>", "score": <0-10>}}')
        lines.append("...except these predicates, which carry ONE extra field:")
        for name, _ in extra:
            lines.append(f'  {{{ident}, "predicate": "<name>", '
                         f'{name}: "<short phrase>", "score": <0-10>}}')
        lines.append("")
        lines.append("State the extra field BEFORE the score, and let it "
                     "determine the number:")
        lines += [block for _, block in extra]
        lines.append("Predicates not listed above carry the score only.")
    else:
        lines.append(f'  {{{ident}, "predicate": "<name>", "score": <0-10>}}')
    lines += [
        "",
        "Return the score only — no rationale, no extra keys, no commentary "
        "outside the JSON." if not extra else
        "Return the JSON only — no commentary outside it.",
    ]
    return "\n".join(lines)
