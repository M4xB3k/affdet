"""Prompt constants for the OmniObject3D VLM labeler (paired-scene renders —
actor + recipient shown together in ONE real composed scene, see
affdet/render/omni_pair_render.py, not two independently auto-zoomed tiles)."""
from __future__ import annotations

AFFORDANCE_QUESTION: dict[str, str] = {
    "canCut":     "Can the actor CUT the food item? (slicing, chopping)",
    "canPierce":  "Can the actor PIERCE the food item? (stabbing, skewering)",
    "canPeel":    "Can the actor PEEL the food item? (removing skin/rind)",
    "canContain": "Can the actor CONTAIN the food item? (food fits inside)",
    "canSupport": "Can the actor SUPPORT the food item on its surface?",
}


_INTRO = """\
You are an expert at judging physical affordances between kitchen objects.

You will be shown ONE image: a real composed scene with two objects placed
together on a table, photographed from multiple camera angles arranged in a
grid (so you can judge their TRUE relative size, shape, and proportions from
several viewpoints — not a guess based on category alone). The two objects
are always the same across every tile in the grid; only the camera angle
changes.

Judge each requested affordance using what is ACTUALLY VISIBLE in the image —
the real relative size, shape, and proportions of these two specific object
instances — not a generic assumption about what a "typical" object of that
category looks like. If the image shows this particular container is
unusually small, or this particular food is unusually large, judge based on
that, not on a category stereotype.

You will be told which affordance(s) are worth asking about for this specific
pair — only score the ones you are asked about. Every one of those has
already been confirmed structurally possible for this actor/recipient class
combination; your job is to judge whether it holds for THESE SPECIFIC
INSTANCES, based on what the image shows.

Score 0–10 for each requested affordance:
  0    = physically impossible for these specific instances — food cannot fit
         through the opening, or the surface cannot support any weight
  1–3  = extremely poor — technically possible for this actor/recipient class
         pairing, but this specific instance combination is so badly sized or
         shaped it barely works at all
  4–5  = poor — works but with real difficulty; not the right fit for this pairing
  6–7  = acceptable — functional but clearly not purpose-built for this exact pairing
  8–9  = good to excellent — this is the natural, well-fitted use of this actor
         on this recipient, given their actual visible sizes
  10   = perfect — ideal fit in every way, given what is visible in the image

IMPORTANT: A well-fitted, correctly-sized pairing doing its natural job scores
8–9. Reserve 5–6 for "technically works, but a poor match." Do NOT anchor 5 as
"average" — anchor 8–9 as "this specific pairing works well."\
"""

# Per-predicate calibration block. Included in the system prompt ONLY for
# predicates actually being requested this call — a dish/durian pair only
# ever needs canContain + canSupport, so it should never see canCut/canPierce/
# canPeel calibration text at all (wasted tokens, and noise the model has no
# reason to read).
_CALIBRATION: dict[str, str] = {
    "canCut": """\
canCut calibration — FIRST identify what TYPE of "knife" this actually is,
THEN score based on that type. The dataset mixes several distinct types that
are NOT equally good at cutting food, regardless of blade sharpness alone:

  - real kitchen knife (chef's knife, paring knife, utility knife): a proper
    handle for a firm cutting grip, a substantial exposed blade meant for
    food prep → these are the ONLY type that should score 8-9 for a
    well-sized food pairing.
  - folding/pocket knife: blade PIVOTS/HINGES out of the handle (look for a
    visible hinge point or pivot pin where a distinct blade meets a distinct
    handle), often short even when open, designed for general utility not
    food prep → cap at 5-6 even for a good size match, since the grip/blade
    geometry is wrong for kitchen use.
  - box cutter / utility/retractable-blade knife: blade SLIDES/RETRACTS
    straight into a one-piece plastic or metal housing (no separate hinged
    handle — look for a thumb-slide track, snap-off segmented blade notches,
    or a smooth continuous body with no pivot point), designed for cutting
    cardboard/packaging, NOT food → cap at 2-3 regardless of the food's size
    or how sharp the blade might be, since this is fundamentally the wrong
    tool for food prep, not a smaller version of a kitchen knife. When the
    blade is fully retracted and no hinge is visible, default to box cutter
    over folding knife unless you can clearly see a pivot/hinge.
  - decorative/toy/plastic knife: not a functional blade at all (display
    piece, child's toy, costume prop) → cap at 0-2 regardless of shape,
    since it cannot actually cut anything.

{novelty_block}

TIP SHAPE IS NOT A CANCUT SIGNAL: whether the tip is pointed or blunt/squared
off (e.g. a bread knife, a santoku, a cleaver) has NO bearing on canCut —
cutting is the BLADE EDGE slicing through food, not the tip stabbing in. A
squared-off or rounded tip is a canPierce consideration (scored separately),
never a reason to cap canCut. Do not let tip shape lower a canCut score.

IMPORTANT — judge the CAPABILITY of the object, not its momentary pose. If a
folding/pocket knife or utility knife appears closed/retracted in this
render, judge it AS IF OPENED: classify its type and size from the handle,
hinge, and any visible blade stub, then score based on what that type of
tool could do once opened. Do NOT score a knife 0 just because the blade
happens to be closed in this particular snapshot — a folding knife that
would have a decent blade once open still scores in its type's normal range
(e.g. 5-6 for a folding/pocket knife), not 0. Only score 0 if the object is
not knife-like at all, or is so small/thin that even fully open it could not
meaningfully cut.

Only AFTER classifying the type should you also weigh the size/shape fit to
this specific food (as in a normal calibration). A real kitchen knife that's
merely a bit short for a large food still scores 6-7, not below — the TYPE
cap only applies to non-kitchen-knife types.

Examples:
  large chef's knife × onion              → 9  (real kitchen knife, purpose-built, correct size)
  large chef's knife × watermelon         → 7  (real kitchen knife, works but needs sawing)
  small kitchen paring knife × apple      → 7  (real kitchen knife, a bit small but still proper grip/blade)
  folding pocket knife (open) × apple     → 5  (right size, but folding-knife type caps it)
  folding pocket knife (closed) × apple   → 5  (same as open — judge capability, not current pose)
  box cutter / utility knife × apple      → 2  (wrong tool type entirely, capped low regardless of size)
  decorative dagger × apple               → 2  (display/costume type, not a real food knife)\
""",
    "canPierce": """\
canPierce definition:
  "Pierce" means the actor's tip can puncture/stab into the food to start a
  hole (skewering, stabbing) — a DIFFERENT capability from canCut's slicing
  motion. A tool can be excellent at one and poor at the other.

canPierce calibration — the tip shape is the whole question:
  - a tool that narrows to an actual POINT (a real knife's tip, a fork's
    tines, a skewer/pick) → 8-9 for a food soft/small enough to be pierced by
    that point, regardless of blade type otherwise (even a box-cutter or
    folding-knife TIP can still puncture — canCut's type-cap does NOT apply
    here, only tip sharpness matters for piercing).
  - a BLUNT or ROUNDED tip (no taper at all, e.g. a butter knife, a spreader,
    a novelty/toy knife with a rounded end) → 0-1, since a blunt tip cannot
    break the food's surface no matter how the rest of the object is shaped.
    This is the same blunt-tip cue used for canCut's novelty-knife check —
    if you already flagged this object as toy/novelty there, it almost
    certainly caps low here too.
  - fork: the tines are real points → 8-9 for most food, lower (4-5) only if
    the food is clearly too hard/large for the tine length to penetrate
    meaningfully (e.g. a whole watermelon).
  - corn: judge it like any other food — can the tip/tine enter a kernel or
    the cob? Score normally on the same tip-sharpness basis as everything
    else, it is not a special case.

BLADE COVERS / SHEATHS / RETRACTABLE HOUSINGS: some renders show the blade
partly or fully covered by a molded guard, sheath, or retractable housing
(a rounded plastic/metal shell the blade sits inside, sometimes with just
the tip peeking out, sometimes almost fully retracted with barely a sliver
visible). Judge CAPABILITY, not the momentary retracted/covered state —
the same principle canCut already uses for a closed folding knife. Do NOT
read a cover's own rounded outer shape as the blade's tip shape, and do NOT
score lower just because little blade is currently showing. If the object
is recognizable as a normal-bladed knife/utility-knife/box-cutter type
(a rigid housing with a slide-out or retractable blade track — the same
type family canCut's calibration identifies), treat the tip as pointed
(8-9 range) regardless of how much blade is extended in this particular
render — these tools are covered/retracted for safety or storage, not
because the blade itself is blunt, and the blade is still there whether or
not it's currently extended. Only score low if this is a decorative/novelty
item you've already flagged as toy-shaped elsewhere (a cover on a toy knife
doesn't make the underlying toy blade real), or if the exposed portion
itself is visibly rounded with no hint of a blade edge anywhere.

Examples:
  chef's knife (pointed tip) × apple       → 9  (sharp point, easy puncture)
  butter knife / spreader (blunt) × apple  → 1  (no point at all, cannot start a hole)
  fork × grape                              → 9  (tines pierce easily)
  fork × watermelon (whole)                 → 4  (tines too short/soft food too tough at that scale)
  box cutter (pointed tip, retracted body) × tomato → 7  (tip is a real point even though
    the blade type caps canCut low -- piercing only needs the tip)\
""",
    "canPeel": """\
canPeel calibration:
  vegetable peeler × apple        → 9  (purpose-built, right size)
  vegetable peeler × orange       → 7  (works but citrus rind is thicker)
  vegetable peeler × pineapple    → 2  (skin too thick and irregular)
  vegetable peeler × watermelon   → 0  (rind far too thick and hard)\
""",
    "canContain": """\
canContain definition:
  "Contain" means the food physically fits THROUGH the opening of the actor
  AND stays WITHIN the actor's footprint/boundary once placed, rather than
  just balancing on top with nothing around it. A truly FLAT plate (no rim
  at all) does NOT contain food — the food just rests on top with nothing
  around it. But a dish or bowl with ANY visible rim, even a shallow one,
  DOES contain food that sits within that rim, even if the walls are low.
  Score based on:
  (1) Can the food pass through the opening, given the sizes actually shown in
      the image? If clearly no → score 0.
  (2) Does the object have a rim/edge at all (however shallow), and does the
      food sit within it rather than spilling over the boundary? A shallow
      dish with a real rim holding a small/medium food within its edge is
      still genuine containment — score based on how well the food fits
      within the rim, not on wall HEIGHT alone.
  Judge the opening size and food size from the image itself, not from a
  category assumption about typical bowl/food proportions.

canContain calibration:
  bowl (wide opening, deep walls) × small-to-medium fruit → 9  (fits easily, walls enclose)
  bowl × food clearly wider than the opening               → 2  (too wide to fit through)
  shallow dish (has a rim) × food that sits within the rim  → 6–8  (real containment,
    just shallower than a bowl — do not score this as low as a truly flat plate)
  truly FLAT plate (no rim at all) × any food               → 1  (no boundary — food just
    rests on the surface with nothing around it; this is the ONLY case that scores this low)
  cup (narrow opening) × food clearly narrower than opening→ 8  (fits through, walls enclose it)
  cup (narrow opening) × food clearly wider than opening   → 0  (cannot fit through)
  frying pan (flat, low walls) × any food                  → 3  (almost no walls, food barely enclosed)
  red wine glass (narrow bowl) × small food that fits       → 6  (fits into the glass bowl, somewhat enclosed)
  bowl/cup × food that is LONG/ELONGATED relative to the container's depth
    (e.g. a long stalk of broccolini standing in a shallow bowl) → 2–3
    (the opening may be wide enough for the food's cross-section, but if the
    food's LENGTH is clearly greater than the container's depth, most of it
    sticks out above the rim uncontained — walls only enclose the bottom
    portion, not "the food" as a whole; do not score this high just because
    the food touches down inside)\
""",
    "canSupport": """\
canSupport definition:
  "Support" means the food rests stably ON TOP of the actor's flat surface
  (upright orientation). This is DIFFERENT from canContain. Ask: is the
  surface flat and wide enough, GIVEN THE ACTUAL SIZES SHOWN, that the food
  sits stably without sliding, rolling, or overhanging the edge?

CRITICAL — compare the ACTUAL VISIBLE SIZE of the food against the actual
visible size of the actor's surface, measured in the SAME image, before
scoring. This is the single most common scoring mistake: a small dish/bowl/
pan can look "purpose-built" in isolation, but if the food shown is clearly
BIGGER than the container's surface (wider, taller, or overhanging on most
sides), no amount of "this is the right kind of object for this affordance"
reasoning can score it above 3. Size mismatch overrides category fit —
always check size FIRST, then refine within the size-appropriate band.

IMPORTANT — judge the object's CAPABILITY, not its momentary pose. If a
container (bowl, dish, cup, pan) happens to be rendered tipped over, on its
side, or upside-down, this is a render artifact, not a real property of the
object. Mentally right the object to its normal UPRIGHT position and judge
based on that — its real size and shape once upright — not the accidental
pose shown. Do not penalize or zero a score just because the object looks
tipped over in this particular snapshot.

canSupport calibration:
  flat dish / plate × food that fits well within its edge     → 9  (purpose-built flat surface, rests perfectly)
  flat dish / plate × food roughly same size as the dish       → 5  (borderline — likely overhangs a little)
  flat dish / plate × food CLEARLY LARGER than the dish        → 2  (severe overhang, unstable regardless of dish quality — e.g. a small saucer under a whole watermelon)
  frying pan × food that fits well within its edge              → 9  (wide flat cooking surface, very stable)
  frying pan × food clearly larger than the pan's surface       → 2  (same size-mismatch rule applies)
  bowl (upright) × small round food                              → 8  (nestles stably in the curved surface)
  bowl (upright) × long/elongated food                            → 5  (curved surface + long shape = tends to roll off)
  bowl (upright) × food clearly larger than the bowl's rim        → 2  (overhangs the rim entirely — very unstable)
  bowl (tipped/on its side in render) × food that would fit if upright → 8  (judge as if upright; the pose is a render artifact)
  cup or red wine glass (upright) × any food                      → 0  (rim too narrow for anything to rest on top)
  mug, travel mug, or any cup/glass with a LID or closed top × any food → 0
    (a closed/lidded top has NO open surface at all for food to rest on —
    this is an even more certain 0 than an open narrow rim; look carefully
    at the top of the object before scoring, do not assume "cup-shaped
    object" means it has a flat resting surface)\
""",
    # canPourInto has no calibration block: it's excluded from VLM calls
    # entirely (see affdet/label/vlm.py's VLM_PREDICATES) — its truth is
    # rule-based (resolved_instances.py's pourable_from/pourable_into, keyed
    # on curation facts like a lidded scan or a sealed bottle), not a visual
    # judgment a model looking at a render could make.
}

_OUTRO_HEAD = """\
Judge the SPECIFIC objects shown in this image — their actual visible size,
depth, and shape relative to each other, using every camera angle in the
grid to cross-check your judgment (not just one view).

Respond with a single JSON object mapping each requested affordance name to
its integer score (0-10) — no reasoning or explanation text, score only —\
"""

_OUTRO_CANSUPPORT_EXCEPTION = """\
  - canSupport is a dict {"score": N, "size_note": "..."} (a short phrase on
    the relative size you observed, e.g. "food much larger than container").\
"""

_OUTRO_CANCONTAIN_EXCEPTION = """\
  - canContain is a dict {"score": N, "enclosure": "..."} (a short phrase on
    what you actually checked: whether the food fits through the opening,
    AND whether the container's walls/depth enclose the food's full length —
    e.g. "fits opening but sticks out above the rim" or "fully enclosed,
    walls on all sides"). Without stating this explicitly, scores drift
    toward "food touches down inside = contained" and miss cases where a
    long food is mostly UNenclosed even though its base sits inside.\
"""

_OUTRO_CANCUT_EXCEPTION = """\
  - canCut is a dict {"type": "...", "score": N} where "type" is exactly one
    of: "kitchen knife", "folding/pocket knife", "box cutter",
    "decorative/toy" — the classification you used to cap the score. Without
    stating this explicitly, scores drift toward an uncommitted middle value
    instead of the type's actual cap, so always include it.\
"""

_OUTRO_TAIL = """\
Only include keys for the affordances you were asked about.\
"""


#: The long novelty/toy discriminator. It names a cue (bulbous handle,
#: keychain hole) and in the same breath tells the model to DISCOUNT a related
#: cue (pale colour) as a rendering artifact. Since canCut's score is a
#: discrete branch on knife type -- kitchen 8-9 vs toy 0-2 -- a wobbly
#: discriminator makes the whole score flip between passes rather than vary
#: smoothly: knife_017 x apple_017 returned [8,2,2,2,2] over five identical
#: calls. Set AFFDET_TERSE_NOVELTY=1 to swap in the short form, which states
#: the cap and leaves the judgement to the examples.
_NOVELTY_LONG = """WATCH FOR: a knife-SHAPED object with no hinge and no retractable slide can
still be a novelty/toy, not a real kitchen knife. The renders in this
dataset often desaturate real steel to a flat pale gray/blue — do NOT use
color or material flatness ALONE as evidence of "toy", since that is a
rendering artifact that affects real knives too. Only flag novelty/toy when
you see a SHAPE tell specific to novelty items, independent of color:
a distinctly bulbous/round handle bulb (clearly rounder than an ergonomic
grip — think spreader or butter-knife silhouette), or a visible
keychain-style hole punched through the handle. Either of those shape tells
alone is enough to cap at 1-2. A merely pale or plain-colored handle, with
an otherwise normal elongated grip shape, is NOT enough on its own — that is
still consistent with a real kitchen knife."""

_NOVELTY_TERSE = """WATCH FOR: a knife-SHAPED object can still be a novelty/toy rather than a
real kitchen knife. Judge this from the overall silhouette, and only cap low
when the object clearly reads as non-functional."""


def _novelty_block() -> str:
    import os
    return (_NOVELTY_TERSE if os.environ.get("AFFDET_TERSE_NOVELTY")
            else _NOVELTY_LONG)


def _build_outro(predicates: list[str]) -> str:
    """canContain/canSupport/canCut each keep an extra reasoning field in
    their output schema (validated: articulating the judgment explicitly
    measurably stabilizes each), but ONLY when that predicate is actually
    requested — otherwise its exception language (e.g. "kitchen knife")
    leaks into unrelated calls like a canPeel-only request."""
    exceptions = []
    if "canContain" in predicates:
        exceptions.append(("canContain", _OUTRO_CANCONTAIN_EXCEPTION))
    if "canSupport" in predicates:
        exceptions.append(("canSupport", _OUTRO_CANSUPPORT_EXCEPTION))
    if "canCut" in predicates:
        exceptions.append(("canCut", _OUTRO_CANCUT_EXCEPTION))

    if not exceptions:
        return "\n".join([_OUTRO_HEAD + " score only.", "", _OUTRO_TAIL])

    names = " and ".join(name for name, _ in exceptions)
    head = (
        f"{_OUTRO_HEAD}\nEXCEPT {names}, which keep one extra field each "
        "because articulating that judgment explicitly measurably improves "
        "scoring reliability:"
    )
    blocks = "\n".join(block for _, block in exceptions)
    examples = {
        "canContain": '"canContain": {"score": 2, "enclosure": "fits opening but sticks out above the rim"}',
        "canSupport": '"canSupport": {"score": 2, "size_note": "food much larger than container"}',
        "canCut": '"canCut": {"type": "box cutter", "score": 2}',
    }
    example_fields = ", ".join(examples[name] for name, _ in exceptions)
    return "\n".join([
        head, blocks,
        "All other predicates stay score-only.",
        f"Example for a pair where {names} " +
        ("was" if len(exceptions) == 1 else "were") + " requested:",
        f"{{{example_fields}}}",
        "", _OUTRO_TAIL,
    ])


def build_system_prompt(predicates: list[str]) -> str:
    """System prompt scoped to ONLY the predicates being requested this call —
    a dish/durian pair (canContain + canSupport candidates only) never sees
    the canCut/canPierce/canPeel calibration blocks, since those affordances
    are never even asked about for that pair."""
    blocks = [_CALIBRATION[p].replace("{novelty_block}", _novelty_block())
              for p in predicates if p in _CALIBRATION]
    return "\n\n".join([_INTRO, *blocks, _build_outro(predicates)])


def build_user_prompt(
    actor_class: str, recipient_class: str, predicates: list[str],
    grid_rows: int, grid_cols: int,
    solo_tile_bottom_row: bool = False,
) -> str:
    """User-turn text for one paired-scene labeling call — describes the
    actual scene (not just bare category labels) and lists exactly which
    affordances to score, using the human-readable AFFORDANCE_QUESTION
    phrasing so the model knows what each predicate means.

    `solo_tile_bottom_row=True` describes the canCut/canPierce composed grid
    (see compose_cancut_grid in vlm.py) instead of the standard all-paired
    grid: top row = 2 real paired-scene angles, bottom row = solo close-up
    tiles of the actor and recipient alone, for unambiguous type/tip
    classification. That layout is NOT 4 identical paired shots, so the
    standard "same two objects appear in every tile" framing would mislead
    the model about what it's looking at in the bottom row.
    """
    n = grid_rows * grid_cols
    questions = "\n".join(
        f"  - {p}: {AFFORDANCE_QUESTION.get(p, p)}" for p in predicates
    )
    if solo_tile_bottom_row:
        scene_desc = (
            f"This image shows a {actor_class} (the actor) and a "
            f"{recipient_class} (the recipient) in a 2x2 grid: the TOP row "
            f"is 2 real camera angles of them placed together on a table "
            f"(use these to judge their true relative size), and the BOTTOM "
            f"row is a close-up solo tile of the {actor_class} alone (left) "
            f"and the {recipient_class} alone (right), zoomed in for an "
            f"unambiguous view of shape/type/tip details that a small "
            f"paired-angle tile can lose. Use the top row for size/scale "
            f"and the bottom row for close-up shape details — they show the "
            f"same two objects, not four different scenes."
        )
    else:
        scene_desc = (
            f"This image shows a real composed scene: a {actor_class} (the actor) "
            f"and a {recipient_class} (the recipient) placed together on a table, "
            f"shown from {n} different camera angles arranged in a {grid_rows}x{grid_cols} "
            f"grid (varying elevation and azimuth) so you can judge their true "
            f"relative size and shape from multiple viewpoints. The same two "
            f"objects appear in every tile."
        )
    return (
        f"{scene_desc}\n\n"
        f"For this {actor_class}/{recipient_class} pair, score ONLY the "
        f"following affordance(s) — these are the only ones structurally "
        f"possible for this actor/recipient class combination:\n{questions}\n\n"
        f"Respond with a single JSON object as described in the system prompt."
    )
