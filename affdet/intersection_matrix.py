"""Tier-1 structural-impossibility mask for OmniObject3D.

A `(actor_class, recipient_class, predicate)` triple is a CANDIDATE if it is
structurally possible per the intersection matrix. Whether the affordance
actually HOLDS for a specific instance is what the model learns (P2); this
mask only removes pairs that no instance could ever satisfy.

Actors:  knife, peeler, fork, scissor (utensils)
         bowl, dish, cup, pan, red_wine_glass (open containers → canContain)
         teapot, vase, kettle, bottle, thermos (pourables → canPourInto)
Predicates: canCut, canPierce, canPeel, canContain, canSupport, canPourInto
Note: canFitOnto / canScoop / canSlideOn skipped — no matching meshes in OmniObject3D.
"""
from __future__ import annotations

# Food categories (recipients for cutting/piercing/peeling/containing).
_FOOD: set[str] = {
    "apple", "banana", "bread", "broccoli", "broccolini", "bun", "cake",
    "carrot", "cauliflower", "chili", "corn", "cucumber", "donut", "dumpling",
    "durian", "egg", "fig", "garlic", "ginger", "hamburger", "hami_melon",
    "hot_dog", "kiwifruit", "lemon", "litchi", "mango", "mushroom", "onion",
    "orange", "peach", "pear", "pineapple", "potato", "pumpkin", "sausage",
    "strawberry", "sweet_potato", "tomato", "watermelon",
}

# Open containers: always canContain food; always canSupport (flat surface).
_OPEN_CONTAINERS: set[str] = {"bowl", "dish", "cup", "pan", "red_wine_glass"}

# Pourables: vessels that participate in canPourInto (either direction). Kept as
# the union for OBJECT_CLASSES; the per-direction actor/recipient gating below is
# what canPourInto actually uses.
_POURABLES: set[str] = {"teapot", "vase", "kettle", "bottle", "thermos"}

# canPourInto is relational (2026-07-05): actor pours FROM, recipient receives INTO.
# Class-level gating = "which classes CAN play each role" (per-instance truth is
# resolved from pourable_from/into in resolved_instances.py). Truth table:
#   from: teapot, kettle, thermos, vase   (bottle pours from nothing)
#   into: bowl, dish, pan, cup, red_wine_glass, teapot, vase  (open tops --
#         these OmniObject3D bottle scans are modeled as sealed, so bottle
#         is ruled out entirely, never sent to the VLM either)
_POUR_FROM_CLASSES: set[str] = {"teapot", "kettle", "thermos", "vase"}
_POUR_INTO_CLASSES: set[str] = _OPEN_CONTAINERS | {"teapot", "vase"}

OBJECT_CLASSES: list[str] = sorted(
    # Utensils
    {"knife", "peeler", "fork", "scissor"}
    # Open containers
    | _OPEN_CONTAINERS
    # Pourables
    | _POURABLES
    # Food
    | _FOOD
)

TIER1_PREDICATES: list[str] = [
    "canCut", "canPierce", "canPeel", "canContain", "canSupport", "canPourInto",
]

_ACTORS: dict[str, set[str]] = {
    "canCut":      {"knife"},
    "canPierce":   {"knife", "fork"},
    "canPeel":     {"peeler"},
    "canContain":  _OPEN_CONTAINERS,
    "canSupport":  _OPEN_CONTAINERS,
    "canPourInto": _POUR_FROM_CLASSES,
}

_RECIPIENTS: dict[str, set[str]] = {
    "canCut":      _FOOD,
    # canPierce was rule-based (true for any knife/fork except against corn)
    # until 72d0ef6; it is now VLM-judged like canCut, so corn is no longer
    # carved out -- whether a tine or tip can enter a cob is a visual call,
    # and pre-excluding it removed the only structural negative the predicate
    # had.
    "canPierce":   _FOOD,
    # canPeel recipients were previously hard-filtered to a fixed hard-skinned
    # subset (_PEEL_FOOD); widened to all food so the VLM judges peelability
    # per-instance instead of a hand-picked pre-filter deciding it up front.
    "canPeel":     _FOOD,
    "canContain":  _FOOD,
    "canSupport":  _FOOD,
    "canPourInto": _POUR_INTO_CLASSES,  # pour into open-top vessels (incl. teapot/vase)
}


def is_candidate_pair(actor_class: str, recipient_class: str, predicate: str) -> bool:
    """True if (actor_class, recipient_class, predicate) is structurally possible."""
    if predicate not in TIER1_PREDICATES:
        return False
    if actor_class not in _ACTORS.get(predicate, set()):
        return False
    if actor_class == recipient_class:
        return False
    return recipient_class in _RECIPIENTS.get(predicate, set())


def candidate_pairs_for_scene(
    object_classes: list[str],
) -> list[tuple[str, str, str]]:
    """All (actor, recipient, predicate) candidates from a list of object classes."""
    pairs: list[tuple[str, str, str]] = []
    for actor in object_classes:
        for recipient in object_classes:
            if actor == recipient:
                continue
            for pred in TIER1_PREDICATES:
                if is_candidate_pair(actor, recipient, pred):
                    pairs.append((actor, recipient, pred))
    return pairs
