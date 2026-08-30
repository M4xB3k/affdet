"""Balanced scene sampling for the list-all training experiment.

Draws N scenes from the training corpus for relabelling with the list-all arm,
balanced across (affordance, polarity) slots rather than taking the corpus's
natural proportions -- a plain random draw would inherit whatever skew the
renderer's coverage sampler left behind.

The `exclude` parameter is the important one. The E7 evaluation corpus
(`omni_scenes_e2e_merged`) must never enter training: the retrained model is
scored on those 110 scenes against 593 human judgements, and a leak would
inflate that score with no visible symptom. Scene ids are disjoint by
construction (`scene_v1_*` vs `scene_e2e_*`, verified 0 intersection), but the
guard is passed explicitly rather than relying on a naming convention.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict


def slot_counts(scenes: list[dict]) -> dict[tuple[str, str], int]:
    """How many scenes realise each (affordance, polarity) slot.

    A scene realising several slots counts toward each of them, so the totals
    sum to more than len(scenes).
    """
    c: Counter = Counter()
    for s in scenes:
        for aff, pol in (s.get("realised_affordances") or []):
            c[(aff, pol)] += 1
    return dict(c)


def balanced_sample(scenes: list[dict], n: int, seed: int = 0,
                    exclude: set[str] | None = None) -> list[dict]:
    """Draw `n` scenes, spread as evenly as possible over the slots.

    Round-robins over the slots, taking one unused scene from each in turn, so
    an under-represented slot is not drowned out by a common one. Scenes
    realising nothing are skipped -- they carry no signal for any slot and
    would consume a draw for nothing.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    exclude = exclude or set()
    rng = random.Random(seed)

    by_slot: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in scenes:
        if s["scene_id"] in exclude:
            continue
        slots = s.get("realised_affordances") or []
        if not slots:
            continue
        for aff, pol in slots:
            by_slot[(aff, pol)].append(s)
    for v in by_slot.values():
        rng.shuffle(v)

    picked: dict[str, dict] = {}
    slots = sorted(by_slot)
    cursor = {k: 0 for k in slots}
    progress = True
    while len(picked) < n and progress:
        progress = False
        for slot in slots:
            if len(picked) >= n:
                break
            bucket = by_slot[slot]
            i = cursor[slot]
            while i < len(bucket) and bucket[i]["scene_id"] in picked:
                i += 1
            cursor[slot] = i
            if i < len(bucket):
                s = bucket[i]
                picked[s["scene_id"]] = s
                cursor[slot] = i + 1
                progress = True
    return list(picked.values())
