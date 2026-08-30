"""Instance-disjoint split for the detector generalization arm (E5 D-b).

Holding out *scenes* at random asks a weak question: the same mesh recurs
across many scenes, so a detector can memorise every instance it will be
tested on and still score well. That measures robustness to rearrangement,
not to novel objects.

This module holds out *mesh instances*. Every scene containing a reserved
mesh is pulled out of training entirely, so at test time the detector meets
bowl meshes it has genuinely never seen, having seen other bowls. That is
the claim the thesis can defend for the detector, and it mirrors the
instance-level rung of the affordance model's own generalization ladder.

Holdout is stratified by class: each class contributes its own share, so
the test set cannot collapse into a single category.
"""
from __future__ import annotations

import random
from typing import Any, Iterable, Mapping


def held_out_instances(
    instances_by_class: Mapping[str, Iterable[str]],
    frac: float = 0.2,
    seed: int = 0,
    min_train: int = 2,
) -> set[str]:
    """Reserve ``frac`` of each class's mesh instances.

    A class keeps at least ``min_train`` instances in training: with fewer,
    "generalizes to a novel instance of this class" is not a claim the data
    can support, and holding one out anyway would produce a number that
    looks like evidence but is not. Such classes contribute nothing to the
    holdout and are reported as untestable instead.

    At least one instance is held out whenever a class can afford it, so a
    small class is not silently rounded out of the test set entirely.
    """
    rng = random.Random(seed)
    held: set[str] = set()

    for cls in sorted(instances_by_class):
        members = sorted(instances_by_class[cls])
        n_hold = max(1, round(len(members) * frac))
        if len(members) - n_hold < min_train:
            continue
        held.update(rng.sample(members, n_hold))

    return held


def scenes_containing(
    scenes: Mapping[str, Any],
    instances: set[str],
) -> set[str]:
    """Scene ids holding at least one of ``instances``."""
    if not instances:
        return set()
    return {
        sid for sid, scene in scenes.items()
        if any(o.obj_id in instances for o in scene.objects)
    }


def split_scenes_by_instance(
    scenes: Mapping[str, Any],
    held: set[str],
) -> tuple[set[str], set[str]]:
    """Partition scene ids into (train, test) by mesh novelty.

    A scene goes to test if it contains ANY held-out mesh. That is
    deliberately strict: a scene with one novel mesh alongside familiar ones
    still cannot be trained on without leaking the novel mesh, and leaking
    it would make the arm measure memorisation.
    """
    test = scenes_containing(scenes, held)
    train = set(scenes) - test
    return train, test
