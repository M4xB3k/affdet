"""Classification scoring — DINOv2 probe vs. a detector's own class head (E5).

The deployed pipeline deliberately discards the detector's class label and
re-classifies each crop with a DINOv2 linear probe. This module produces the
measurement that justifies that design decision, by scoring both methods on
the *same* objects: whatever crop is fed to one is fed to the other, so the
only variable is the classifier.

Also holds the prompt-vocabulary mapping used by the zero-shot detector arms.
The taxonomy's class IDs are snake_case (``red_wine_glass``). A fine-tuned
detector never reads that string — a class is a learned index — but a
zero-shot detector consumes it as English text, where the underscore form is
not a phrase anyone would write. Feeding the raw ID would measure a text
model's tolerance for underscores rather than its ability to detect the
object, so prompts are de-underscored while scoring stays on the canonical
ID, keeping every arm in one label space.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence


def prompt_vocabulary(classes: Sequence[str]) -> list[str]:
    """Taxonomy class IDs -> natural-language prompts, order preserved.

    Order is the index mapping for a detector's output, so it must survive.
    """
    return [c.replace("_", " ") for c in classes]


def canonical_from_prompt(prompt: str, classes: Sequence[str]) -> str | None:
    """Map a detector's emitted text back to a canonical class ID.

    Returns ``None`` for anything outside the taxonomy. An open-vocabulary
    detector can emit text no class corresponds to; that is a wrong answer
    and is scored as one, never coerced into the nearest valid class.
    """
    if prompt is None:
        return None
    normalised = prompt.strip().lower()
    for c in classes:
        if normalised in (c.lower(), c.replace("_", " ").lower()):
            return c
    return None


def _score_one(true: Sequence[str],
               pred: Sequence[str | None]) -> dict[str, Any]:
    correct = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusions: Counter = Counter()

    for t, p in zip(true, pred):
        per_class[t][1] += 1
        if p == t:
            correct += 1
            per_class[t][0] += 1
        else:
            # A None prediction (out-of-vocabulary) is a wrong answer, not a
            # row to skip -- dropping those would inflate accuracy.
            confusions[(t, p)] += 1

    n = len(true)
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "n": n,
        "per_class": {
            cls: {"correct": c[0], "total": c[1],
                  "acc": c[0] / c[1] if c[1] else 0.0}
            for cls, c in sorted(per_class.items())
        },
        "top_confusions": [
            {"true": t, "pred": p, "count": n_}
            for (t, p), n_ in confusions.most_common(20)
        ],
    }


def score_classification(
    true: Sequence[str],
    probe: Sequence[str | None],
    classhead: Sequence[str | None],
) -> dict[str, Any]:
    """Compare two classifiers over one shared set of objects.

    ``true[i]``, ``probe[i]`` and ``classhead[i]`` must describe the same
    object; a length mismatch means they do not, and is refused rather than
    silently zipped into a comparison of different objects.
    """
    if not (len(true) == len(probe) == len(classhead)):
        raise ValueError(
            "probe and class head must be scored over the same objects: got "
            f"{len(true)} true, {len(probe)} probe, {len(classhead)} classhead"
        )

    probe_r = _score_one(true, probe)
    head_r = _score_one(true, classhead)
    return {
        "n": len(true),
        "probe": probe_r,
        "classhead": head_r,
        "delta_accuracy": probe_r["accuracy"] - head_r["accuracy"],
    }
