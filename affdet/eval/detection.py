"""Detection scoring — precision / recall / F1 at an IoU threshold (E5).

Answers "how good is stage 1?" on the synthetic corpus, which carries an
exact ground-truth box and class for every object, so no hand annotation is
involved and the measurement is reproducible from the render metadata alone.

Deliberately separate from ``affdet.detect.yolo_boxes.match_boxes``, which
solves a different problem and cannot be reused here:

* ``match_boxes`` DROPS predictions that match no gold object, because the
  crop-source ablation has no gold ``obj_id`` to hang them on. Those dropped
  predictions are exactly the false positives precision is made of.
* ``match_boxes`` is class-agnostic by design (it keeps the gold class so
  affordance labels stay aligned). Detection quality is normally reported
  class-aware: a well-placed box carrying the wrong name has not found the
  object.
* ``match_boxes`` thresholds with strict ``>``; the thesis reports
  "IoU >= 0.5", and the boundary should match the claim.

Both scorings are available here — ``class_agnostic=True`` measures box
quality alone, which is the number that actually governs this pipeline,
since crops are re-classified downstream by the DINOv2 probe and the
detector's own labels are discarded.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from affdet.detect.yolo_boxes import iou
from affdet.schemas import SceneObject


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision/recall/F1 from counts.

    A zero denominator means "no evidence of success", scored 0.0 rather
    than left undefined: a detector that predicts nothing has recall 0, and
    one whose every prediction is wrong has precision 0.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def _rollup(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """Attach P/R/F1 to each class's raw TP/FP/FN counts."""
    return {
        cls: {**c, **_prf(c["tp"], c["fp"], c["fn"])}
        for cls, c in sorted(counts.items())
    }


def score_detections(
    gold: Sequence[SceneObject],
    preds: Sequence[dict[str, Any]],
    iou_thresh: float = 0.5,
    class_agnostic: bool = False,
    subgroups: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Score one scene's predictions against its ground-truth objects.

    Greedy highest-IoU-first matching: every gold object and every
    prediction is consumed at most once, so duplicate boxes over the same
    object cost precision instead of earning a second true positive.

    Unless ``class_agnostic``, a match also requires the predicted class to
    equal the gold class. A right-box/wrong-class prediction then scores as
    both a false negative (the object was not found) and a false positive
    (something that is not there was reported) — the standard convention,
    and the honest one: exactly one detection was wrong, and it damaged both
    the missed class's recall and the claimed class's precision.

    ``subgroups`` maps a name to a set of gold ``obj_id``s, and additionally
    scores those objects on their own (everything else lands in ``rest``).
    The instance-disjoint arm needs this: it reserves MESHES but pulls whole
    SCENES into test, and a scene holds 3-4 objects of which most are
    ordinary training meshes — on the real split only 28.6% of scored objects
    are novel. Without the split, that arm's headline F1 is a blend
    dominated by seen meshes and does not measure what its name claims.

    A subgroup's false positives are the predictions that matched one of its
    objects but were named wrong. A prediction matching nothing cannot be
    attributed to a subgroup — there is no gold object to hang it on — so it
    is counted only in the top-level totals, and subgroup precision is
    therefore an upper bound where hallucinations are concerned. Recall is
    exact.

    Returns TP/FP/FN plus micro P/R/F1, a per-class breakdown, and the
    matched pairs (with their IoU) for inspection.
    """
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(preds):
            if not class_agnostic and p.get("cls") != g.obj_class:
                continue
            score = iou(list(g.bbox), list(p["bbox"]))
            if score >= iou_thresh:
                candidates.append((score, gi, pi))
    # Best overlap first, so a gold object is paired with its tightest box
    # rather than whichever merely-adequate one happened to come first.
    # Ties break on index, keeping the result deterministic.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    matches: list[dict[str, Any]] = []
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for score, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matches.append({
            "obj_id": gold[gi].obj_id,
            "true_cls": gold[gi].obj_class,
            "pred_cls": preds[pi].get("cls"),
            "iou": score,
            "gold_bbox": list(gold[gi].bbox),
            "pred_bbox": list(preds[pi]["bbox"]),
        })

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0})
    for m in matches:
        counts[m["true_cls"]]["tp"] += 1
    for gi, g in enumerate(gold):
        if gi not in used_gold:
            counts[g.obj_class]["fn"] += 1
    for pi, p in enumerate(preds):
        if pi not in used_pred:
            # Charged to the class the detector CLAIMED — that is the class
            # whose precision this prediction actually damaged.
            counts[str(p.get("cls"))]["fp"] += 1

    tp = len(matches)
    fn = len(gold) - tp
    fp = len(preds) - tp
    out = {
        "tp": tp, "fp": fp, "fn": fn,
        **_prf(tp, fp, fn),
        "n_gold": len(gold),
        "n_pred": len(preds),
        "per_class": _rollup(counts),
        "matches": matches,
    }

    if subgroups is not None:
        matched_ids = {m["obj_id"] for m in matches}
        groups: dict[str, dict[str, int]] = {}
        assigned: set[str] = set()
        for name, members in subgroups.items():
            ids = {g.obj_id for g in gold if g.obj_id in members}
            assigned |= ids
            groups[name] = {
                "tp": len(ids & matched_ids),
                "fn": len(ids - matched_ids),
                "n_gold": len(ids),
            }
        rest = {g.obj_id for g in gold} - assigned
        groups["rest"] = {
            "tp": len(rest & matched_ids),
            "fn": len(rest - matched_ids),
            "n_gold": len(rest),
        }
        # Unmatched predictions have no gold object to attribute them to, so
        # they stay in the top-level fp only; see the docstring.
        out["subgroups"] = {
            name: {**c, "fp": 0, **_prf(c["tp"], 0, c["fn"])}
            for name, c in groups.items()
        }

    return out


def aggregate_detection_scores(
    per_scene: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Pool per-scene results into one corpus-level score.

    Counts are summed and P/R/F1 recomputed (a *micro* average), rather than
    averaging per-scene F1s — otherwise a scene holding one object would
    weigh as heavily as a scene holding ten.

    ``macro_f1`` is also reported: the unweighted mean of per-class F1, in
    which a rare class counts as much as a common one. The two answer
    different questions and can diverge sharply on this corpus, where class
    frequencies are very uneven, so both are reported rather than one being
    passed off as "the" number.
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0})
    sub: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fn": 0, "n_gold": 0})
    tp = fp = fn = n_gold = n_pred = n_scenes = 0

    for r in per_scene:
        n_scenes += 1
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        n_gold += r.get("n_gold", 0)
        n_pred += r.get("n_pred", 0)
        for cls, c in r["per_class"].items():
            counts[cls]["tp"] += c["tp"]
            counts[cls]["fp"] += c["fp"]
            counts[cls]["fn"] += c["fn"]
        for name, c in r.get("subgroups", {}).items():
            sub[name]["tp"] += c["tp"]
            sub[name]["fn"] += c["fn"]
            sub[name]["n_gold"] += c.get("n_gold", 0)

    per_class = _rollup(counts)
    macro_f1 = (sum(c["f1"] for c in per_class.values()) / len(per_class)
                if per_class else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        **_prf(tp, fp, fn),
        "macro_f1": macro_f1,
        "n_scenes": n_scenes,
        "n_gold": n_gold,
        "n_pred": n_pred,
        "per_class": per_class,
        **({"subgroups": {
            name: {**c, "fp": 0, **_prf(c["tp"], 0, c["fn"])}
            for name, c in sub.items()}} if sub else {}),
    }
