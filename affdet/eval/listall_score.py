"""Scoring the E7 list-all arm against the human judgements.

The other E7 arms are handed the candidate list, so every human judgement is
guaranteed an answer and the join is total. List-all is not: the VLM reports
what it chooses, so the join is partial and three quantities carry the finding.

  **omission**       human judgements the VLM never mentioned. The headline.
                     An omitted pair is not a wrong answer, it is an absent
                     one, and it is what makes recall unmeasurable.
  **hallucination**  pairs it volunteered that the structural gate calls
                     impossible.
  **covered**        accuracy / F1 / MAE on the intersection only.

Why covered metrics are never reported alone
--------------------------------------------
The covered subset is chosen by the VLM. If it answers the easy pairs and
stays silent on the hard ones, its covered score goes UP as its omission rate
goes up — the metric rewards exactly the behaviour under investigation. So
every covered number is published beside two explicit imputations of the
missing rows:

  `score_covered`                omissions excluded  (optimistic bound)
  `score_omissions_as_negative`  omissions scored 0  (pessimistic bound)

Neither is the truth; the truth is bracketed by them, and the width of the
bracket IS the cost of omission. Silently choosing one would beg the question
the arm exists to ask.

Convention note: "hallucination" is measured against the **structural gate**
(`is_candidate_pair` on gold classes), not against the human label set. A
structurally valid pair the labeller simply never judged is reported separately
as `n_plausible_unlabelled` and not charged — matching `e7_score.score_full`,
which skips those same pairs for the student, so the two sides stay comparable.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from affdet.intersection_matrix import is_candidate_pair

#: manual_tool._HUMAN_SCORE_THRESHOLD — a human score >= 5 is a positive.
HUMAN_POS = 5
#: The VLM emits 0-10; arm G+ binarises at 4.5 (see E7_END_TO_END_SETUP.md §5).
VLM_POS = 4.5

#: Key type for every map here: (scene_id, actor_obj_id, recipient_obj_id,
#: predicate). Scene-scoped, because the same instance pair can recur across
#: scenes at different sizes and poses.
Key = tuple[str, str, str, str]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2.0


# --------------------------------------------------------------------------
# Omission
# --------------------------------------------------------------------------

def coverage_report(vlm: dict[Key, float], human: dict[Key, int]) -> dict:
    """Which human judgements did the VLM raise at all?

    Coverage is about being MENTIONED, never about being right: a pair the VLM
    reported and scored 0 is covered (and will be charged as a disagreement by
    the metric functions), whereas a pair it never named is omitted. Collapsing
    the two would erase the distinction this arm exists to measure.

    `omission_rate_positive` is the number to lead with. Omitting a
    human-negative is close to agreement — the VLM declined to claim an
    affordance that is not there. Omitting a human-positive is the damaging
    case: a real affordance the system would simply never propose.
    """
    covered = [k for k in human if k in vlm]
    omitted = [k for k in human if k not in vlm]
    omitted_pos = [k for k in omitted if human[k] >= HUMAN_POS]
    omitted_neg = [k for k in omitted if human[k] < HUMAN_POS]
    n_pos = sum(1 for k in human if human[k] >= HUMAN_POS)
    n_neg = len(human) - n_pos

    per: dict[str, dict] = {}
    by_pred: dict[str, list[Key]] = defaultdict(list)
    for k in human:
        by_pred[k[3]].append(k)
    for predicate, keys in by_pred.items():
        om = [k for k in keys if k not in vlm]
        om_pos = [k for k in om if human[k] >= HUMAN_POS]
        kp = sum(1 for k in keys if human[k] >= HUMAN_POS)
        per[predicate] = {
            "n_human": len(keys),
            "n_covered": len(keys) - len(om),
            "n_omitted": len(om),
            "omission_rate": len(om) / len(keys) if keys else 0.0,
            "n_omitted_positive": len(om_pos),
            "omission_rate_positive": len(om_pos) / kp if kp else 0.0,
        }

    return {
        "n_human": len(human),
        "n_covered": len(covered),
        "n_omitted": len(omitted),
        "omission_rate": len(omitted) / len(human) if human else 0.0,
        "n_omitted_positive": len(omitted_pos),
        "n_omitted_negative": len(omitted_neg),
        "omission_rate_positive": len(omitted_pos) / n_pos if n_pos else 0.0,
        "omission_rate_negative": len(omitted_neg) / n_neg if n_neg else 0.0,
        "per_predicate": per,
    }


# --------------------------------------------------------------------------
# Hallucination
# --------------------------------------------------------------------------

def hallucination_report(vlm: dict[Key, float], human: dict[Key, int],
                         cls_of: dict[str, str]) -> dict:
    """What did the VLM volunteer that it should not have?

    Buckets every reported triple:

      labelled            in the human set — the scorable intersection
      plausible_unlabelled  passes the structural gate, never human-judged.
                          NOT charged: the labeller's coverage is not the
                          VLM's fault, and `e7_score.score_full` skips these
                          for the student too.
      impossible          fails the gate on gold classes. This is the
                          hallucination count.
      unresolvable        an obj_id not in the scene's class map (a box number
                          that resolved to nothing, or a malformed row).
    """
    n_lab = n_plaus = n_imp = n_unres = 0
    impossible: list[Key] = []
    for k in vlm:
        if k in human:
            n_lab += 1
            continue
        _, a, r, predicate = k
        ac, rc = cls_of.get(a), cls_of.get(r)
        if ac is None or rc is None:
            n_unres += 1
            continue
        if is_candidate_pair(ac, rc, predicate):
            n_plaus += 1
        else:
            n_imp += 1
            impossible.append(k)

    n = len(vlm)
    return {
        "n_reported": n,
        "n_labelled": n_lab,
        "n_plausible_unlabelled": n_plaus,
        "n_impossible": n_imp,
        "n_unresolvable": n_unres,
        "hallucination_rate": n_imp / n if n else 0.0,
        "impossible_examples": sorted(impossible)[:20],
    }


# --------------------------------------------------------------------------
# Covered subset, and the two imputations
# --------------------------------------------------------------------------

def _metrics(rows: list[tuple[float, int]]) -> dict:
    """rows = [(vlm_score_0_10, human_score_0_9)]."""
    tp = fp = fn = tn = 0
    errs = []
    for v, h in rows:
        vp, hp = v >= VLM_POS, h >= HUMAN_POS
        if vp and hp:
            tp += 1
        elif vp:
            fp += 1
        elif hp:
            fn += 1
        else:
            tn += 1
        errs.append(abs(v - h))
    p, r, f = _prf(tp, fp, fn)
    n = len(rows)
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": _balanced_accuracy(tp, fp, fn, tn),
        "precision": p, "recall": r, "f1": f,
        "mae": statistics.mean(errs) if errs else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _by_predicate(pairs: list[tuple[Key, float, int]]) -> dict:
    by: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for k, v, h in pairs:
        by[k[3]].append((v, h))
    return {p: _metrics(rows) for p, rows in by.items()}


def score_covered(vlm: dict[Key, float], human: dict[Key, int],
                  per_predicate: bool = False) -> dict:
    """Optimistic bound: omitted rows are excluded from the denominator.

    This is the number most flattering to the arm, and the one a naive report
    would quote alone. It answers "when the VLM chose to answer, was it right?"
    — never "does the VLM answer the question asked of it?".
    """
    pairs = [(k, vlm[k], h) for k, h in human.items() if k in vlm]
    out = _metrics([(v, h) for _, v, h in pairs])
    out["imputation"] = "excluded"
    if per_predicate:
        out["per_predicate"] = _by_predicate(pairs)
    return out


def score_omissions_as_negative(vlm: dict[Key, float], human: dict[Key, int],
                                per_predicate: bool = False) -> dict:
    """Pessimistic bound: an unmentioned pair is treated as a scored 0.

    Defensible as the deployment reading — a TAMP planner never told about an
    affordance cannot use it, so silence and "no" have identical consequences
    downstream. It is a bound and not the truth because the VLM was never asked
    about these pairs, and might have said yes if it had been.
    """
    pairs = [(k, vlm.get(k, 0.0), h) for k, h in human.items()]
    out = _metrics([(v, h) for _, v, h in pairs])
    out["imputation"] = "omissions_as_negative"
    if per_predicate:
        out["per_predicate"] = _by_predicate(pairs)
    return out


# --------------------------------------------------------------------------
# Multi-pass aggregation
# --------------------------------------------------------------------------

def aggregate_passes(passes: list[dict[Key, float]], with_counts: bool = False):
    """Median score per triple across passes, matching the existing arms.

    A pair reported in only one pass of three is KEPT. Requiring a majority
    would turn aggregation into a second, hidden omission filter and would
    understate what the VLM volunteered; the mention count is returned instead
    (`with_counts=True`) so instability is reported as its own quantity rather
    than silently folded into the omission rate.
    """
    scores: dict[Key, list[float]] = defaultdict(list)
    for p in passes:
        for k, v in p.items():
            scores[k].append(v)
    agg = {k: statistics.median(v) for k, v in scores.items()}
    if with_counts:
        return agg, {k: len(v) for k, v in scores.items()}
    return agg
