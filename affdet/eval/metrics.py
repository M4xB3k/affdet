"""Per-predicate + macro + micro P/R/F1 over matched (actor, recipient, predicate) keys.

canPourInto is excluded from score_fidelity: its label "score" is a binary
provenance flag (open/closed), not a VLM 0-9 fidelity score (Spec Decision 3
says canPourInto is binary-only), so MAE/corr against it is meaningless.
"""
from __future__ import annotations
from pathlib import Path
import csv

from affdet.schemas import Labels, Predictions
from affdet.intersection_matrix import TIER1_PREDICATES

_SCORE_FIDELITY_PREDICATES = [p for p in TIER1_PREDICATES if p != "canPourInto"]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _iter_matched(predictions_dir: Path, labels_dir: Path):
    """Yield (LabelEntry, PredictionEntry) pairs matched by (actor, recipient,
    predicate) across every scene, skipping scenes with no matching predictions
    file and labels with no matching prediction entry."""
    for lab_path in sorted(labels_dir.glob("*.json")):
        prd_path = predictions_dir / lab_path.name
        if not prd_path.exists():
            continue
        labels = Labels.model_validate_json(lab_path.read_text())
        preds = Predictions.model_validate_json(prd_path.read_text())
        prd_by_key = {(e.actor, e.recipient, e.predicate): e for e in preds.tier1}
        for lab in labels.tier1:
            prd = prd_by_key.get((lab.actor, lab.recipient, lab.predicate))
            if prd is not None:
                yield lab, prd


def calibrate_thresholds(
    predictions_dir: Path | str, labels_dir: Path | str,
    candidates: list[float] | None = None,
) -> dict[str, float]:
    """Find the decision threshold per predicate that maximizes F1 against
    ground-truth verdicts, sweeping PredictionEntry.score (NOT LabelEntry.score,
    which is a soft VLM distillation target, not the binary ground truth).

    Intended to run on a VALIDATION split's predictions, then be applied to a
    held-out test split via compute_metrics(..., thresholds=...) -- calibrating
    and evaluating on the same split would leak information and overstate F1.
    Predicates with no matched data default to 0.5 (the original uncalibrated
    cutoff), never left unset.
    """
    if candidates is None:
        candidates = [i / 100 for i in range(5, 100, 5)]

    scored: dict[str, list[tuple[int, float]]] = {p: [] for p in TIER1_PREDICATES}
    for lab, prd in _iter_matched(Path(predictions_dir), Path(labels_dir)):
        scored[lab.predicate].append((lab.verdict, prd.score))

    best: dict[str, float] = {}
    for pred, pairs in scored.items():
        if not pairs:
            best[pred] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            tp = sum(1 for v, s in pairs if v == 1 and s >= t)
            fp = sum(1 for v, s in pairs if v == 0 and s >= t)
            fn = sum(1 for v, s in pairs if v == 1 and s < t)
            f1 = _prf(tp, fp, fn)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[pred] = best_t
    return best


def compute_metrics(
    predictions_dir: Path | str, labels_dir: Path | str,
    thresholds: dict[str, float] | None = None,
    macro_exclude: list[str] | None = None,
    auto_exclude_zero_support: bool = False,
) -> dict:
    """macro_exclude names predicates that are still reported individually but
    left OUT of the macro average. Needed because canPierce has zero negatives
    corpus-wide, so it cannot be predicted incorrectly and its F1 of 1.0 is an
    artifact of the label set rather than a measure of performance -- averaging
    it in silently inflates the headline number.

    auto_exclude_zero_support additionally excludes any predicate with NO test
    pairs at all (tp=fp=fn=tn=0) from the macro. Opt-in and off by default:
    the plain zero-pairs case (e.g. a small/mock fixture that only has data for
    one predicate) is existing, deliberate behaviour elsewhere (see
    test_metrics_perfect_predictions) where an absent predicate scores 0.0 and
    IS averaged in. This flag exists for holdout-mode runs (E1c/E1c'), where a
    low-instance predicate (canPeel: only 4 actor meshes corpus-wide) can be
    stranded with zero test pairs by an unlucky mesh-holdout draw -- that is
    "no evidence", not a wrong prediction, and must not silently drag the
    macro toward zero the way a real failure would."""
    predictions_dir = Path(predictions_dir)
    labels_dir = Path(labels_dir)

    counts: dict[str, dict[str, int]] = {
        p: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for p in TIER1_PREDICATES
    }
    paired: dict[str, list[tuple[float, float]]] = {p: [] for p in _SCORE_FIDELITY_PREDICATES}

    for lab, prd in _iter_matched(predictions_dir, labels_dir):
        if lab.score is not None and prd.score is not None and lab.predicate in paired:
            paired[lab.predicate].append((lab.score, prd.score))
        # A predicate absent from a hand-supplied threshold dict falls back to
        # the original 0.5 cutoff rather than raising.
        pred_verdict = (
            prd.verdict if thresholds is None
            else (1 if prd.score >= thresholds.get(lab.predicate, 0.5) else 0)
        )
        c = counts[lab.predicate]
        if lab.verdict == 1 and pred_verdict == 1:
            c["tp"] += 1
        elif lab.verdict == 0 and pred_verdict == 1:
            c["fp"] += 1
        elif lab.verdict == 1 and pred_verdict == 0:
            c["fn"] += 1
        else:
            c["tn"] += 1

    per_pred = {p: _prf(c["tp"], c["fp"], c["fn"]) for p, c in counts.items()}
    excluded = list(macro_exclude or [])
    if auto_exclude_zero_support:
        excluded += [p for p, c in counts.items()
                    if c["tp"] + c["fp"] + c["fn"] + c["tn"] == 0
                    and p not in excluded]
    macro_preds = [p for p in per_pred if p not in excluded]
    macro_f1 = (
        sum(per_pred[p]["f1"] for p in macro_preds) / len(macro_preds)
        if macro_preds else 0.0
    )
    tot_tp = sum(c["tp"] for c in counts.values())
    tot_fp = sum(c["fp"] for c in counts.values())
    tot_fn = sum(c["fn"] for c in counts.values())
    micro = _prf(tot_tp, tot_fp, tot_fn)

    def _fidelity(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"mae": None, "corr": None, "n": 0}
        import numpy as np
        y = np.array([a for a, _ in pairs]); yp = np.array([b for _, b in pairs])
        mae = float(np.abs(y - yp).mean())
        corr = float(np.corrcoef(y, yp)[0, 1]) if len(pairs) > 1 and y.std() > 0 and yp.std() > 0 else None
        return {"mae": mae, "corr": corr, "n": len(pairs)}
    score_fidelity = {p: _fidelity(pairs) for p, pairs in paired.items()}

    return {
        "per_predicate": per_pred,
        "macro_f1": macro_f1,
        "macro_excluded": excluded,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "counts": counts,
        "score_fidelity": score_fidelity,
        "thresholds": dict(thresholds) if thresholds is not None else None,
    }


def write_metrics_csv(predictions_dir: Path | str, labels_dir: Path | str,
                      results_dir: Path | str,
                      thresholds: dict[str, float] | None = None,
                      macro_exclude: list[str] | None = None,
                      auto_exclude_zero_support: bool = False) -> Path:
    """Write metrics.csv. When `thresholds` is given, verdicts are recomputed
    at those per-predicate cutoffs and the thresholds are persisted alongside
    the CSV as thresholds.json -- a reported result is not reproducible without
    the decision cutoff that produced it."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(predictions_dir, labels_dir,
                              thresholds=thresholds, macro_exclude=macro_exclude,
                              auto_exclude_zero_support=auto_exclude_zero_support)
    if thresholds is not None:
        import json
        (results_dir / "thresholds.json").write_text(
            json.dumps(thresholds, indent=2, sort_keys=True))
    csv_path = results_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predicate", "precision", "recall", "f1", "tp", "fp", "fn", "tn"])
        for pred in TIER1_PREDICATES:
            r = metrics["per_predicate"][pred]
            c = metrics["counts"][pred]
            w.writerow([pred, r["precision"], r["recall"], r["f1"],
                        c["tp"], c["fp"], c["fn"], c["tn"]])
        w.writerow(["MACRO", "", "", metrics["macro_f1"], "", "", "", ""])
        w.writerow(["MICRO", metrics["micro_precision"], metrics["micro_recall"],
                    metrics["micro_f1"], "", "", "", ""])
        w.writerow([])
        w.writerow(["predicate", "mae", "corr", "n_scored"])
        for pred in _SCORE_FIDELITY_PREDICATES:
            fid = metrics["score_fidelity"][pred]
            w.writerow([pred, fid["mae"], fid["corr"], fid["n"]])
    return csv_path
