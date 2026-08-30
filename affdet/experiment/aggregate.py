"""Seed-family aggregation over the run ledger, and LaTeX results-table export.

A thesis reports mean±std across seeds, not one arbitrary run. Nothing in the
harness previously grouped seeds -- `-s0/-s1/-s2` was a filename convention no
code understood, so multi-seed claims were assembled by hand. Likewise every
results table was hand-transcribed, which is how this project once shipped a
table built from a stale label corpus (the v3/v3 naming collision).

Both are fixed here: `aggregate_seed_families` groups runs that differ ONLY in
seed, and `write_results_table` renders the aggregate to a .tex file the thesis
\\input's, so the numbers in the PDF can never drift from the numbers on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from affdet.intersection_matrix import TIER1_PREDICATES

#: Config fields that define an experimental ARM. Two runs agreeing on all of
#: these but differing in `seed` are repetitions of one experiment. `seed` is
#: deliberately absent; `name` is too, since it usually encodes the seed.
_ARM_FIELDS = ("scenes", "labels", "bbox_source", "crop_padding", "backbone",
               "head", "epochs", "lr", "unbounded", "unbounded_neg_ratio",
               "resolved_instances", "split_strategy",
               "holdout_mode", "holdout_frac", "all_pairs")


def seed_family_key(record: dict) -> tuple:
    """Identity of the experimental arm a run belongs to.

    Includes `labels_hash` so that two runs on different label corpora are
    never averaged together even when every config field matches -- exactly
    the failure the v3/v3 collision produced.
    """
    cfg = record.get("config", {})
    arm = tuple((f, cfg.get(f)) for f in _ARM_FIELDS)
    return arm + (("labels_hash", record.get("labels_hash")),)


def _mean_std(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1}
    # Population std: these are all the runs of this arm, not a sample of them.
    var = sum((v - mean) ** 2 for v in values) / n
    return {"mean": mean, "std": var ** 0.5, "n": n}


def _aggregate_block(records: list[dict], block: str) -> dict:
    """Aggregate one metrics block ('raw' or 'calibrated') across seeds."""
    blocks = [r["metrics"][block] for r in records if block in r.get("metrics", {})]
    if not blocks:
        return {}

    out: dict = {"macro_f1": _mean_std([b["macro_f1"] for b in blocks
                                       if b.get("macro_f1") is not None])}
    for key in ("micro_f1", "micro_precision", "micro_recall"):
        vals = [b[key] for b in blocks if b.get(key) is not None]
        if vals:
            out[key] = _mean_std(vals)

    per_pred: dict = {}
    for pred in TIER1_PREDICATES:
        metrics: dict = {}
        for metric in ("precision", "recall", "f1"):
            vals = [b["per_predicate"][pred][metric] for b in blocks
                    if pred in b.get("per_predicate", {})]
            if vals:
                metrics[metric] = _mean_std(vals)
        if metrics:
            per_pred[pred] = metrics
    out["per_predicate"] = per_pred

    excl = next((b["macro_excluded"] for b in blocks if b.get("macro_excluded")), [])
    out["macro_excluded"] = list(excl)

    # E1d: score fidelity vs the VLM teacher (MAE/corr), already computed per
    # run and previously dropped at this aggregation boundary. `corr` is None
    # (not zero) when undefined, e.g. canPierce has no negatives, so it must
    # be skipped rather than pulled into the mean like a real correlation.
    fid_preds = {p for b in blocks for p in b.get("score_fidelity", {})}
    fidelity: dict = {}
    for pred in fid_preds:
        entries = [b["score_fidelity"][pred] for b in blocks
                  if pred in b.get("score_fidelity", {})]
        mae_vals = [e["mae"] for e in entries if e.get("mae") is not None]
        corr_vals = [e["corr"] for e in entries if e.get("corr") is not None]
        fidelity[pred] = {
            "mae": _mean_std(mae_vals) if mae_vals else {"mean": None, "std": None, "n": 0},
            "corr": _mean_std(corr_vals) if corr_vals else {"mean": None, "std": None, "n": 0},
        }
    if fidelity:
        out["score_fidelity"] = fidelity

    thr = [b["thresholds"] for b in blocks if b.get("thresholds")]
    if thr:
        out["thresholds"] = {
            p: _mean_std([t[p] for t in thr if p in t])
            for p in {k for t in thr for k in t}
        }
    return out


def aggregate_seed_families(records: list[dict]) -> list[dict]:
    """Group ledger records into seed families and reduce each to mean±std."""
    families: dict[tuple, list[dict]] = {}
    for r in records:
        families.setdefault(seed_family_key(r), []).append(r)

    out = []
    for key, runs in families.items():
        runs = sorted(runs, key=lambda r: r.get("config", {}).get("seed", 0))
        cfg = runs[0].get("config", {})
        out.append({
            "arm": dict(key[:-1]),
            "name": cfg.get("name"),
            "labels_hash": runs[0].get("labels_hash"),
            "n_seeds": len(runs),
            "seeds": [r.get("config", {}).get("seed") for r in runs],
            "run_ids": [r.get("run_id") for r in runs],
            "git_shas": sorted({r.get("git_sha") for r in runs}),
            "raw": _aggregate_block(runs, "raw"),
            "calibrated": _aggregate_block(runs, "calibrated"),
        })
    return out


def load_and_aggregate(ledger_path: Path | str) -> list[dict]:
    from affdet.experiment.ledger import load_runs
    return aggregate_seed_families(load_runs(ledger_path))


def _fmt(cell: dict | None, places: int = 3) -> str:
    """mean±std, or an em-dash when the metric is absent."""
    if not cell or cell.get("mean") is None:
        return "--"
    mean = f"{cell['mean']:.{places}f}"
    if cell.get("n", 0) > 1 and cell.get("std"):
        return f"{mean} $\\pm$ {cell['std']:.{places}f}"
    return mean


def write_results_table(families: list[dict], out_path: Path | str,
                        block: str = "calibrated",
                        caption: str | None = None,
                        label: str | None = None) -> Path:
    """Render aggregated families as a per-predicate LaTeX table.

    Predicates excluded from the macro average (canPierce, which has no
    negatives in the corpus) are still listed but explicitly marked, so a
    reader cannot mistake an artifactual 1.000 for real performance.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Generated by affdet.experiment.aggregate -- do not edit by hand.",
        "% Regenerate from the run ledger so the thesis can never drift from",
        "% the numbers actually on disk.",
        "\\begin{table}[htbp]",
        "  \\centering",
    ]
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines += [
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Predicate & Precision & Recall & F1 \\\\",
        "    \\midrule",
    ]

    excluded: list[str] = []
    for fam in families:
        blk = fam.get(block, {})
        excluded = blk.get("macro_excluded", []) or excluded
        per_pred = blk.get("per_predicate", {})
        if len(families) > 1:
            lines.append(f"    \\multicolumn{{4}}{{l}}{{\\textit{{{fam['name']}}}}} \\\\")
        for pred in TIER1_PREDICATES:
            if pred not in per_pred:
                continue
            m = per_pred[pred]
            mark = "$^{\\dagger}$" if pred in (blk.get("macro_excluded") or []) else ""
            lines.append(
                f"    {pred}{mark} & {_fmt(m.get('precision'))} & "
                f"{_fmt(m.get('recall'))} & {_fmt(m.get('f1'))} \\\\"
            )
        lines.append("    \\midrule")
        lines.append(
            f"    \\textbf{{Macro}} & & & {_fmt(blk.get('macro_f1'))} \\\\"
        )

    lines += ["    \\bottomrule", "  \\end{tabular}"]

    # E1d: score fidelity vs the VLM teacher -- MAE/corr per predicate,
    # a second small table rather than cramming into the P/R/F1 columns.
    any_fidelity = any(fam.get(block, {}).get("score_fidelity") for fam in families)
    if any_fidelity:
        lines += [
            "  \\begin{tabular}{lcc}",
            "    \\toprule",
            "    Predicate & Score MAE & Score Corr \\\\",
            "    \\midrule",
        ]
        for fam in families:
            fid = fam.get(block, {}).get("score_fidelity", {})
            if not fid:
                continue
            if len(families) > 1:
                lines.append(f"    \\multicolumn{{3}}{{l}}{{\\textit{{{fam['name']}}}}} \\\\")
            for pred in TIER1_PREDICATES:
                if pred not in fid:
                    continue
                lines.append(
                    f"    {pred} & {_fmt(fid[pred].get('mae'), 3)} & "
                    f"{_fmt(fid[pred].get('corr'), 3)} \\\\"
                )
        lines += ["    \\bottomrule", "  \\end{tabular}"]
    if excluded:
        names = ", ".join(excluded)
        lines.append(
            f"  \\par\\smallskip\\footnotesize $^{{\\dagger}}$ {names} is "
            "\\emph{excluded} from the macro average: it has no negative "
            "examples in the corpus, so it cannot be predicted incorrectly "
            "and its score reflects the label distribution rather than model "
            "performance."
        )
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_aggregate_json(families: list[dict], out_path: Path | str) -> Path:
    """Machine-readable companion to the LaTeX table."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(families, indent=2, sort_keys=True))
    return out_path
