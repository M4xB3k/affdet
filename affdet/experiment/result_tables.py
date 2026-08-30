"""Export E1 ladder + E7 arm rows -> the thesis tables.

Companion to ``component_tables`` (E5), which this deliberately mirrors:
every number reaches the thesis via ``\\input{tables/...}`` generated from
the ledger, never transcribed. The v3/v3 naming collision already produced
one wrong results table in this project, and the E1 ladder once produced a
wrong 6-seed family by blending a superseded rerun with the runs it
replaced. Both failures are guarded here rather than left to care.

Three tables:

  per_predicate.tex     E1 per-predicate P/R/F1, raw and calibrated, seed
                        averaged, with both macro rows
  e7_arms.tex           the G+/G/E ladder, VLM against student
  e7_significance.tex   the paired bootstrap, with and without canPierce

Unlike ``e1_ladder_v4.tex`` these emit full floats with a caption and a
label: a bare ``tabular`` cannot be cross-referenced, which is why nothing
ever \\input it.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

#: Taxonomy order, not alphabetical: containment, support, then the
#: tool-use predicates. The table reads as the taxonomy does in Ch. 4.
PREDICATE_ORDER = ("canContain", "canSupport", "canCut", "canPourInto",
                   "canPeel", "canPierce")

#: canPourInto is supervised by a hand-authored per-instance rule rather
#: than the teacher, so a macro including it mixes "reproduces a teacher"
#: with "recovers a hand-authored rule". Both macros are reported.
RULE_SUPERVISED = ("canPourInto",)

_ARM_LABEL = {"gplus": "G+", "g": "G", "e": "E"}
_ARM_ORDER = ("gplus", "g", "e")

_METRIC_LABEL = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1",
    "mae": "MAE",
    "spearman": "Spearman $\\rho$",
}
_METRIC_ORDER = ("accuracy", "balanced_accuracy", "f1", "mae", "spearman")

_HEADER = (
    "% GENERATED FILE — DO NOT EDIT BY HAND.",
    "% Produced by `affdet.experiment.result_tables` from the committed",
    "% ledger snapshot. Hand edits will be overwritten. If a number here",
    "% looks wrong, fix the ledger or the exporter, never this file.",
)


def _escape(s: str) -> str:
    """Underscores are math-mode subscripts in LaTeX and break the build."""
    return str(s).replace("_", r"\_")


def _fmt(v: Any, places: int = 3) -> str:
    return "--" if v is None else f"{v:.{places}f}"


def _rung_of(name: str) -> str:
    """``e1c-strict-s0-v4fix`` -> ``e1c-strict``."""
    parts = str(name).split("-")
    return "-".join(parts[:2])


def ladder_rows(snapshot_path: Path | str,
                corpus: str | None = None) -> dict[str, list[dict]]:
    """E1 runs from a committed ledger snapshot, grouped by rung.

    ``corpus`` keeps only runs whose name carries that version tag ("v4",
    "v5"). The ledger accumulates every corpus the project has trained on,
    and a mean over a v4 seed and a v5 seed describes a model that was never
    trained. Pass it whenever a table must speak for one corpus.

    Applies the supersede rule: a rung holding any ``-v4fix`` run drops its
    ``-v4`` runs entirely rather than averaging the two families. The fixed
    holdout changed which meshes were held out, so the older runs measure a
    different split — blending them produces a mean of two different
    experiments, which is how this project once shipped a wrong 6-seed
    family.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in Path(snapshot_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") not in (None, "affordance"):
            continue
        name = str(r["config"]["name"])
        if corpus is not None and f"-{corpus}" not in name:
            continue
        grouped[_rung_of(name)].append(r)

    out: dict[str, list[dict]] = {}
    for rung, runs in grouped.items():
        has_fix = any("-v4fix" in r["config"]["name"] for r in runs)
        out[rung] = [r for r in runs
                     if not has_fix or "-v4fix" in r["config"]["name"]]
    return out


def _support(block: dict, pred: str) -> int:
    counts = block.get("counts", {}).get(pred, {})
    return sum(counts.get(k, 0) for k in ("tp", "fp", "fn", "tn"))


def _mean_std(vals: list[float]) -> tuple[float, float]:
    return (statistics.mean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _per_predicate_stats(runs: list[dict], block: str) -> dict[str, dict]:
    """Seed-mean P/R/F1 per predicate, skipping seeds with no support.

    A predicate with no test pairs in a given seed is "no evidence", not a
    zero score: averaging a 0.0 in would drag the macro down and read as a
    model failure rather than an absent measurement.
    """
    stats: dict[str, dict] = {}
    for pred in PREDICATE_ORDER:
        vals: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
        supports: list[int] = []
        for r in runs:
            b = r["metrics"][block]
            if pred not in b.get("per_predicate", {}):
                continue
            n = _support(b, pred)
            if n == 0:
                continue
            for k in vals:
                vals[k].append(b["per_predicate"][pred][k])
            supports.append(n)
        if not supports:
            continue
        stats[pred] = {k: _mean_std(v)[0] for k, v in vals.items()}
        stats[pred]["f1_std"] = _mean_std(vals["f1"])[1]
        stats[pred]["support"] = statistics.mean(supports)
        stats[pred]["n_seeds"] = len(supports)
    return stats


def _macro(stats: dict[str, dict], exclude: tuple[str, ...] = ()) -> dict:
    """Macro over the predicates that HAVE evidence, minus any excluded."""
    keep = [p for p in stats if p not in exclude]
    if not keep:
        return {}
    return {k: statistics.mean([stats[p][k] for p in keep])
            for k in ("precision", "recall", "f1")}


#: How each level is described in a table caption. The level decides what the
#: number means, so the caption must name it rather than say "test split".
_RUNG_DESCRIPTION = {
    "e1a-scene": "test split of unseen arrangements, familiar meshes and pairings",
    "e1b-pair": "test split of unseen pairings, in which the two meshes were "
                "seen in training but never together",
    "e1c-actor_only": "test split in which the actor mesh is unseen",
    "e1c-strict": "test split in which both meshes are unseen",
}


def write_per_predicate_tex(
    snapshot_path: Path | str,
    out_path: Path | str,
    rung: str = "e1a-scene",
    corpus: str | None = None,
) -> Path:
    """Render the per-predicate table for one ladder rung.

    ``corpus`` restricts to one corpus version, for the reason given on
    ``ladder_rows``: a snapshot holding both v4 and v5 rows would otherwise
    average two different experiments into one column.
    """
    runs = ladder_rows(snapshot_path, corpus=corpus).get(rung, [])
    if not runs:
        raise ValueError(
            f"no runs for rung {rung!r} in the snapshot — refusing to write a "
            "table of dashes that would read as a measurement")

    raw = _per_predicate_stats(runs, "raw")
    cal = _per_predicate_stats(runs, "calibrated")
    n_seeds = len(runs)

    lines = list(_HEADER)
    lines += [
        f"% Rung: {rung}. Calibrated thresholds fitted on validation.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{lcccccc}",
        "    \\toprule",
        "    & \\multicolumn{3}{c}{Raw ($\\tau=0.5$)} & "
        "\\multicolumn{3}{c}{Calibrated} \\\\",
        "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "    Predicate & P & R & F1 & P & R & F1 \\\\",
        "    \\midrule",
    ]

    for pred in PREDICATE_ORDER:
        r, c = raw.get(pred), cal.get(pred)
        if r is None and c is None:
            # Never scored as a zero: this is absent evidence, and a dash
            # is the only honest rendering of it.
            lines.append(f"    {_escape(pred)} & " + " & ".join(["--"] * 6)
                         + " \\\\")
            continue
        cells = [_fmt((r or {}).get(k)) for k in ("precision", "recall")]
        cells += [_fmt((r or {}).get("f1"))]
        cells += [_fmt((c or {}).get(k)) for k in ("precision", "recall")]
        cells += [f"\\textbf{{{_fmt((c or {}).get('f1'))}}}"] if c is not None else ["--"]
        lines.append(f"    {_escape(pred)} & " + " & ".join(cells) + " \\\\")

    lines.append("    \\midrule")
    for label, exclude in (("Macro (6 pred.)", ()),
                           (f"Macro (5 pred., excl.\\ {_escape('canPourInto')})",
                            RULE_SUPERVISED)):
        mr, mc = _macro(raw, exclude), _macro(cal, exclude)
        cells = [_fmt(mr.get(k)) for k in ("precision", "recall", "f1")]
        cells += [_fmt(mc.get(k)) for k in ("precision", "recall")]
        cells += [f"\\textbf{{{_fmt(mc.get('f1'))}}}"]
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")

    missing = [p for p in PREDICATE_ORDER if p not in cal and p not in raw]
    zero_note = ""
    if missing:
        zero_note = (
            " " + ", ".join(_escape(p) for p in missing)
            + (" has" if len(missing) == 1 else " have")
            + " no test support on this rung and is excluded from the macro "
              "rather than scored as a zero --- absent evidence, not a "
              "failure.")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption[Per-predicate performance]{Per-predicate performance on "
        f"the {_RUNG_DESCRIPTION.get(rung, 'test split')}, averaged over "
        f"{n_seeds} "
        f"seed{'s' if n_seeds != 1 else ''}. \\emph{{Raw}} thresholds the "
        "score at $\\tau=0.5$; \\emph{Calibrated} applies per-predicate "
        "thresholds fitted on the validation split and left unchanged on "
        "test. Two macros are given because "
        f"{_escape('canPourInto')} is supervised by a hand-authored "
        "per-instance rule rather than by the teacher, so its F1 measures "
        "recovery of an authored property and is not directly comparable to "
        "the five VLM-taught predicates; averaging it in mixes reproducing a "
        f"teacher with recovering a rule.{zero_note}}}",
        "  \\label{tab:per-predicate}",
        "\\end{table}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> None:
    """Regenerate every table this module owns. Driven by `make tables`."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot",
                    default="data/eval/e1_v4_ledger_snapshot.jsonl")
    ap.add_argument("--rung", default="e1a-scene",
                    help="ladder rung for the per-predicate table")
    ap.add_argument("--arms", default="data/eval/e7_arms.json")
    ap.add_argument("--significance",
                    default="data/eval/e7_significance_e.json")
    ap.add_argument("--significance-nopierce",
                    default="data/eval/e7_significance_e_nopierce.json")
    ap.add_argument("--out-dir", default="tools/docs/thesis/tables")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    for path in (
        write_per_predicate_tex(args.snapshot, out / "per_predicate.tex",
                                rung=args.rung),
        write_e7_arms_tex(args.arms, out / "e7_arms.tex"),
        write_e7_significance_tex(args.significance,
                                  args.significance_nopierce,
                                  out / "e7_significance.tex"),
    ):
        print(f"wrote {path}")


def write_e7_arms_tex(
    arms_json: Path | str,
    out_path: Path | str,
    aggregate: str = "median",
) -> Path:
    """Render the E7 ladder: student against a live scene-level VLM.

    Rows descend G+ -> G -> E, each restoring one pipeline stage to both
    sides, because the ordering is the argument: the VLM degrades as
    information is removed and the student does not.
    """
    doc = json.loads(Path(arms_json).read_text())
    arms = doc["arms"]
    vlm_key = f"vlm_{aggregate}"

    lines = list(_HEADER)
    lines += [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{llccccccc}",
        "    \\toprule",
        "    Condition & Judge & $n$ & Acc & B.acc & P & R & F1 & MAE \\\\",
        "    \\midrule",
    ]

    n_passes = None
    for i, key in enumerate(_ARM_ORDER):
        arm = arms.get(key)
        if arm is None:
            continue
        n_passes = arm.get("n_passes", n_passes)
        if i > 0:
            lines.append("    \\addlinespace")
        for judge, label in ((vlm_key, "VLM"), ("student", "student")):
            m = arm[judge]
            head = (f"\\textbf{{{_ARM_LABEL[key]}}}"
                    if judge == vlm_key else "")
            lines.append(
                f"    {head} & {label} & {m['n']} & "
                f"{_fmt(m['accuracy'])} & {_fmt(m['balanced_accuracy'])} & "
                f"{_fmt(m['precision'])} & {_fmt(m['recall'])} & "
                f"{_fmt(m['f1'])} & {_fmt(m['mae'], 2)} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption[Student against a live scene-level VLM]{Student and "
        "teacher-class VLM judged against the same human labels on the same "
        "scenes, over the pairs both cover. \\textbf{G+} gives the VLM boxes "
        "numbered and named and the student ground-truth boxes and "
        "ground-truth classes; "
        "\\textbf{G} gives the VLM numbered boxes only and runs the student's "
        "DINOv2 probe; \\textbf{E} gives the VLM the raw image and runs the "
        "student's full detector--probe--gate--head path. Each row down "
        "restores one pipeline stage to both sides. VLM scores aggregate "
        f"{n_passes} passes by {aggregate}; MAE is on the 0--9 scale. "
        "The evaluation corpus is \\emph{scene-disjoint but not "
        "instance-disjoint}: no arrangement, camera angle or physics outcome "
        "was seen in training, but the meshes were --- novel scenes, familiar "
        "objects. The VLM is off-the-shelf and has no exposure to any of it.}",
        "  \\label{tab:e7-arms}",
        "\\end{table}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_e7_significance_tex(
    full_json: Path | str,
    nopierce_json: Path | str,
    out_path: Path | str,
) -> Path:
    """Render the paired bootstrap, with and without canPierce.

    The two conditions sit side by side because the contrast between them
    *is* the finding: the VLM's lead is significant on four metrics over all
    five predicates, and every interval spans zero once canPierce — whose
    failure is a labelling defect, not a model one — is removed.
    """
    full = json.loads(Path(full_json).read_text())
    nop = json.loads(Path(nopierce_json).read_text())

    if full.get("arm") != nop.get("arm"):
        raise ValueError(
            f"both conditions must score the same arm, got {full.get('arm')!r} "
            f"and {nop.get('arm')!r} — comparing different arms would be a "
            "category error")

    lines = list(_HEADER)
    lines += [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    & \\multicolumn{2}{c}{All five predicates} & "
        "\\multicolumn{2}{c}{Excluding "
        f"{_escape('canPierce')}}} \\\\",
        "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        f"    Metric & $\\Delta$ ($n{{=}}{full['n_pairs']}$) & 95\\% CI & "
        f"$\\Delta$ ($n{{=}}{nop['n_pairs']}$) & 95\\% CI \\\\",
        "    \\midrule",
    ]

    for key in _METRIC_ORDER:
        a, b = full["metrics"].get(key), nop["metrics"].get(key)
        if a is None and b is None:
            continue
        cells = []
        for m in (a, b):
            if m is None:
                cells += ["--", "--"]
                continue
            # A star, not a bare number: a reader must be able to see which
            # intervals exclude zero without recomputing anything.
            star = "$^{*}$" if m.get("significant") else ""
            cells.append(f"{m['diff']:+.3f}{star}")
            cells.append(f"[{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]")
        lines.append(f"    {_METRIC_LABEL[key]} & " + " & ".join(cells)
                     + " \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption[Paired bootstrap: student against the VLM]{Paired "
        f"bootstrap over arm~{full['arm'].upper()}, "
        f"{full['n_boot']:,} replicates. $\\Delta$ is student minus VLM, so a "
        "negative value favours the VLM on accuracy-like metrics and a "
        "positive one favours the student on MAE, where lower is better. "
        "Starred rows have a 95\\% interval excluding zero. Over all five "
        "predicates the VLM leads significantly on four metrics; excluding "
        f"{_escape('canPierce')} every interval spans zero, so the whole "
        "significant gap is that one predicate --- whose failure is traced to "
        "a defect in the labelling corpus rather than to the model.}",
        "  \\label{tab:e7-significance}",
        "\\end{table}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()


#: E9 ladder keys, with the descriptive label each condition is reported
#: under. The balanced set is the reported one: the natural eval is 71.3%
#: positive, where an always-yes judge scores 0.713 accuracy and 0.833 F1 —
#: above several of the systems being compared.
_E9_ARMS = (
    ("G+", "Ground-truth boxes and classes"),
    ("G", "Ground-truth boxes only"),
    ("E", "End-to-end"),
)


def write_e9_arms_tex(
    ladder_json: Path | str,
    out_path: Path | str,
    eval_set: str = "BALANCED 50/50",
) -> Path:
    """Render the v5 student against the live VLM, balanced accuracy first.

    Column order is deliberate. Accuracy and F1 on the natural eval partly
    measure the class prior, so balanced accuracy leads and the rest follow;
    a reader who takes the first number column gets the one that cannot be
    gamed by answering yes.

    Only the v5 student is shown: v5 supersedes v4 (the fork-coverage fix,
    narrated in prose) and this table is not the place to relitigate that.
    """
    d = json.loads(Path(ladder_json).read_text())

    def get(arm: str, judge: str) -> dict | None:
        return d.get(f"{eval_set}|{arm}|{judge}")

    lines = list(_HEADER)
    lines += [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabularx}{\\textwidth}"
        "{>{\\raggedright\\arraybackslash}Xlccccc}",
        "    \\toprule",
        "    Condition & Judge & B.acc & Acc & F1 & Prec & MAE \\\\",
        "    \\midrule",
    ]

    def row(head: str, label: str, m: dict) -> str:
        def cell(key: str, places: int = 3) -> str:
            return _fmt(m.get(key), places)
        return (f"    {head} & {label} & "
                f"{cell('bacc')} & {cell('acc')} & {cell('f1')} & "
                f"{cell('prec')} & {cell('mae', 2)} \\\\")

    sd_max_unit, sd_max_mae = 0.0, 0.0
    for i, (key, label) in enumerate(_E9_ARMS):
        if i > 0:
            lines.append("    \\addlinespace")
        arm_rows = []
        vlm = get(key, "VLM")
        if vlm:
            arm_rows.append(("VLM", vlm))
        tl = get(key, "teacher_labels")
        if tl:
            arm_rows.append(("teacher (stored)", tl))
        v5, sd = get(key, "student_v5_mean"), get(key, "student_v5_sd")
        if v5:
            arm_rows.append(("student", v5))
        n = len(arm_rows)
        head = (f"\\multirow{{{n}}}{{=}}{{\\textbf{{{label}}}}}"
                if n > 1 else f"\\textbf{{{label}}}")
        for j, (judge, m) in enumerate(arm_rows):
            lines.append(row(head if j == 0 else "", judge, m))
        if sd:
            sd_max_unit = max(sd_max_unit,
                               *(sd[k] for k in ("bacc", "acc", "f1", "prec")))
            sd_max_mae = max(sd_max_mae, sd["mae"])

    balanced = eval_set.startswith("BALANCED")
    note = (
        "Scores on a 50/50 resample of the human labels: the natural set is "
        "71.3\\,\\% positive, where a judge answering yes to everything reaches "
        "0.713 accuracy and 0.833 F1. The resample discards 83 positives per "
        "draw and averages 20 draws, so it is a reweighting of the same "
        "judgements and not a truer ground truth; it is reported because it "
        "makes that dependence visible. Balanced accuracy leads for the same "
        "reason and is immune by construction, scoring 0.500 for the "
        "always-yes judge. "
        if balanced else
        "Scores on the natural human labels, which are 71.3\\,\\% positive. "
        "Accuracy and F1 on this set partly measure the class prior --- an "
        "always-yes judge reaches 0.713 and 0.833 --- so balanced accuracy "
        "leads. ")

    lines += [
        "    \\bottomrule",
        "  \\end{tabularx}",
        "  \\caption[Student against a live VLM]{Student and live VLM judged "
        "against the same human labels on the same scenes, under the three "
        f"conditions described in the text. {note}Student rows are means "
        f"over three seeds; the spread across seeds is at most "
        f"$\\pm${sd_max_unit:.3f} on the 0--1 metrics and $\\pm${sd_max_mae:.2f} "
        "on MAE, in every condition.}",
        "  \\label{tab:e9-arms}",
        "\\end{table}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


#: Ladder rungs in order of increasing difficulty, with the claim each
#: licenses. The drop between rungs is the result; the top number alone
#: overstates how well the model generalises.
#: The experiment identifiers (e1a, e1b, ...) stay out of the thesis: they
#: are internal names and carry no meaning for a reader. The last level is
#: emphasised because it is the strongest claim the dataset supports.
_RUNG_LABEL = {
    "e1a-scene": "Unseen scenes",
    "e1b-pair": "Unseen pairings",
    "e1c-actor_only": "Unseen actor mesh",
    "e1c-strict": "Unseen both meshes",
}
_RUNG_EMPHASISED = "e1c-strict"
_RUNG_ORDER = ("e1a-scene", "e1b-pair", "e1c-actor_only", "e1c-strict")


def write_ladder_tex(
    snapshot_path: Path | str,
    out_path: Path | str,
    corpus: str = "v5",
) -> Path:
    """Render the generalisation ladder for one corpus version.

    Two macros for the same reason as the per-predicate table: canPourInto
    is supervised by a hand-authored rule, so a macro including it mixes
    reproducing a teacher with recovering a rule.
    """
    rungs = ladder_rows(snapshot_path, corpus=corpus)
    if not any(rungs.get(r) for r in _RUNG_ORDER):
        raise ValueError(
            f"no runs for corpus {corpus!r} in the snapshot — refusing to "
            "write a ladder with no numbers in it")

    lines = list(_HEADER)
    lines += [
        f"% Corpus: {corpus}. Calibrated, seed-averaged.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    Level & Macro F1 (6 pred.) & Macro F1 (5 pred.) \\\\",
        "    \\midrule",
    ]

    for rung in _RUNG_ORDER:
        runs = rungs.get(rung, [])
        if not runs:
            continue
        desc = _RUNG_LABEL[rung]
        stats = _per_predicate_stats(runs, "calibrated")
        # Per-seed macros, so the spread describes the rung and not the
        # spread of a single averaged number.
        six, five = [], []
        for r in runs:
            b = r["metrics"]["calibrated"]
            vals = {p: b["per_predicate"][p]["f1"]
                    for p in b.get("per_predicate", {})
                    if _support(b, p) > 0}
            if not vals:
                continue
            six.append(statistics.mean(vals.values()))
            rest = [v for p, v in vals.items() if p not in RULE_SUPERVISED]
            if rest:
                five.append(statistics.mean(rest))
        if not six:
            continue
        m6, s6 = _mean_std(six)
        m5, s5 = _mean_std(five) if five else (None, 0.0)
        if rung == _RUNG_EMPHASISED:
            name = f"\\textbf{{{desc}}}"
            six_cell = f"$\\mathbf{{{m6:.3f} \\pm {s6:.3f}}}$"
            five_cell = ("--" if m5 is None
                         else f"$\\mathbf{{{m5:.3f} \\pm {s5:.3f}}}$")
        else:
            name = desc
            six_cell = f"${m6:.3f} \\pm {s6:.3f}$"
            five_cell = ("--" if m5 is None
                         else f"${m5:.3f} \\pm {s5:.3f}$")
        lines.append(f"    {name} & {six_cell} & {five_cell} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption[Generalisation levels]{Calibrated macro F1 at each "
        "level, with the spread across seeds. Each level withholds strictly "
        "more than the one above it. The second macro excludes "
        f"{_escape('canPourInto')}, which is supervised by a hand-authored "
        "rule. Instance levels are averaged over six seeds, the others over "
        "three.}",
        "  \\label{tab:ladder}",
        "\\end{table}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_fidelity_tex(
    snapshot_path: Path | str,
    out_path: Path | str,
    corpus: str = "v5",
    rung: str = "e1a-scene",
) -> Path:
    """Render score fidelity: how closely the student tracks the teacher.

    This is faithfulness and not correctness, and the caption has to say so:
    a student reproducing a teacher perfectly scores 0 error here and is
    still wrong wherever the teacher was.
    """
    runs = ladder_rows(snapshot_path, corpus=corpus).get(rung, [])
    if not runs:
        raise ValueError(
            f"no runs for rung {rung!r} on corpus {corpus!r} — refusing to "
            "write a fidelity table with no numbers in it")

    per: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        fid = r["metrics"]["calibrated"].get("score_fidelity", {})
        for pred, m in fid.items():
            if m.get("n"):
                d = per.setdefault(pred, {"mae": [], "corr": [], "n": []})
                d["mae"].append(m["mae"])
                d["corr"].append(m["corr"])
                d["n"].append(m["n"])

    lines = list(_HEADER)
    lines += [
        f"% Corpus: {corpus}, rung {rung}. Calibrated, seed-averaged.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{lrcc}",
        "    \\toprule",
        "    Predicate & Pairs & MAE & Correlation \\\\",
        "    \\midrule",
    ]
    for pred in PREDICATE_ORDER:
        d = per.get(pred)
        if not d:
            continue
        lines.append(
            f"    {_escape(pred)} & {statistics.mean(d['n']):.0f} & "
            f"{_fmt(statistics.mean(d['mae']))} & "
            f"{_fmt(statistics.mean(d['corr']))} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption[Score fidelity to the teacher]{How closely the student "
        "reproduces the teacher's graded score on held-out pairs, averaged "
        "over seeds. Mean absolute error is on the unit interval the model "
        "is trained against, so 0.10 is a tenth of the full range. "
        "\\textbf{This measures faithfulness and not correctness}: a student "
        "that reproduced its teacher exactly would score zero error here and "
        "would still be wrong wherever the teacher was. Whether either is "
        "right is asked in Section~\\ref{sec:res-human} against human labels.}",
        "  \\label{tab:fidelity}",
        "\\end{table}",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
