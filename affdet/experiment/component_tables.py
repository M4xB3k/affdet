"""Export E5 ledger rows -> committed results (md/json) + the thesis table.

Every number reaches the thesis via ``\\input{tables/...}`` generated from
the ledger, never transcribed: the v3/v3 naming collision already produced
one wrong results table in this project.

Results are written as markdown and JSON into ``data/eval/`` first and the
LaTeX table generated from the same rows, so the committed record and the
thesis always agree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from affdet.experiment.ledger import load_runs

#: Human-readable detector names for the table.
_DETECTOR_LABELS = {
    "yolo11s": "YOLOv11-S",
    "yolov8s-worldv2": "YOLO-World-S",
    "yolov8s-world": "YOLO-World-S",
    "grounding-dino": "Grounding DINO-T",
}

#: The generalisation ladder. Da and Db rows carry identical detector
#: labels but score different splits, so the table must name which.
_ARM_LABELS = {
    "e5-Da-": "Scene-disjoint",
    "e5-Db-": "Instance-disjoint",
}


def _arm_prefix(name: str) -> str:
    for p in _ARM_LABELS:
        if str(name).startswith(p):
            return p
    return ""


def _escape(s: str) -> str:
    """Underscores are math-mode subscripts in LaTeX and break the build."""
    return str(s).replace("_", r"\_")


def _fmt(v: Any, places: int = 3) -> str:
    return "--" if v is None else f"{v:.{places}f}"


def _fmt_pair(a: Any, b: Any, places: int = 3) -> tuple[str, str]:
    """Format two competing values, bolding the larger.

    Bold marks the winner of the comparison the row makes, which is the
    convention a reader scanning the table will assume. Ties are left
    unbolded rather than bolding both, since a tie is not a win.
    """
    fa, fb = _fmt(a, places), _fmt(b, places)
    if a is None or b is None or a == b:
        return fa, fb
    return (f"\\textbf{{{fa}}}", fb) if a > b else (fa, f"\\textbf{{{fb}}}")


def load_component_runs(
    ledger_path: Path | str,
    include: tuple[str, ...] | None = None,
) -> list[dict]:
    """Component rows from the ledger, one per arm, newest wins.

    The ledger holds both experiment kinds, so affordance rows are filtered
    out — one in the detector table would be nonsense. Re-running an arm
    supersedes its earlier row instead of appearing twice; silently merging
    stale rows with fresh ones is exactly how the E1 ladder once produced a
    wrong 6-seed family.

    ``include`` keeps only arms whose name starts with one of the given
    prefixes. "Newest wins" keys on arm NAME, so it can only retire a row
    that was re-run under the SAME name — it is no defence against a row
    whose name has since changed. The pre-ladder E5 rows (``e5-yolo11s-ft``,
    before the Da/Db split existed) were scored with detectors trained on a
    superseded corpus, and no amount of re-running the current arms will
    displace them. Pass ``include=("e5-Da-", "e5-Db-")`` to build the table
    from the current ladder only.
    """
    rows = [r for r in load_runs(ledger_path) if r.get("kind") == "component"]
    if include is not None:
        rows = [r for r in rows
                if str(r["config"]["name"]).startswith(tuple(include))]
    by_arm: dict[str, dict] = {}
    for r in rows:
        by_arm[r["config"]["name"]] = r
    return list(by_arm.values())


def load_classification_runs(
    ledger_path: Path | str,
    include: tuple[str, ...] | None = ("e5-Da-", "e5-Db-"),
) -> list[dict]:
    """Probe-vs-class-head rows, one per arm, newest wins.

    Separate from ``load_component_runs`` because these rows measure a
    different thing (which classifier names a crop correctly) over a
    different unit (matched objects, not boxes), and only some arms have one.
    """
    rows = [r for r in load_runs(ledger_path)
            if r.get("kind") == "classification"]
    if include is not None:
        rows = [r for r in rows
                if str(r["config"]["name"]).startswith(tuple(include))]
    by_arm: dict[str, dict] = {}
    for r in rows:
        by_arm[r["config"]["name"]] = r
    return list(by_arm.values())


def _arm_view(run: dict) -> dict:
    cfg = run["config"]
    det = run["metrics"]["detection"]
    agn = run["metrics"]["detection_class_agnostic"]
    return {
        "name": cfg["name"],
        "detector": cfg["detector"],
        "detector_label": _DETECTOR_LABELS.get(cfg["detector"], cfg["detector"]),
        "zero_shot": bool(cfg.get("zero_shot")),
        "iou_thresh": cfg.get("iou_thresh"),
        "conf": cfg.get("conf"),
        "arm_label": _ARM_LABELS.get(_arm_prefix(cfg["name"]), ""),
        "arm_prefix": _arm_prefix(cfg["name"]),
        "n_scenes": run.get("n_scenes_scored"),
        "detection": det,
        "detection_class_agnostic": agn,
    }


def _ordered(runs: list[dict]) -> list[dict]:
    """Group by ladder rung (scene-disjoint before instance-disjoint), then
    fine-tuned before zero-shot: the table reads as an argument — the easier
    generalisation first, then the harder one, and within each what the
    pipeline deploys before what the alternatives score."""
    arms = [_arm_view(r) for r in runs]
    return sorted(arms, key=lambda a: (a["arm_prefix"], a["zero_shot"],
                                       a["name"]))


def write_component_tex(
    runs: list[dict],
    out_path: Path | str,
    per_class: bool = False,
) -> Path:
    """Render the component-quality table."""
    if not runs:
        raise ValueError(
            "no component runs in the ledger — refusing to write a table with "
            "no numbers in it")

    arms = _ordered(runs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iou = arms[0]["iou_thresh"]
    lines = [
        "% GENERATED FILE — DO NOT EDIT BY HAND.",
        "% Produced by `affdet.scripts.component_tables` from the experiment",
        "% ledger (experiments/runs.jsonl). Hand edits will be overwritten.",
        "% If a number here looks wrong, fix the ledger or the exporter,",
        "% never this file.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{lllcccc}",
        "    \\toprule",
        "    & & & \\multicolumn{3}{c}{Class-aware} & Box-only \\\\",
        "    \\cmidrule(lr){4-6}\\cmidrule(lr){7-7}",
        "    Split & Detector & Training & P & R & F1 & F1 \\\\",
        "    \\midrule",
    ]

    # Rule between ladder rungs, not between fine-tuned and zero-shot: with
    # two rungs the rung is the outer grouping, and a separator inside one
    # rung would suggest the rows above and below score the same split.
    prev_arm = None
    for i, arm in enumerate(arms):
        if i > 0 and arm["arm_prefix"] != prev_arm:
            lines.append("    \\midrule")
        prev_arm = arm["arm_prefix"]
        det, agn = arm["detection"], arm["detection_class_agnostic"]
        training = "zero-shot" if arm["zero_shot"] else "fine-tuned"
        lines.append(
            f"    {_escape(arm['arm_label'] or '--')} & "
            f"{_escape(arm['detector_label'])} & {training} & "
            f"{_fmt(det.get('precision'))} & {_fmt(det.get('recall'))} & "
            f"{_fmt(det.get('f1'))} & {_fmt(agn.get('f1'))} \\\\"
        )

    lines += ["    \\bottomrule", "  \\end{tabular}"]

    if per_class:
        lines += [
            "  \\par\\smallskip",
            "  \\begin{tabular}{lccc}",
            "    \\toprule",
            "    Class & P & R & F1 \\\\",
            "    \\midrule",
        ]
        best = next((a for a in arms if not a["zero_shot"]), arms[0])
        for cls, m in sorted(best["detection"].get("per_class", {}).items()):
            lines.append(
                f"    {_escape(cls)} & {_fmt(m.get('precision'))} & "
                f"{_fmt(m.get('recall'))} & {_fmt(m.get('f1'))} \\\\"
            )
        lines += ["    \\bottomrule", "  \\end{tabular}"]

    n_scenes = arms[0]["n_scenes"]
    # The caption must stand on its own: a LaTeX table is read detached from
    # the markdown that carries the full Reliability section, so a caveat
    # that changes how a column may be read belongs here too.
    confs = {a["conf"] for a in arms if a.get("conf") is not None}
    conf_note = ""
    if len(confs) > 1:
        # Name the exceptions against the majority threshold rather than
        # listing every arm: a caption that repeats "0.05" three times buries
        # the one value a reader needs to notice.
        majority = max(confs, key=lambda c: sum(
            1 for a in arms if a.get("conf") == c))
        odd = sorted({(a["detector_label"], a["conf"]) for a in arms
                      if a.get("conf") not in (None, majority)})
        conf_note = (
            f" Confidence thresholds differ across architectures: {majority} "
            "throughout, except "
            + ", ".join(f"{_escape(lbl)} at {c}" for lbl, c in odd)
            + ", whose threshold also sets its text-matching threshold and is "
            "therefore not the same quantity as a YOLO confidence. Precision "
            "is comparable within an architecture, not across them; recall "
            "and Box-only are the safer cross-architecture comparisons.")
    lines += [
        "  \\caption[Component quality: detection]{Detection quality on the "
        f"held-out render splits, at IoU $\\geq$ {iou}. "
        "\\emph{Scene-disjoint} holds out arrangements of meshes seen in "
        "training; \\emph{instance-disjoint} holds out whole mesh instances, "
        "so the detector meets novel objects of a class it knows. "
        f"\\emph{{Class-aware}} requires the predicted label to match; "
        "\\emph{Box-only} scores localisation alone, which is what governs "
        "this pipeline, since every crop is re-classified by the DINOv2 probe "
        "and detector labels are discarded. Zero-shot rows use the same "
        "taxonomy as their vocabulary but were never trained on the render "
        f"corpus.{conf_note}}}",
        "  \\label{tab:component-quality}",
        "\\end{table}",
    ]

    # Second table, not extra columns on the first: this one answers a
    # different question (does mesh novelty hurt?) over a different unit
    # (objects, split by mesh identity) and only the instance-disjoint arm
    # can populate it.
    gen = [a for a in arms if a["detection"].get("subgroups")]
    if gen:
        nov = gen[0]["detection"]["subgroups"]["novel"]["n_gold"]
        seen = gen[0]["detection"]["subgroups"]["rest"]["n_gold"]
        lines += [
            "",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\begin{tabular}{llccc}",
            "    \\toprule",
            "    & & \\multicolumn{2}{c}{Naming (class-aware)} & Box-only \\\\",
            "    \\cmidrule(lr){3-4}\\cmidrule(lr){5-5}",
            "    Detector & Training & Novel & Seen & Novel \\\\",
            "    \\midrule",
        ]
        for a in gen:
            d = a["detection"]["subgroups"]
            b = a["detection_class_agnostic"].get("subgroups", {})
            lines.append(
                f"    {_escape(a['detector_label'])} & "
                f"{'zero-shot' if a['zero_shot'] else 'fine-tuned'} & "
                f"{_fmt(d['novel']['recall'])} & {_fmt(d['rest']['recall'])} & "
                f"{_fmt(b.get('novel', {}).get('recall'))} \\\\")
        lines += [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption[Generalisation to unseen mesh instances]{Recall on "
            f"mesh instances held out of training ({nov} objects) against "
            f"instances seen in training ({seen} objects) that share the same "
            "test scenes, so lighting and clutter are held constant. Recall "
            "rather than F1: a prediction matching no ground-truth object "
            "cannot be attributed to either group, so subgroup precision "
            "would be an upper bound. A fine-tuned detector loses ground on "
            "novel meshes when \\emph{naming} them, not when \\emph{finding} "
            "them, and localisation is the column this pipeline depends "
            "on, since it discards detector labels and re-classifies every "
            "crop with the DINOv2 probe. A zero-shot detector scoring higher on "
            "novel meshes never trained on either group, which also rules "
            "out the novel set being simply the harder objects.}",
            "  \\label{tab:component-generalisation}",
            "\\end{table}",
        ]

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_component_json(runs: list[dict], out_path: Path | str) -> Path:
    """Machine-readable companion to the table."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arms": _ordered(runs),
        "provenance": [
            {"name": r["config"]["name"], "run_id": r.get("run_id"),
             "git_sha": r.get("git_sha"), "timestamp": r.get("timestamp"),
             "scenes_hash": r.get("scenes_hash")}
            for r in runs
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def write_component_markdown(runs: list[dict], out_path: Path | str,
                             classification: list[dict] | None = None) -> Path:
    """Committed results doc, including what the numbers do and do not
    support — the "Reliability" section the other data/eval docs carry."""
    arms = _ordered(runs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ft = [a for a in arms if not a["zero_shot"]]
    zs = [a for a in arms if a["zero_shot"]]
    n_scenes = arms[0]["n_scenes"]
    iou = arms[0]["iou_thresh"]

    lines = [
        "# E5 — Component quality: detection",
        "",
        "**Generated from the experiment ledger.** Do not edit by hand; "
        "re-run `affdet.scripts.component_tables` instead.",
        "",
        f"Synthetic-only, on the held-out splits of the render corpus, "
        f"scored at IoU >= {iou} against the exact ground-truth boxes and "
        "classes the renderer wrote. Scene counts differ per level and are "
        "given in the table.",
        "",
        "Two levels of generalization: **scene-disjoint** holds out "
        "arrangements of meshes seen in training, **instance-disjoint** holds "
        "out whole mesh instances, so the detector meets novel objects of a "
        "class it knows.",
        "",
        "## Detection",
        "",
        "| Split | Detector | Training | Scenes | P | R | F1 | Box-only F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        det, agn = a["detection"], a["detection_class_agnostic"]
        lines.append(
            f"| {a['arm_label'] or '--'} | {a['detector_label']} | "
            f"{'zero-shot' if a['zero_shot'] else 'fine-tuned'} | "
            f"{a['n_scenes']} | "
            f"{_fmt(det.get('precision'))} | {_fmt(det.get('recall'))} | "
            f"{_fmt(det.get('f1'))} | {_fmt(agn.get('f1'))} |"
        )

    lines += [
        "",
        "*Class-aware* requires the predicted label to match the gold class. "
        "*Box-only* scores localisation alone — the number that actually "
        "governs this pipeline, which re-classifies every crop with the "
        "DINOv2 probe and discards the detector's own labels.",
        "",
    ]

    gen = [a for a in arms if a["detection"].get("subgroups")]
    if gen:
        nov = gen[0]["detection"]["subgroups"]["novel"]["n_gold"]
        seen = gen[0]["detection"]["subgroups"]["rest"]["n_gold"]
        lines += [
            "## Generalisation to unseen mesh instances",
            "",
            "The instance-disjoint arm reserves whole mesh instances from "
            "training, but moves whole *scenes* into the test split — and a "
            f"scene holds several objects. Of {nov + seen} scored objects, "
            f"{nov} are novel meshes and {seen} are meshes the detector "
            "trained on, sharing the same scenes. Pooling them hides the "
            "effect, so both are reported.",
            "",
            "Recall is reported rather than F1: a prediction matching no gold "
            "object cannot be attributed to either group, so subgroup "
            "precision would be an upper bound. Recall is exact.",
            "",
            "| Detector | Training | Naming, novel | Naming, seen | Gap | "
            "Box-only, novel | Box-only, seen |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in gen:
            d = a["detection"]["subgroups"]
            b = a["detection_class_agnostic"].get("subgroups", {})
            gap = d["novel"]["recall"] - d["rest"]["recall"]
            lines.append(
                f"| {a['detector_label']} | "
                f"{'zero-shot' if a['zero_shot'] else 'fine-tuned'} | "
                f"{_fmt(d['novel']['recall'])} | {_fmt(d['rest']['recall'])} | "
                f"{gap:+.3f} | "
                f"{_fmt(b.get('novel', {}).get('recall'))} | "
                f"{_fmt(b.get('rest', {}).get('recall'))} |")
        lines += [
            "",
            "Read the two halves against each other: where a fine-tuned "
            "detector loses ground on a novel mesh it is *naming*, not "
            "*finding*. Localisation is near-identical on both groups, which "
            "is the column this pipeline depends on — it discards detector "
            "labels and re-classifies each crop with the DINOv2 probe. The "
            "generalisation weakness sits precisely in the component the "
            "pipeline does not use.",
            "",
            "A zero-shot arm scoring *higher* on novel meshes is not an "
            "anomaly: it never trained on either group, so its scores track "
            "how nameable an object is in ordinary English rather than what "
            "this detector has met. That also rules out the novel set simply "
            "being the harder objects — a model with no exposure to the "
            "corpus ranks them the other way.",
            "",
        ]

    lines += [
    ]

    if classification:
        cls_rows = sorted(classification, key=lambda r: r["config"]["name"])
        lines += [
            "## Naming: DINOv2 probe vs. the detector's own class head",
            "",
            "Both classifiers see the **same crop**, cut from the same "
            "detector-predicted box, so any difference is the classifier and "
            "not the localisation. Gold objects the detector never boxed are "
            "excluded rather than charged to either side: a detection miss is "
            "not a classification error.",
            "",
            "| Arm | Objects | Probe | Class head | Probe advantage |",
            "|---|---|---|---|---|",
        ]
        for r in cls_rows:
            c = r["metrics"]["classification"]
            lines.append(
                f"| {r['config']['name']} | {c['n']} | "
                f"{_fmt(c['probe']['accuracy'])} | "
                f"{_fmt(c['classhead']['accuracy'])} | "
                f"{c['delta_accuracy']:+.4f} |")

        sub_rows = [r for r in cls_rows
                    if r["metrics"]["classification"].get("subgroups")]
        if sub_rows:
            lines += [
                "",
                "Split by whether the probe had ever seen that mesh:",
                "",
                "| Arm | Meshes | Objects | Probe | Class head | "
                "Probe advantage |",
                "|---|---|---|---|---|---|",
            ]
            for r in sub_rows:
                for g, label in (("novel", "never seen"), ("rest", "seen")):
                    v = r["metrics"]["classification"]["subgroups"][g]
                    d = v["probe"]["accuracy"] - v["classhead"]["accuracy"]
                    lines.append(
                        f"| {r['config']['name']} | {label} | {v['n']} | "
                        f"{_fmt(v['probe']['accuracy'])} | "
                        f"{_fmt(v['classhead']['accuracy'])} | {d:+.4f} |")
            lines += [
                "",
                "This is the measurement the pipeline's design rests on. On "
                "meshes seen in training the class head is marginally ahead, "
                "so discarding its labels costs a fraction of a point. On a "
                "mesh it has never seen the class head falls away while the "
                "probe holds, and discarding those labels gains far more than "
                "it cost. A detector's class head learns the instances it was "
                "trained on; frozen DINOv2 features describe what an object "
                "looks like, which is what survives meeting a new one.",
                "",
                "This arm is reported at n=1, like detection: under "
                "`split_from_dataset` the seed feeds nothing (the split comes "
                "from the built dataset) and the probe is deterministic, so "
                "repeated seeds return identical numbers rather than samples "
                "of a distribution. This was verified rather than assumed: "
                "the scene-disjoint arm was run at three seeds and returned "
                "bit-identical metrics. Runs made after `d1a0367` also carry "
                "`seed_affects_result: false` in the ledger; the rows here "
                "predate that field.",
                "",
            ]

    lines += [
        "## Reliability — what these numbers do and do not support",
        "",
        "**They support:** a like-for-like comparison of detectors on this "
        "corpus. Every arm was scored on the same held-out scenes, over the "
        "same 52-class vocabulary, with the same IoU threshold and the same "
        "matching code, so differences between rows are attributable to the "
        "detector (subject to the confidence-threshold caveat below, where "
        "it applies).",
        "",
    ]

    confs = {a["conf"] for a in arms if a.get("conf") is not None}
    if len(confs) > 1:
        by_conf = ", ".join(
            f"{a['detector_label']} ({'zero-shot' if a['zero_shot'] else 'fine-tuned'}) "
            f"at {a['conf']}"
            for a in arms if a.get("conf") is not None)
        lines += [
            "**Arms did not all run at the same confidence threshold:** "
            f"{by_conf}. Grounding DINO's threshold also sets its "
            "`text_threshold`, which decides which words of the 52-class "
            "prompt may attach to a box; running it at the YOLO arms' 0.05 "
            "would label boxes with barely-activated words rather than make "
            "the setting comparable. The two thresholds are therefore not the "
            "same knob and cannot simply be equalised. Consequence: precision "
            "is not directly comparable *across architectures* — a higher "
            "threshold suppresses low-confidence predictions and flatters "
            "precision while costing recall. Recall and the box-only column "
            "are the safer cross-architecture comparisons; precision is "
            "comparable within an architecture (fine-tuned vs zero-shot), "
            "where the threshold is held fixed.",
            "",
        ]

    lines += [
        "**Deterministic, not single-seed.** Detection against a fixed "
        "checkpoint and a fixed split has no random component: repeat runs "
        "return bit-identical counts (pinned by a test). These are exact "
        "values, not one sample from a distribution, so no error bars are "
        "reported. What a seed *would* vary is which scenes land in the "
        "held-out split, not the score given that split.",
        "",
        "**They do not support** any claim about real photographs. This is "
        "rendered data end to end; the sim-to-real gap is not measured here "
        "and no number in this table should be read as a deployment "
        "estimate.",
        "",
        "**Zero-shot rows are not a like-for-like baseline.** They use the "
        "same taxonomy as their vocabulary, but were never trained on this "
        "corpus, and they read class names as English text (so the "
        "snake_case IDs are de-underscored for the prompt: `red_wine_glass` "
        "-> \"red wine glass\") while being scored against canonical IDs. "
        "They answer \"what does fine-tuning buy?\", not \"which detector is "
        "better?\".",
        "",
    ]
    if ft and zs:
        best_ft = max(ft, key=lambda a: a["detection"]["f1"])
        best_zs = max(zs, key=lambda a: a["detection"]["f1"])
        lines += [
            f"On this corpus the best fine-tuned arm "
            f"({best_ft['detector_label']}, F1 {_fmt(best_ft['detection']['f1'])}) "
            f"and the best zero-shot arm ({best_zs['detector_label']}, F1 "
            f"{_fmt(best_zs['detection']['f1'])}) differ by "
            f"{_fmt(best_ft['detection']['f1'] - best_zs['detection']['f1'])} F1.",
            "",
        ]
    lines += [
        "## Provenance",
        "",
        "| Arm | run_id | git SHA |",
        "|---|---|---|",
    ]
    for r in runs:
        lines.append(f"| {r['config']['name']} | `{r.get('run_id')}` | "
                     f"`{str(r.get('git_sha'))[:7]}` |")
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


def write_classification_tex(runs: list[dict], out_path: Path | str) -> Path:
    """Render the probe-vs-class-head table.

    Separate from the detection table because it answers a different question
    over a different unit: which classifier names a crop correctly, counted
    over matched objects rather than boxes.

    The novel/seen split is the point of the table. Pooled, the probe looks
    marginally ahead and the design argument looks thin; split, the class head
    is slightly better on meshes it trained on and collapses on meshes it never
    met, which is what justifies discarding its labels.
    """
    if not runs:
        raise ValueError(
            "no classification runs in the ledger — refusing to write a table "
            "with no numbers in it")

    rows = sorted(runs, key=lambda r: r["config"]["name"])
    lines = [
        "% GENERATED FILE — DO NOT EDIT BY HAND.",
        "% Produced by `affdet.experiment.component_tables` from the",
        "% experiment ledger. Hand edits will be overwritten.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{llrcc}",
        "    \\toprule",
        "    Split & Meshes & Objects & DINOv2 probe & Class head \\\\",
        "    \\midrule",
    ]

    for r in rows:
        c = r["metrics"]["classification"]
        name = _escape(r["config"]["name"])
        subs = c.get("subgroups")
        if not subs:
            pr, ch = _fmt_pair(c["probe"]["accuracy"],
                               c["classhead"]["accuracy"])
            lines.append(f"    {name} & all & {c['n']} & {pr} & {ch} \\\\")
            continue
        for key, label in (("novel", "never seen"), ("rest", "seen")):
            v = subs[key]
            head = name if key == "novel" else ""
            pr, ch = _fmt_pair(v["probe"]["accuracy"],
                               v["classhead"]["accuracy"])
            lines.append(
                f"    {head} & {label} & {v['n']} & {pr} & {ch} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption[Naming: DINOv2 probe against the detector's class head]{"
        "Classification accuracy on matched objects, split by whether the probe "
        "had met that mesh instance in training. Both classifiers are scored on "
        "the \\emph{same crop}, cut from the same predicted box, so the "
        "difference between the columns is the classifier and not the "
        "localisation; objects the detector never boxed are excluded from both "
        "sides, since a detection miss is not a naming error. On meshes seen in "
        "training the detector's class head is marginally ahead, so discarding "
        "its labels costs a fraction of a point. On a mesh it has never met the "
        "class head falls away while the probe holds near its seen-mesh level. "
        "A class head learns the instances it was trained on; frozen DINOv2 "
        "features describe appearance, which is what survives meeting a new "
        "instance.}",
        "  \\label{tab:classification}",
        "\\end{table}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
