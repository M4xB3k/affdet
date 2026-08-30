#!/usr/bin/env python3
"""Build a pair list for the manual labelling tool.

Two modes:

  --dry-run   Sample pairs from the rendered pair corpus with NO reference to
              any label. Safe to run now; produces a practice list.

  (default)   Sample from a split's test set, balanced on the VLM's
              thresholded label. Requires --labels and --split-map, and is
              blocked on the split decision (state doc §5).

**The output never contains the VLM score or the model prediction.** Using the
VLM score to *select* pairs is fine and declared here; letting the labeller
*see* it is not (tools/docs/labeling_protocol.md §2). The selection metadata is
written to a sidecar so the sampling is reproducible and reweightable at
analysis time, while the list the tool loads stays blind.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from affdet.intersection_matrix import is_candidate_pair
from affdet.label.manual_tool import (
    LABELLED_PREDICATES, compute_pair_id, grid_image_name,
)

#: load_labels() returns the NORMALIZED score (score/9, per
#: affdet.schemas.LabelEntry / affdet/label/vlm_resolver.py), not the raw 0-9
#: VLM prompt scale -- these constants are on that normalized scale. The
#: original raw-scale threshold (5) and border band (4-6) map to ~0.556 and
#: ~0.444-0.667 respectively.
VLM_THRESHOLD = 5.0 / 9.0
BORDER_LO, BORDER_HI = 4.0 / 9.0, 6.0 / 9.0


def score_band(score: float) -> str:
    """Classify a normalized (0-1) VLM score into border/pos/neg."""
    if BORDER_LO <= score <= BORDER_HI:
        return "border"
    return "pos" if score >= VLM_THRESHOLD else "neg"


def _split_instance(inst: str) -> str:
    """'apple_003' -> 'apple'."""
    head, _, tail = inst.rpartition("_")
    return head if tail.isdigit() else inst


def _pair_dirs(renders: Path) -> dict[tuple[str, str], Path]:
    out = {}
    for d in renders.iterdir():
        if not d.is_dir() or "__" not in d.name:
            continue
        actor, _, recip = d.name.partition("__")
        out[(actor, recip)] = d
    return out


def _angles(pair_dir: Path) -> list[str]:
    mf = pair_dir / "manifest.json"
    if not mf.exists():
        return []
    try:
        return json.loads(mf.read_text()).get("angle_files", [])
    except json.JSONDecodeError:
        return []


def _emit(rows: list[dict], version: str, out: Path, renders: Path,
          selection: dict) -> None:
    pairs = []
    for r in rows:
        pd = renders / f"{r['actor']}__{r['recipient']}"
        pairs.append({
            "pair_id": compute_pair_id(
                r["actor"], r["recipient"], r["predicate"], version),
            "actor": {"obj_id": r["actor"], "obj_class": _split_instance(r["actor"])},
            "recipient": {"obj_id": r["recipient"],
                          "obj_class": _split_instance(r["recipient"])},
            "predicate": r["predicate"],
            "pair_dir": pd.name,
            "angles": _angles(pd),
            "split_level": r.get("split_level"),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"pair_list_version": version, "pairs": pairs}, indent=2))

    # Selection metadata lives OUTSIDE the pair list, so the tool can never
    # load it and the labeller can never see it — but the sampling stays
    # reproducible and the analysis can reweight for the balancing.
    side = out.with_suffix(".selection.json")
    side.write_text(json.dumps({
        "pair_list_version": version,
        "n": len(pairs),
        "warning": "Contains VLM scores used for SELECTION ONLY. Never merge "
                   "this into the pair list — the labeller must stay blind.",
        **selection,
    }, indent=2))
    print(f"wrote {len(pairs)} pairs -> {out}")
    print(f"selection metadata -> {side}")


def dry_run(args) -> int:
    """Practice list: pure structural sampling, no labels consulted at all."""
    renders = Path(args.renders)
    dirs = _pair_dirs(renders)
    if not dirs:
        raise SystemExit(f"no <actor>__<recipient> dirs under {renders}")

    rng = random.Random(args.seed)
    by_pred: dict[str, list[dict]] = defaultdict(list)
    for (actor, recip), d in dirs.items():
        a_cls, r_cls = _split_instance(actor), _split_instance(recip)
        for pred in LABELLED_PREDICATES:
            # Structural candidacy from the intersection matrix — the same
            # authority the renderer and labeller used. Hand-rolled class rules
            # here drifted from it immediately (a peeler was offered for
            # canCut, which is knife-only).
            if not is_candidate_pair(a_cls, r_cls, pred):
                continue
            if (d / grid_image_name(pred)).exists():
                by_pred[pred].append(
                    {"actor": actor, "recipient": recip, "predicate": pred,
                     "split_level": "dry-run"})

    per = max(1, args.n // max(1, len(by_pred)))
    rows: list[dict] = []
    for pred in LABELLED_PREDICATES:
        pool = by_pred.get(pred, [])
        rng.shuffle(pool)
        rows.extend(pool[:per])
    rng.shuffle(rows)
    rows = rows[:args.n]

    _emit(rows, args.version, Path(args.out), renders, {
        "mode": "dry-run",
        "seed": args.seed,
        "note": "Practice list. Sampled structurally from rendered pair dirs; "
                "no VLM label or model prediction was consulted.",
        "per_predicate": {p: sum(r["predicate"] == p for r in rows)
                          for p in LABELLED_PREDICATES},
    })
    return 0


def holdout_json_to_split_map(holdout: dict) -> dict[str, str]:
    """Convert an InstanceHoldout record (test_meshes/val_meshes/all_meshes,
    as written by runner.py's `record["holdout"]` for a holdout_mode=strict/
    actor_only run) into the flat {instance: split} shape real_run() expects.

    Meshes in all_meshes but neither test_meshes nor val_meshes are "train"
    -- the meshes the model actually trained on for that run.
    """
    test = set(holdout["test_meshes"])
    val = set(holdout["val_meshes"])
    return {
        m: ("test" if m in test else "val" if m in val else "train")
        for m in holdout["all_meshes"]
    }


def load_holdout_split_map(path: Path | str) -> dict[str, str]:
    """Read a holdout JSON file (the ledger's `holdout` field, saved to its
    own file) and convert it to a flat split map."""
    return holdout_json_to_split_map(json.loads(Path(path).read_text()))


def load_labels(path: Path | str) -> dict[tuple[str, str, str], float]:
    """Load VLM labels keyed by (actor, recipient, predicate) -> score.

    Accepts two shapes:
      - a flat JSONL file, one label row {actor, recipient, predicate, score}
        per line (the older labels_omni_score corpus).
      - a DIRECTORY of per-scene affdet.schemas.Labels JSON files
        ({"scene_id", "tier1": [{actor, recipient, predicate, score, ...}]})
        -- the real labels_all_merged_v3 corpus that Inferencer/
        AffordanceDataset actually consume. label_coverage.json (a
        corpus-wide summary with no `tier1` key) sits alongside these and
        is skipped rather than crashing the load.
    """
    labels: dict[tuple[str, str, str], float] = {}
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.glob("*.json")):
            doc = json.loads(p.read_text())
            for r in doc.get("tier1", []):
                if r["predicate"] in LABELLED_PREDICATES and r.get("score") is not None:
                    labels[(r["actor"], r["recipient"], r["predicate"])] = r["score"]
        return labels
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["predicate"] in LABELLED_PREDICATES and r.get("score") is not None:
                labels[(r["actor"], r["recipient"], r["predicate"])] = r["score"]
    return labels


def real_run(args) -> int:
    """Balanced sample from a split's test set.

    Balanced on the VLM's thresholded label and oversampling near the
    threshold: the corpus is heavily positive-skewed (canSupport 89% positive),
    so uniform sampling would spend the budget on easy positives and surface
    almost none of the disagreements E3 exists to find.
    """
    if not args.labels or not (args.split_map or args.holdout_json):
        raise SystemExit(
            "--labels and one of --split-map/--holdout-json are required for "
            "a real pair list. Use --dry-run to build a practice list now.")

    labels = load_labels(args.labels)

    split_map = (load_holdout_split_map(args.holdout_json) if args.holdout_json
                else json.loads(Path(args.split_map).read_text()))
    test_instances = {k for k, v in split_map.items() if v == "test"}

    renders = Path(args.renders)
    dirs = _pair_dirs(renders)
    rng = random.Random(args.seed)

    # Three strata per predicate. `border` (4-6) is where the teacher is least
    # certain and therefore where human disagreement is most informative.
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (actor, recip, pred), score in labels.items():
        if (actor, recip) not in dirs:
            continue
        if not is_candidate_pair(_split_instance(actor), _split_instance(recip), pred):
            continue
        if test_instances and not (
                actor in test_instances or recip in test_instances):
            continue
        band = score_band(score)
        strata[(pred, band)].append(
            {"actor": actor, "recipient": recip, "predicate": pred,
             "split_level": args.split_level, "_score": score, "_band": band})

    per_pred = max(1, args.n // len(LABELLED_PREDICATES))
    quota = {"border": per_pred // 2,
             "pos": per_pred - per_pred // 2 - (per_pred - per_pred // 2) // 2,
             "neg": (per_pred - per_pred // 2) // 2}
    quota["pos"] = per_pred - quota["border"] - quota["neg"]

    rows, audit = [], []
    for pred in LABELLED_PREDICATES:
        for band, want in quota.items():
            pool = strata.get((pred, band), [])
            rng.shuffle(pool)
            take = pool[:want]
            rows.extend(take)
            audit.append({"predicate": pred, "band": band,
                          "available": len(pool), "taken": len(take)})
    rng.shuffle(rows)

    scores = [{"actor": r["actor"], "recipient": r["recipient"],
               "predicate": r["predicate"], "vlm_score": r.pop("_score"),
               "band": r.pop("_band")} for r in rows]

    _emit(rows, args.version, Path(args.out), renders, {
        "mode": "balanced-test-sample",
        "seed": args.seed,
        "split_level": args.split_level,
        "vlm_threshold": VLM_THRESHOLD,
        "strata_quota": quota,
        "strata_audit": audit,
        "note": "Balanced on the VLM's thresholded label with the 4-6 band "
                "oversampled. Reweight at analysis time; the raw corpus is "
                "far more positive than this sample.",
        "selection_scores": scores,
    })
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True, help="pair_list_version")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--labels", help="labels_omni.jsonl (selection only)")
    ap.add_argument("--split-map", help="instance -> split JSON")
    ap.add_argument("--holdout-json",
                    help="InstanceHoldout record (test_meshes/val_meshes/"
                         "all_meshes), e.g. the ledger's `holdout` field for "
                         "a holdout_mode run, saved to its own file. "
                         "Alternative to --split-map.")
    ap.add_argument("--split-level", default="instance",
                    choices=("scene", "pair", "instance"))
    args = ap.parse_args(argv)
    return dry_run(args) if args.dry_run else real_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
