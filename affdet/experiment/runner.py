"""run_experiment — sequence extract→train→infer→eval for one config, cache
features by extract-hash, and append the run to the ledger."""
from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from affdet.experiment.config import ExperimentConfig
from affdet.experiment.registry import build_head
from affdet.experiment.ledger import append_run, git_sha
from affdet.model.dataset import build_split_map, instance_holdout


#: Predicates kept OUT of the macro average because the corpus makes them
#: structurally unscoreable -- a single verdict everywhere means a model scores
#: F1=1.0 by answering the same way every time, without looking at the image,
#: and averaging that in inflates the headline for free.
#:
#: This is a REPORTING switch, consumed by compute_metrics/write_metrics_csv
#: after predict(). It never reaches training, so changing it does not
#: invalidate checkpoints (pinned by test_degenerate_predicates.py).
#:
#: canPierce was listed here until 2026-08-11. Its v3 targets came from a
#: hardcoded `return 1.0` in pair_vlm_resolver (fixed in b80441f), giving the
#: whole corpus exactly one distinct value and zero negatives. The v4 relabel
#: gives it 1,092 real negatives and 18 distinct values, so it is now scored
#: like any other predicate. Note this makes v4 macros span 5 predicates
#: against v3's 4 -- for a like-for-like comparison against the v3 ladder,
#: recompute with macro_exclude=["canPierce"] from the same predictions.
DEGENERATE_PREDICATES: list[str] = []


def _labels_hash(labels_dir: Path) -> str:
    """Content hash of a label corpus. ExperimentConfig.extract_hash()
    deliberately excludes `labels`, so without this two runs on DIFFERENT
    label corpora are indistinguishable in the ledger -- the exact failure
    mode behind the v3/v3 naming collision."""
    h = hashlib.sha256()
    for p in sorted(Path(labels_dir).glob("*.json")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_experiment(cfg: ExperimentConfig, root: Path | str, mock: bool = False) -> dict:
    root = Path(root)
    feat_dir = root / "features" / cfg.extract_hash()
    run_id = f"{cfg.name}-{cfg.extract_hash()}-{git_sha()[:7]}"
    run_dir = root / "runs" / run_id
    ckpt = run_dir / "ckpt"; pred = run_dir / "pred"; results = run_dir / "results"
    pred_val = run_dir / "pred_val"
    for d in (feat_dir, ckpt, pred, pred_val, results):
        d.mkdir(parents=True, exist_ok=True)

    scenes = Path(cfg.scenes)
    labels = Path(cfg.labels)

    # 1. extract (cache) — skip if features already present
    if not any(feat_dir.glob("*.npz")):
        if mock:
            from affdet.extract.mock import MockExtractor
            MockExtractor(seed=cfg.seed).run(scenes, feat_dir)
        else:
            from affdet.extract.dinov2 import DINOv2Extractor
            DINOv2Extractor(variant=cfg.backbone,
                            crop_padding=cfg.crop_padding).run(
                scenes, feat_dir, bbox_source=cfg.bbox_source)

    # 2. train
    if mock:
        from affdet.model.mock import MockTrainer
        MockTrainer(seed=cfg.seed).fit(feat_dir, labels, ckpt)
    else:
        from affdet.model.train import Trainer
        Trainer(epochs=cfg.epochs, lr=cfg.lr, seed=cfg.seed,
                split_strategy=cfg.split_strategy,
                holdout_mode=cfg.holdout_mode,
                holdout_frac=cfg.holdout_frac,
                resolved_instances=cfg.resolved_instances,
                unbounded=cfg.unbounded,
                unbounded_neg_ratio=cfg.unbounded_neg_ratio,
                balanced=cfg.balanced,
                model_factory=lambda d: build_head(cfg.head, d)).fit(
            scenes, feat_dir, labels, ckpt)

    # 3. infer + 4. eval
    if mock:
        from affdet.model.mock import MockInferencer
        MockInferencer(seed=cfg.seed).predict(feat_dir, labels, ckpt, pred)
    else:
        from affdet.model.infer import Inferencer
        # Score ONLY the held-out TEST split (same seeded 70/15/15 scene split
        # Trainer used for early stopping) -- scoring every scene, as before
        # this fix, mixes memorized training data into the reported F1.
        Inferencer(resolved_instances=cfg.resolved_instances,
                   model_factory=lambda d: build_head(cfg.head, d),
                   split="test", seed=cfg.seed,
                   split_strategy=cfg.split_strategy,
                   holdout_mode=cfg.holdout_mode,
                   holdout_frac=cfg.holdout_frac,
                   all_pairs=cfg.all_pairs).predict(
            scenes, feat_dir, labels, ckpt, pred)
        # Also score VAL, so callers can calibrate per-predicate decision
        # thresholds (affdet.eval.metrics.calibrate_thresholds) without ever
        # touching test predictions -- calibrating and evaluating on the same
        # split would leak information and overstate reported F1.
        Inferencer(resolved_instances=cfg.resolved_instances,
                   model_factory=lambda d: build_head(cfg.head, d),
                   split="val", seed=cfg.seed,
                   split_strategy=cfg.split_strategy,
                   holdout_mode=cfg.holdout_mode,
                   holdout_frac=cfg.holdout_frac,
                   all_pairs=cfg.all_pairs).predict(
            scenes, feat_dir, labels, ckpt, pred_val)
    from affdet.eval.metrics import (
        calibrate_thresholds, compute_metrics, write_metrics_csv,
    )
    # Under a pair-level holdout, an unlucky mesh draw can strand a
    # low-instance predicate (canPeel: only 4 actor meshes corpus-wide) with
    # ZERO test pairs -- that's "no evidence", not a wrong prediction, and
    # must not be scored F1=0.0 and averaged in like a real failure. Scene/
    # pair splits don't have this pathology, so only opt in under holdout_mode.
    auto_exclude = cfg.holdout_mode is not None

    # Raw (threshold=0.5) metrics, for the honest "uncalibrated" column.
    results_csv = write_metrics_csv(pred, labels, results,
                                    macro_exclude=DEGENERATE_PREDICATES,
                                    auto_exclude_zero_support=auto_exclude)
    raw_metrics = compute_metrics(pred, labels,
                                  macro_exclude=DEGENERATE_PREDICATES,
                                  auto_exclude_zero_support=auto_exclude)

    # Calibrated metrics: thresholds are fit on the VAL split only and then
    # applied to TEST. Calibrating and evaluating on the same split would leak
    # information and overstate the reported F1.
    thresholds = calibrate_thresholds(pred_val, labels)
    cal_results = run_dir / "results_calibrated"
    write_metrics_csv(pred, labels, cal_results, thresholds=thresholds,
                      macro_exclude=DEGENERATE_PREDICATES,
                      auto_exclude_zero_support=auto_exclude)
    cal_metrics = compute_metrics(pred, labels, thresholds=thresholds,
                                  macro_exclude=DEGENERATE_PREDICATES,
                                  auto_exclude_zero_support=auto_exclude)

    record = {
        "run_id": run_id, "git_sha": git_sha(), "config": cfg.model_dump(),
        "features_dir": str(feat_dir), "ckpt_dir": str(ckpt),
        "results_csv": str(results_csv),
        "labels_hash": _labels_hash(labels),
        "split_strategy": cfg.split_strategy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {"raw": raw_metrics, "calibrated": cal_metrics},
    }
    if cfg.holdout_mode is not None:
        # Pair-level holdout (E1c/E1c'): a scene-level split_map has nothing
        # to do with what was actually trained on, so record WHICH meshes were
        # drawn instead -- without this the run is not reproducible (H2).
        # Same effective_classes the dataset/infer used, or the recorded draw
        # would not be the draw that actually ran.
        from affdet.label.resolved_instances import load_effective_class_map
        eff_map = (load_effective_class_map(cfg.resolved_instances)
                   if cfg.resolved_instances is not None else None)
        record["holdout"] = instance_holdout(
            scenes, seed=cfg.seed, frac=cfg.holdout_frac,
            effective_classes=eff_map).to_dict()
    else:
        split_map = build_split_map(scenes, cfg.split_strategy, seed=cfg.seed)
        record["split_sizes"] = {
            s: sum(1 for v in split_map.values() if v == s)
            for s in ("train", "val", "test")}
    append_run(root / "runs.jsonl", record)
    return record
