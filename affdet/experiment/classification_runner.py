"""run_classification_eval — DINOv2 probe vs. a detector's own class head.

The pipeline deliberately discards the detector's class label and
re-classifies each crop with a DINOv2 linear probe. This produces the
measurement justifying that decision.

Ported from a cluster script that established the methodology but lived
outside the repo, with no test, no config, and no ledger row — so its
result could not be reproduced or defended. The design is kept intact:

  1. Split scenes by the seeded split (probe never sees eval scenes).
  2. Fit the probe on oracle-crop features from the train split.
  3. On the eval split, run the detector and IoU-match its boxes to gold.
  4. For every matched pair, take the detector's own predicted class AND
     the probe's prediction from a feature extracted from THAT SAME
     predicted box — so both methods see one identical crop.
  5. Gold objects the detector never boxed are reported separately, never
     folded into either accuracy: a detection miss is not a classification
     error, and charging it to the classifier would blame the wrong stage.

Step 4 is what makes the comparison honest. Scoring the probe on oracle
crops while the class head sees its own noisier boxes would credit the
probe for the detector's localisation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from affdet.detect.yolo_boxes import _find_scene_jsons, match_boxes
from affdet.eval.classification import score_classification
from affdet.experiment.component_config import ComponentEvalConfig
from affdet.experiment.component_runner import (
    Predictor, _dataset_val_stems, _scenes_hash)
from affdet.experiment.ledger import append_run, git_sha
from affdet.model.dataset import scene_split_map
from affdet.schemas import Scene

#: crops -> feature matrix, one row per crop. Injected so the path is
#: testable without a GPU; production builds a DINOv2 extractor.
FeatureFn = Callable[[list[dict[str, Any]]], np.ndarray]

#: Builds the (unfitted) probe. Injected like FeatureFn so the wiring is
#: testable without scikit-learn, which is an optional dependency.
ProbeFactory = Callable[[], Any]


def _default_probe() -> Any:
    """Multinomial logistic regression on frozen DINOv2 CLS features — the
    "linear probe" of the original diagnostic, kept identical so the ported
    result is comparable to the one it replaces."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=1.0)


def _crop_records(scene: Scene, image_path: Path,
                  boxes: dict[str, list[float]]) -> list[dict[str, Any]]:
    by_id = {o.obj_id: o for o in scene.objects}
    return [
        {"obj_id": oid, "true_cls": by_id[oid].obj_class,
         "bbox": bbox, "image_path": image_path}
        for oid, bbox in boxes.items() if oid in by_id
    ]


def _dinov2_feature_fn(cfg: ComponentEvalConfig) -> FeatureFn:
    """Real DINOv2 CLS features, cropped exactly as the shipped extractor
    crops (resize 224, ImageNet normalise) so the probe sees the same
    representation it would at inference."""
    import torch
    from PIL import Image

    from affdet.extract.dinov2 import _build_backbone, _preprocess

    backbone = _build_backbone("vitb14")
    device = next(backbone.parameters()).device

    def extract(crops: list[dict[str, Any]]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), dtype=np.float32)
        tensors = []
        for c in crops:
            img = Image.open(c["image_path"]).convert("RGB")
            tensors.append(_preprocess(img.crop(tuple(c["bbox"]))))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = backbone.forward_features(batch)["x_norm_clstoken"]
        return feats.cpu().float().numpy()

    return extract


def _extract_batched(extract: "FeatureFn", crops: list[dict[str, Any]],
                     batch: int) -> np.ndarray:
    """Feature-extract in bounded chunks and concatenate.

    Kept out of the feature_fn itself so an injected test double still sees
    the batching (the defect this guards against was in the caller, not the
    extractor).
    """
    if not crops:
        return extract(crops)
    size = max(1, int(batch))
    parts = [extract(crops[i:i + size]) for i in range(0, len(crops), size)]
    return np.concatenate(parts, axis=0)


def run_classification_eval(
    cfg: ComponentEvalConfig,
    root: Path | str,
    detector_factory: Callable[[ComponentEvalConfig], Predictor] | None = None,
    feature_fn: FeatureFn | None = None,
    probe_factory: ProbeFactory | None = None,
) -> dict:
    """Fit the probe on train scenes, compare both classifiers on held-out."""
    root = Path(root)
    scenes_dir = Path(cfg.scenes)
    scene_jsons = _find_scene_jsons(scenes_dir)
    if not scene_jsons:
        raise ValueError(f"No scenes found under {scenes_dir}")

    if cfg.split_from_dataset:
        # The instance-disjoint arm's test set is defined by MESH IDENTITY at
        # dataset-build time; a seeded scene shuffle does not reproduce it and
        # would fit the probe on scenes holding the very meshes the arm
        # reserves for testing. Train is the complement of that val split, so
        # the two cannot overlap by construction.
        if cfg.data_yaml is None:
            raise ValueError(
                f"arm {cfg.name!r} sets split_from_dataset but has no "
                "data_yaml to read the split from")
        val_stems = _dataset_val_stems(Path(cfg.data_yaml))
        eval_jsons = [p for p in scene_jsons if p.stem in val_stems]
        train_jsons = [p for p in scene_jsons if p.stem not in val_stems]
    else:
        split_map = scene_split_map(scenes_dir, seed=cfg.seed)
        train_jsons = [p for p in scene_jsons
                       if split_map.get(p.stem) == "train"]
        eval_jsons = [p for p in scene_jsons
                      if split_map.get(p.stem) == cfg.split]

    if cfg.max_scenes is not None:
        eval_jsons = eval_jsons[:cfg.max_scenes]
    if cfg.max_train_scenes is not None:
        train_jsons = train_jsons[:cfg.max_train_scenes]

    predictor = (detector_factory or _build_detector)(cfg)
    extract = feature_fn or _dinov2_feature_fn(cfg)

    # --- fit the probe on ORACLE crops from the train split ---------------
    train_crops: list[dict[str, Any]] = []
    for sj in train_jsons:
        scene = Scene.model_validate_json(sj.read_text())
        image_path = (sj.parent / scene.image_path).resolve()
        train_crops.extend(
            {"obj_id": o.obj_id, "true_cls": o.obj_class,
             "bbox": list(o.bbox), "image_path": image_path}
            for o in scene.objects
        )
    # Batched: extracting all ~43k train crops in one call stacks ~26 GB of
    # float32 before the backbone runs, which OOM-killed every seed. The
    # eval loop below is already bounded (one scene, ~4 crops, at a time).
    X_train = _extract_batched(extract, train_crops, cfg.feature_batch)
    y_train = [c["true_cls"] for c in train_crops]
    probe = (probe_factory or _default_probe)()
    probe.fit(X_train, y_train)

    # --- score both classifiers on the SAME detector-predicted crops ------
    true_cls: list[str] = []
    probe_cls: list[str | None] = []
    head_cls: list[str | None] = []
    scored_ids: list[str] = []
    n_gold = 0

    for sj in eval_jsons:
        scene = Scene.model_validate_json(sj.read_text())
        n_gold += len(scene.objects)
        image_path = (sj.parent / scene.image_path).resolve()
        preds = predictor(image_path, scene)
        if not preds:
            continue

        # Geometric matching only: the crop is defined by the box, and the
        # detector's naming is one of the things under test, so it must not
        # gate which crops get scored.
        matched = match_boxes(scene.objects, preds, iou_thresh=cfg.iou_thresh)
        if not matched:
            continue

        crops = _crop_records(scene, image_path, matched)
        if not crops:
            continue

        head_by_box = {tuple(p["bbox"]): p.get("cls") for p in preds}
        feats = extract(crops)
        preds_probe = probe.predict(feats)

        for crop, p_cls in zip(crops, preds_probe):
            true_cls.append(crop["true_cls"])
            probe_cls.append(str(p_cls))
            head_cls.append(head_by_box.get(tuple(crop["bbox"])))
            scored_ids.append(crop["obj_id"])

    classification = score_classification(true_cls, probe_cls, head_cls)

    novel_ids: set[str] = set()
    if cfg.split_novel_meshes:
        # The detection arm showed the class head losing 0.98 -> 0.67 on an
        # unseen mesh. Whether the probe degrades the same way is the actual
        # case for re-classifying every crop, and a pooled accuracy hides it.
        #
        # "Novel" is derived from what the PROBE was fitted on, not from the
        # dataset's train directory as in the detection arm. The two coincide
        # only when train+val cover the corpus; where they do not, a scene in
        # neither directory still trains the probe, and calling its meshes
        # novel would overstate exactly the generalisation under test.
        novel_ids = {oid for oid in set(scored_ids)
                     if oid not in {c["obj_id"] for c in train_crops}}
        idx_novel = [i for i, oid in enumerate(scored_ids) if oid in novel_ids]
        idx_rest = [i for i, oid in enumerate(scored_ids)
                    if oid not in novel_ids]
        classification["subgroups"] = {
            name: score_classification(
                [true_cls[i] for i in idx],
                [probe_cls[i] for i in idx],
                [head_cls[i] for i in idx])
            for name, idx in (("novel", idx_novel), ("rest", idx_rest))
        }

    run_id = f"{cfg.name}-cls-{cfg.arm_hash()}-{git_sha()[:7]}"
    result = {
        "run_id": run_id,
        "kind": "classification",
        "git_sha": git_sha(),
        "config": cfg.model_dump(),
        "arm_hash": cfg.arm_hash(),
        "split": cfg.split,
        # Under split_from_dataset the seed feeds nothing: the split comes
        # from the built dataset, and the probe (LogisticRegression/lbfgs on
        # fixed data) is deterministic. Recorded explicitly so repeated
        # "seeds" are not mistaken for independent samples and given error
        # bars they have not earned.
        "seed_affects_result": not cfg.split_from_dataset,
        "novel_mesh_ids": sorted(novel_ids) if cfg.split_novel_meshes else [],
        "n_train_scenes": len(train_jsons),
        "n_eval_scenes": len(eval_jsons),
        "n_gold_objects": n_gold,
        "n_matched_objects": len(true_cls),
        # A detection miss is not a classification error; reported apart so
        # it cannot silently deflate either classifier's accuracy.
        "n_detector_misses": n_gold - len(true_cls),
        "train_scene_ids": [p.stem for p in train_jsons],
        "eval_scene_ids": [p.stem for p in eval_jsons],
        "scenes_hash": _scenes_hash(eval_jsons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
    }

    record = {k: v for k, v in result.items()
              if k not in ("train_scene_ids", "eval_scene_ids",
                           "classification")}
    record["metrics"] = {"classification": classification}
    append_run(root / "runs.jsonl", record)
    return result


def _build_detector(cfg: ComponentEvalConfig) -> Predictor:
    from affdet.detect.predictors import build_predictor
    return build_predictor(cfg)
