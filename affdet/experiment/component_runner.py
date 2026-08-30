"""run_component_eval — score one detector arm of E5 and ledger the result.

Sits alongside ``run_experiment`` rather than inside it: that function
sequences extract -> train -> infer -> eval for the affordance model, and
E5 trains nothing. Both append to the same ``runs.jsonl``, and E5 rows carry
``kind: "component"`` so the two never merge when results are aggregated.

The detector is injected (as with ``DINOv2Extractor``'s backbone and
``yolo_boxes``' predictor) so the whole path is unit-testable without
weights, a GPU, or the render corpus.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from affdet.detect.yolo_boxes import _find_scene_jsons
from affdet.eval.detection import aggregate_detection_scores, score_detections
from affdet.experiment.component_config import ComponentEvalConfig
from affdet.experiment.ledger import append_run, git_sha
from affdet.model.dataset import scene_split_map
from affdet.schemas import Scene

#: (image_path, scene) -> [{"bbox": [x1,y1,x2,y2], "cls": str, "conf": float}]
Predictor = Callable[[Path, Scene], list[dict[str, Any]]]
DetectorFactory = Callable[[ComponentEvalConfig], Predictor]


def _scenes_hash(scene_jsons: list[Path]) -> str:
    """Content hash of the scored scene set, so a run states exactly which
    scenes produced its number (the reproducibility gap that made an earlier
    headline result uncitable)."""
    h = hashlib.sha256()
    for p in sorted(scene_jsons):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _dataset_val_stems(data_yaml: Path) -> set[str]:
    """Scene ids in a built dataset's val split."""
    import yaml

    spec = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(spec.get("path", Path(data_yaml).parent))
    val_dir = base / spec.get("val", "images/val")
    return {p.stem for p in val_dir.iterdir() if p.is_file()}


def _select_scenes(cfg: ComponentEvalConfig) -> list[Path]:
    """Which scenes to score.

    By default, the held-out portion of the seeded scene split — the same
    machinery the affordance experiments use, so "test" means the same
    scenes in both chapters.

    Under ``split_from_dataset`` the eval set is read from the built
    dataset's own val split instead. The instance-disjoint arm REQUIRES
    this: its test set is defined by mesh identity at build time, and a
    seeded shuffle would hand back scenes the detector trained on, which is
    exactly the contamination that arm exists to rule out.
    """
    scenes_dir = Path(cfg.scenes)
    scene_jsons = _find_scene_jsons(scenes_dir)
    if not scene_jsons:
        raise ValueError(f"No scenes found under {scenes_dir}")

    if cfg.split_from_dataset:
        if cfg.data_yaml is None:
            raise ValueError(
                f"arm {cfg.name!r} sets split_from_dataset but has no "
                "data_yaml to read the split from")
        keep = _dataset_val_stems(Path(cfg.data_yaml))
        selected = [p for p in scene_jsons if p.stem in keep]
    else:
        split_map = scene_split_map(scenes_dir, seed=cfg.seed)
        selected = [p for p in scene_jsons if split_map.get(p.stem) == cfg.split]

    if cfg.max_scenes is not None:
        selected = selected[:cfg.max_scenes]
    return selected


def _dataset_train_stems(data_yaml: Path) -> set[str]:
    """Scene ids in a built dataset's train split."""
    import yaml

    spec = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(spec.get("path", Path(data_yaml).parent))
    train_dir = base / spec.get("train", "images/train")
    return {p.stem for p in train_dir.iterdir() if p.is_file()}


def _novel_mesh_ids(cfg: ComponentEvalConfig) -> set[str]:
    """Gold obj_ids in the eval split whose mesh never appears in training.

    The instance-disjoint split reserves MESHES but moves whole SCENES, and a
    scene holds several objects — so most objects in the eval split are
    ordinary training meshes that merely share a scene with a reserved one.
    Only these ids answer "how does it do on an instance it has never seen".
    """
    if cfg.data_yaml is None:
        raise ValueError(
            f"arm {cfg.name!r} sets split_novel_meshes but has no data_yaml "
            "to read the training split from")
    scenes_dir = Path(cfg.scenes)
    train_stems = _dataset_train_stems(Path(cfg.data_yaml))

    seen: set[str] = set()
    for p in _find_scene_jsons(scenes_dir):
        if p.stem in train_stems:
            for o in Scene.model_validate_json(p.read_text()).objects:
                seen.add(o.obj_id)

    novel: set[str] = set()
    for p in _select_scenes(cfg):
        for o in Scene.model_validate_json(p.read_text()).objects:
            if o.obj_id not in seen:
                novel.add(o.obj_id)
    return novel


def _build_detector(cfg: ComponentEvalConfig) -> Predictor:
    """Real detector, loaded lazily — heavy deps, and never imported when a
    test injects its own."""
    from affdet.detect.predictors import build_predictor
    return build_predictor(cfg)


def run_component_eval(
    cfg: ComponentEvalConfig,
    root: Path | str,
    detector_factory: DetectorFactory | None = None,
) -> dict:
    """Score one detector over the held-out split; append to the ledger.

    Detection is reported twice: class-aware (did it find the object and name
    it right) and class-agnostic (were the boxes right, regardless of name).
    The second is the number that actually governs this pipeline, which
    re-classifies every crop with the DINOv2 probe and discards detector
    labels — so reporting only the first would understate what the detector
    contributes here, and only the second would hide a real weakness.
    """
    root = Path(root)
    scene_jsons = _select_scenes(cfg)
    predictor = (detector_factory or _build_detector)(cfg)

    subgroups = ({"novel": _novel_mesh_ids(cfg)}
                 if cfg.split_novel_meshes else None)

    per_scene: list[dict] = []
    per_scene_agnostic: list[dict] = []
    for scene_json in scene_jsons:
        scene = Scene.model_validate_json(scene_json.read_text())
        image_path = (scene_json.parent / scene.image_path).resolve()
        preds = predictor(image_path, scene)
        per_scene.append(score_detections(
            scene.objects, preds, iou_thresh=cfg.iou_thresh,
            subgroups=subgroups))
        per_scene_agnostic.append(score_detections(
            scene.objects, preds, iou_thresh=cfg.iou_thresh,
            class_agnostic=True, subgroups=subgroups))

    detection = aggregate_detection_scores(per_scene)
    detection_agnostic = aggregate_detection_scores(per_scene_agnostic)

    run_id = f"{cfg.name}-{cfg.arm_hash()}-{git_sha()[:7]}"
    result = {
        "run_id": run_id,
        "kind": "component",
        "git_sha": git_sha(),
        "config": cfg.model_dump(),
        "arm_hash": cfg.arm_hash(),
        "split": cfg.split,
        "n_scenes_scored": len(scene_jsons),
        "scene_ids": [p.stem for p in scene_jsons],
        "scenes_hash": _scenes_hash(scene_jsons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detection": detection,
        "detection_class_agnostic": detection_agnostic,
    }

    record = {k: v for k, v in result.items() if k != "scene_ids"}
    record["metrics"] = {
        "detection": detection,
        "detection_class_agnostic": detection_agnostic,
    }
    del record["detection"], record["detection_class_agnostic"]
    append_run(root / "runs.jsonl", record)
    return result
