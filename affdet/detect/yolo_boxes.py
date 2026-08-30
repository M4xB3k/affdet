"""Rewrite Scene bboxes with IoU-matched YOLO predictions — the "YOLO arm".

For the detector-influence ablation (does detector localisation error propagate
into affordance accuracy?), we need two crop sources over the *same* scenes:

  * oracle  — the gold render-metadata bbox on each SceneObject
  * yolo    — the trained detector's predicted box for that same object

This module builds the yolo arm. Each gold object keeps its ``obj_id`` and
``obj_class`` (so the affordance labels/pairs stay aligned one-to-one) but its
``bbox`` is replaced by the best IoU-matched YOLO prediction. Gold objects the
detector missed (no prediction above the IoU threshold) are *dropped* — there is
no crop to feed downstream, mirroring the deployed pipeline. YOLO false
positives (predictions matching no gold object) are ignored: there is no gold
``obj_id`` to attach a label to.

Matching is purely geometric (IoU), independent of the predicted class — the
ablation isolates the effect of the *box*, not the detector's naming, and the
gold class is retained regardless.

The YOLO predictor is injected (like DINOv2Extractor's backbone) so the matching
logic is unit-testable without loading a network. ``_ultralytics_predictor``
builds the real one lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from affdet.schemas import Scene, SceneObject

# A predictor takes (image_path, scene) and returns a list of predictions,
# each a dict {"bbox": [x1, y1, x2, y2], "cls": <class name>}.
Predictor = Callable[[Path, Scene], list[dict[str, Any]]]


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def match_boxes(
    gold: list[SceneObject],
    preds: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> dict[str, list[float]]:
    """Greedily match each gold object to at most one predicted box by IoU.

    Returns ``{obj_id: pred_bbox}`` for gold objects whose best-matching
    prediction has IoU > ``iou_thresh``. Each prediction is consumed at most once
    (highest-IoU gold/pred pairs first), so two gold objects cannot claim the
    same box. Unmatched gold objects are simply absent from the result.
    """
    # All (iou, gold_idx, pred_idx) candidate pairs above threshold, best first.
    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(preds):
            score = iou(g.bbox, p["bbox"])
            if score > iou_thresh:
                candidates.append((score, gi, pi))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched: dict[str, list[float]] = {}
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for _score, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        matched[gold[gi].obj_id] = list(preds[pi]["bbox"])
        used_gold.add(gi)
        used_pred.add(pi)
    return matched


def rewrite_scene(
    scene: Scene,
    preds: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> Scene:
    """Return a copy of ``scene`` with each object's bbox replaced by its matched
    YOLO box; objects with no match above threshold are dropped."""
    matched = match_boxes(scene.objects, preds, iou_thresh=iou_thresh)
    new_objects: list[SceneObject] = []
    for o in scene.objects:
        if o.obj_id not in matched:
            continue
        new_objects.append(o.model_copy(update={"bbox": matched[o.obj_id]}))
    return scene.model_copy(update={"objects": new_objects})


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _containment_frac(small: list[float], big: list[float]) -> float:
    """Fraction of ``small``'s own area that overlaps ``big``."""
    sx1, sy1, sx2, sy2 = small
    bx1, by1, bx2, by2 = big
    ix1, iy1 = max(sx1, bx1), max(sy1, by1)
    ix2, iy2 = min(sx2, bx2), min(sy2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = _box_area(small)
    return inter / area if area > 0 else 0.0


def dedup_contained_boxes(
    preds: list[dict[str, Any]],
    containment_thresh: float = 0.85,
    max_container_image_frac: float = 0.5,
    image_size: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Drop predictions that are sub-parts of a bigger prediction.

    Standard IoU-based NMS misses a box that sits almost entirely inside a
    much larger one (e.g. a knife blade's reflective tip detected as its own
    "object" inside the knife's own bounding box): IoU stays low because the
    union is dominated by the big box's area, so NMS never suppresses it.
    This checks CONTAINMENT instead — what fraction of the smaller box's own
    area overlaps the bigger one — and drops the smaller box once that
    fraction clears ``containment_thresh``.

    A box larger than ``max_container_image_frac`` of the image is never
    treated as a legitimate "container" (only as a possible sub-part victim
    itself): a single spurious whole-scene detection would otherwise absorb
    every real, smaller detection inside it, which is backwards — it's the
    giant box that's spurious, not everything inside it. ``image_size``
    defaults to the bounding extent of all predicted boxes when not given.
    """
    if not preds:
        return []

    image_area = image_size[0] * image_size[1] if image_size is not None else None

    indexed = sorted(
        enumerate(preds), key=lambda t: _box_area(t[1]["bbox"]), reverse=True,
    )

    kept: list[dict[str, Any]] = []
    dropped: set[int] = set()
    for i, pred in indexed:
        is_subpart = False
        for kept_pred in kept:
            if image_area:
                container_frac_of_image = _box_area(kept_pred["bbox"]) / image_area
                if container_frac_of_image > max_container_image_frac:
                    continue
            if _containment_frac(pred["bbox"], kept_pred["bbox"]) >= containment_thresh:
                is_subpart = True
                break
        if is_subpart:
            dropped.add(i)
        else:
            kept.append(pred)

    return [pred for i, pred in enumerate(preds) if i not in dropped]


def _find_scene_jsons(scenes_dir: Path) -> list[Path]:
    """Flat, nested (scene_X/scene_X.json), or cluster worker_N/ layouts."""
    for pattern in ("scene_*.json", "scene_*/scene_*.json",
                    "worker_*/scene_*/scene_*.json"):
        found = sorted(scenes_dir.glob(pattern))
        if found:
            return found
    return []


def _link_or_copy_image(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def rewrite_scenes(
    scenes_dir: Path | str,
    out_dir: Path | str,
    predictor: Predictor,
    iou_thresh: float = 0.5,
) -> dict[str, int]:
    """Write a parallel scene set with YOLO-matched boxes. Returns match stats.

    The output is a flat directory of ``<scene_id>.json`` files (bboxes swapped,
    missed objects dropped) with each scene's image linked alongside, so the
    existing extractor (which reads ``scene.image_path`` relative to the JSON's
    parent) works unchanged.
    """
    scenes_dir = Path(scenes_dir)
    out_dir = Path(out_dir)
    scene_jsons = _find_scene_jsons(scenes_dir)
    if not scene_jsons:
        raise ValueError(f"No scenes found under {scenes_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gold = n_matched = 0
    for scene_json in scene_jsons:
        scene = Scene.model_validate_json(scene_json.read_text())
        image_src = (scene_json.parent / scene.image_path).resolve()
        preds = predictor(image_src, scene)

        rewritten = rewrite_scene(scene, preds, iou_thresh=iou_thresh)
        n_gold += len(scene.objects)
        n_matched += len(rewritten.objects)

        # Keep image_path a bare filename so it resolves next to the new JSON.
        image_name = Path(scene.image_path).name
        rewritten = rewritten.model_copy(update={"image_path": image_name})
        (out_dir / f"{scene.scene_id}.json").write_text(
            rewritten.model_dump_json())
        _link_or_copy_image(image_src, out_dir / image_name)

    return {
        "n_scenes": len(scene_jsons),
        "n_gold_objects": n_gold,
        "n_matched": n_matched,
        "n_dropped": n_gold - n_matched,
    }


def _ultralytics_predictor(weights: str, conf: float = 0.05) -> Predictor:
    """Build a real YOLO-backed predictor. Loaded lazily; heavy dep.

    ``conf`` defaults low (0.05, not Ultralytics' 0.25) deliberately: a
    missed detection can never be recovered downstream, while a spurious one
    can — either by a confidence/size filter or by a second-stage classifier
    rejecting the pair it forms. Runs with ``agnostic_nms=True`` (Ultralytics'
    default NMS only suppresses duplicates within the same predicted class,
    so two near-identical boxes with different class guesses on the same
    physical object both survive) and a containment-based dedup pass on top
    (catches a sub-part box sitting almost entirely inside a bigger one,
    which IoU-based NMS misses since IoU stays low when the union is
    dominated by the big box) — see ``dedup_contained_boxes``. Real-photo
    audit (19 hand-labeled kitchen photos, IoU>=0.5 box-only scoring):
    F1 78.4%->93.0% for the fine-tuned checkpoint with the fixed conf/NMS/
    dedup settings vs. the pre-fix defaults.
    """
    from ultralytics import YOLO, YOLOWorld

    try:
        model = YOLO(weights)
    except TypeError:
        # A fine-tuned YOLO-World checkpoint's own filename doesn't reliably
        # say "world" (the deployed checkpoint is plain best.pt), so unlike
        # affdet.detect.train._ultralytics_factory (which dispatches on the
        # weights string at TRAINING time, before a checkpoint exists), this
        # inference-time loader can't tell YOLO from YOLO-World up front.
        # Ultralytics' plain YOLO() raises TypeError on a YOLO-World
        # checkpoint ("references types outside the supported Ultralytics
        # checkpoint format") -- retry with YOLOWorld specifically on that.
        model = YOLOWorld(weights)

    def predict(image_path: Path, scene: Scene) -> list[dict[str, Any]]:
        res = model.predict(
            str(image_path), conf=conf, iou=0.5, agnostic_nms=True, verbose=False,
        )[0]
        out: list[dict[str, Any]] = []
        if res.boxes is None:
            return out
        for xyxy, cls_idx, box_conf in zip(
            res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()
        ):
            out.append({
                "bbox": list(xyxy), "cls": res.names[int(cls_idx)],
                "conf": box_conf,
            })
        image_w, image_h = res.orig_shape[1], res.orig_shape[0]
        return dedup_contained_boxes(out, image_size=(image_w, image_h))

    return predict
