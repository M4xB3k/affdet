"""Build a synthetic Scene directly from raw YOLO detections on an arbitrary
photo — no gold Scene JSON to match against (unlike yolo_boxes.rewrite_scene,
which needs pre-existing gold obj_ids from a render).

Camera pose/intrinsics are never read downstream (extract/model only use
bbox + image_size), so a placeholder identity Camera is used.
"""
from __future__ import annotations

from typing import Any

from affdet.schemas import Camera, Scene, SceneObject

_IDENTITY_CAMERA = Camera(
    pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
)


def scene_from_predictions(
    scene_id: str,
    image_path: str,
    image_size: tuple[int, int],
    preds: list[dict[str, Any]],
    conf_thresh: float = 0.25,
) -> Scene:
    """Turn raw detector predictions into a Scene with one SceneObject per
    kept prediction. Each prediction is a dict with "bbox" [x1,y1,x2,y2],
    "cls" (class name), and optionally "conf" (confidence, filtered against
    conf_thresh; predictions without "conf" are always kept)."""
    counts: dict[str, int] = {}
    objects: list[SceneObject] = []
    for p in preds:
        if p.get("conf", 1.0) < conf_thresh:
            continue
        cls = p["cls"]
        idx = counts.get(cls, 0)
        counts[cls] = idx + 1
        objects.append(SceneObject(
            obj_id=f"{cls}_{idx}", obj_class=cls, bbox=list(p["bbox"]),
        ))

    return Scene(
        scene_id=scene_id,
        image_path=image_path,
        image_size=list(image_size),
        source="real",
        objects=objects,
        camera=_IDENTITY_CAMERA,
    )
