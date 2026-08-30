"""MockRenderer — produces schema-valid fake scenes with no real Blender calls.

Writes 1px gray PNGs (just to satisfy image_path existence) and full scene
metadata with random-but-plausible bboxes and complete gt_attrs. Used by
Phase 0 to prove the pipeline plumbing before any real renderer exists.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

from affdet.schemas import Scene, SceneObject, GTAttrs, Camera
from affdet.intersection_matrix import OBJECT_CLASSES


_CLASS_GT_TEMPLATES: dict[str, dict] = {
    "knife":        {"members": ["body", "handle", "edge"], "bbox_size": [0.20, 0.03, 0.02], "weight_kg": 0.15, "edge_type": "serrated"},
    "banana":       {"members": ["body"], "bbox_size": [0.18, 0.04, 0.04], "weight_kg": 0.12, "edge_type": "none"},
    "bowl":         {"members": ["body"], "bbox_size": [0.16, 0.08, 0.16], "weight_kg": 0.30, "edge_type": "none"},
    "cutting_board":{"members": ["body"], "bbox_size": [0.30, 0.02, 0.20], "weight_kg": 0.50, "edge_type": "none"},
    "spoon":        {"members": ["body", "handle"], "bbox_size": [0.18, 0.03, 0.04], "weight_kg": 0.05, "edge_type": "none"},
    "skillet":      {"members": ["body", "handle"], "bbox_size": [0.28, 0.06, 0.28], "weight_kg": 0.80, "edge_type": "none"},
    "mug":          {"members": ["body", "handle"], "bbox_size": [0.09, 0.10, 0.09], "weight_kg": 0.35, "edge_type": "none"},
    "plate":        {"members": ["body"], "bbox_size": [0.26, 0.03, 0.26], "weight_kg": 0.45, "edge_type": "none"},
    "tray":         {"members": ["body"], "bbox_size": [0.35, 0.03, 0.25], "weight_kg": 0.40, "edge_type": "none"},
}


def _default_gt(obj_class: str) -> GTAttrs:
    template = _CLASS_GT_TEMPLATES.get(
        obj_class,
        {"members": ["body"], "bbox_size": [0.10, 0.05, 0.05], "weight_kg": 0.10, "edge_type": "none"},
    )
    return GTAttrs(**template)


@dataclass
class MockRenderer:
    seed: int = 0
    n_scenes: int = 10
    objs_per_scene: Tuple[int, int] = (3, 6)
    image_size: Tuple[int, int] = (128, 96)

    def run(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(self.seed)

        for i in range(1, self.n_scenes + 1):
            scene_id = f"scene_{i:04d}"
            png_path = out_dir / f"{scene_id}.png"
            Image.new("RGB", self.image_size, color=(128, 128, 128)).save(png_path)

            n_objs = int(rng.randint(self.objs_per_scene[0], self.objs_per_scene[1] + 1))
            chosen_classes = list(rng.choice(OBJECT_CLASSES, size=n_objs, replace=True))
            objects: list[SceneObject] = []
            for j, cls in enumerate(chosen_classes):
                W, H = self.image_size
                x1 = float(rng.randint(0, W // 2))
                y1 = float(rng.randint(0, H // 2))
                x2 = float(min(W - 1, x1 + rng.randint(20, W // 2)))
                y2 = float(min(H - 1, y1 + rng.randint(20, H // 2)))
                obj_id = f"{cls}_{j}"
                objects.append(SceneObject(
                    obj_id=obj_id,
                    obj_class=cls,
                    bbox=[x1, y1, x2, y2],
                    mask_path=f"{scene_id}_{obj_id}_mask.png",
                    gt_attrs=_default_gt(cls),
                ))

            W, H = self.image_size
            scene = Scene(
                scene_id=scene_id,
                image_path=f"{scene_id}.png",
                image_size=list(self.image_size),
                source="blender",
                objects=objects,
                camera=Camera(
                    pose=[[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.5],
                          [0.0, 0.0, 0.0, 1.0]],
                    intrinsics=[[500.0, 0.0, W / 2],
                                [0.0, 500.0, H / 2],
                                [0.0, 0.0, 1.0]],
                ),
            )
            (out_dir / f"{scene_id}.json").write_text(scene.model_dump_json(indent=2))
