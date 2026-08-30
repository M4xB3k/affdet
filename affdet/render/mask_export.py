from __future__ import annotations

import numpy as np


def mask_to_bbox(mask: np.ndarray, padding: float = 0.0,
                 img_size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """Binary mask → [x1, y1, x2, y2] with optional % padding."""
    if mask.ndim != 2:
        raise ValueError("mask_to_bbox expects a 2D array.")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("mask_to_bbox received an empty mask.")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if padding > 0:
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        pad_x = int(np.ceil(width * padding))
        pad_y = int(np.ceil(height * padding))
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y
        if img_size is None:
            img_size = (mask.shape[1], mask.shape[0])
        max_x = img_size[0] - 1
        max_y = img_size[1] - 1
        x1 = max(0, min(max_x, x1))
        x2 = max(0, min(max_x, x2))
        y1 = max(0, min(max_y, y1))
        y2 = max(0, min(max_y, y2))
    return x1, y1, x2, y2


def extract_bboxes_from_pass(object_index_pass: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    """Blender object-index pass (2D array of int IDs) → {obj_id: (x1,y1,x2,y2)}."""
    if object_index_pass.ndim != 2:
        raise ValueError("extract_bboxes_from_pass expects a 2D array.")
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    obj_ids = np.unique(object_index_pass)
    for obj_id in obj_ids:
        if obj_id == 0:
            continue
        mask = object_index_pass == obj_id
        bboxes[int(obj_id)] = mask_to_bbox(mask, img_size=(
            object_index_pass.shape[1], object_index_pass.shape[0]
        ))
    return bboxes
