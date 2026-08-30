"""Scene JSONs → Ultralytics detection dataset (images/, labels/, data.yaml).

The render pipeline already writes a pixel bbox + obj_class per object into
each Scene JSON, so YOLO training labels are a pure reformatting job:

    out/
    ├── images/{train,val}/<scene_id>.png    (symlinks into scenes_dir)
    ├── labels/{train,val}/<scene_id>.txt    (one "cls cx cy w h" line per obj)
    └── data.yaml

Class ids are indices into the sorted list of classes found across all
converted scenes; `allowed_classes` (e.g. the curated category set) makes
unknown classes a hard error instead of a silent new class.
"""
from __future__ import annotations
import random
from pathlib import Path

import yaml

from affdet.schemas import Scene


def _find_scene_jsons(scenes_dir: Path) -> list[Path]:
    """Flat, nested (scene_X/scene_X.json), or cluster worker_N/ layouts."""
    for pattern in ("scene_*.json", "scene_*/scene_*.json",
                    "worker_*/scene_*/scene_*.json"):
        found = sorted(scenes_dir.glob(pattern))
        if found:
            return found
    return []


def _is_trainable(obj) -> bool:
    """Objects the renderer marked invisible are not detection targets.

    The render pipeline sets ``visible=False`` when an object occupies less
    than a minimum fraction of the frame; such an object has essentially no
    pixels to detect, and one of them in the corpus (``sweet_potato_013`` in
    ``scene_v3_00888``, 2 visible px) has a zero-width bbox as a result.
    Training on it is meaningless and validating it aborts the conversion of
    12,753 otherwise-good scenes, so it is skipped at both points.
    """
    return getattr(obj, "visible", True)


def _validate_scene(scene: Scene, scene_json: Path,
                    allowed_classes: set[str] | None) -> None:
    if not scene.objects:
        raise ValueError(f"{scene_json}: scene has no objects")
    width, height = scene.image_size
    for obj in scene.objects:
        if not _is_trainable(obj):
            continue
        x1, y1, x2, y2 = obj.bbox
        if allowed_classes is not None and obj.obj_class not in allowed_classes:
            raise ValueError(
                f"{scene_json}: class '{obj.obj_class}' not in allowed classes")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{scene_json}: degenerate bbox {obj.bbox} on {obj.obj_id}")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(
                f"{scene_json}: bbox {obj.bbox} on {obj.obj_id} outside image "
                f"bounds {width}x{height}")


def _label_lines(scene: Scene, class_ids: dict[str, int]) -> list[str]:
    width, height = scene.image_size
    lines = []
    for obj in scene.objects:
        if not _is_trainable(obj):
            continue
        x1, y1, x2, y2 = obj.bbox
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{class_ids[obj.obj_class]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert(
    scenes_dir: Path | str,
    out_dir: Path | str,
    val_fraction: float = 0.1,
    seed: int = 0,
    allowed_classes: set[str] | None = None,
    exclude_scene_ids: set[str] | None = None,
) -> Path:
    """Convert a scenes directory to an Ultralytics dataset. Returns data.yaml path."""
    scenes_dir = Path(scenes_dir)
    out_dir = Path(out_dir)

    scene_jsons = _find_scene_jsons(scenes_dir)
    if exclude_scene_ids:
        scene_jsons = [p for p in scene_jsons if p.stem not in exclude_scene_ids]
    if not scene_jsons:
        raise ValueError(f"No scenes found under {scenes_dir}")

    scenes = [Scene.model_validate_json(p.read_text()) for p in scene_jsons]
    for scene, scene_json in zip(scenes, scene_jsons):
        _validate_scene(scene, scene_json, allowed_classes)

    names = sorted({o.obj_class for s in scenes for o in s.objects})
    class_ids = {name: i for i, name in enumerate(names)}

    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    n_val = round(len(scenes) * val_fraction)
    val_idx = set(order[:n_val])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, (scene, scene_json) in enumerate(zip(scenes, scene_jsons)):
        split = "val" if i in val_idx else "train"
        image_path = (scene_json.parent / scene.image_path).resolve()
        link = out_dir / "images" / split / image_path.name
        if not link.is_symlink():
            link.symlink_to(image_path)
        (out_dir / "labels" / split / f"{scene_json.stem}.txt").write_text(
            "\n".join(_label_lines(scene, class_ids)) + "\n")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }, sort_keys=False))
    return data_yaml
