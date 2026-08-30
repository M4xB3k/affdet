"""Thin wrapper around Ultralytics detector training.

Ultralytics handles the whole train/val loop; this module only pins the
settings and keeps the heavy dependency out of the rest of affdet (imported
lazily, only when no factory is injected). Swapping the weights string
("yolo11s.pt" / "rtdetr-l.pt" / "yolov8s-worldv2.pt") is the entire
model-architecture decision.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable


def _ultralytics_factory(weights: str) -> Any:
    try:
        from ultralytics import RTDETR, YOLO, YOLOWorld
    except ImportError as e:
        raise ImportError(
            "ultralytics is not installed — pip install 'affdet[detect]' "
            "(or pip install ultralytics)"
        ) from e
    weights_lower = weights.lower()
    if "rtdetr" in weights_lower:
        return RTDETR(weights)
    if "world" in weights_lower:
        return YOLOWorld(weights)
    return YOLO(weights)


def train_detector(
    data_yaml: Path | str,
    model: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    seed: int = 0,
    project: Path | str | None = None,
    name: str | None = None,
    device: str | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> Any:
    """Fine-tune a detector on a converted dataset. Returns Ultralytics results."""
    factory = model_factory or _ultralytics_factory
    detector = factory(model)

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "seed": seed,
    }
    if project is not None:
        kwargs["project"] = str(project)
    if name is not None:
        kwargs["name"] = name
    if device is not None:
        kwargs["device"] = device
    return detector.train(**kwargs)
