"""Detector predictors for the E5 arms — one interface, four detectors.

The E5 table compares:

  * ``yolo11s``          fine-tuned on the render corpus
  * ``yolov8s-worldv2``  fine-tuned on the render corpus
  * ``yolov8s-worldv2``  ZERO-SHOT, taxonomy given as its vocabulary
  * ``grounding-dino``   ZERO-SHOT, taxonomy given as a text prompt

The middle pair is the controlled comparison: same architecture, same
pretrained weights, the only variable being whether it saw the renders. The
Grounding DINO row asks whether a stronger open-vocabulary detector closes
that gap.

Every arm predicts over the *same* 52-class vocabulary, so no arm gets an
easier label space than another. The zero-shot arms are handed a de-
underscored form of it ("red wine glass"), because they consume class names
as English text and the snake_case ID is not a phrase anyone would write;
their answers are mapped straight back to canonical IDs before scoring, so
all four arms are scored in one label space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from affdet.detect.yolo_boxes import dedup_contained_boxes
from affdet.eval.classification import canonical_from_prompt, prompt_vocabulary
from affdet.experiment.component_config import ComponentEvalConfig
from affdet.schemas import Scene

Predictor = Callable[[Path, Scene], list[dict[str, Any]]]

#: Detector names this module knows how to build.
ULTRALYTICS_DETECTORS = ("yolo11s", "yolov8s-worldv2", "yolov8s-world")
GROUNDING_DINO = "grounding-dino"
_GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def taxonomy_classes(data_yaml: Path | str) -> list[str]:
    """Class list, read from the same ``data.yaml`` the detectors were
    trained against rather than a hardcoded copy that could drift."""
    return list(yaml.safe_load(Path(data_yaml).read_text())["names"])


def normalise_predictions(
    preds: list[dict[str, Any]],
    classes: list[str],
) -> list[dict[str, Any]]:
    """Map emitted class text back into the taxonomy's own label space.

    A label outside the taxonomy becomes ``None`` rather than being dropped:
    the box was still predicted and still costs precision, so removing it
    would flatter the detector. ``None`` matches no gold class, which is the
    correct outcome.
    """
    return [{**p, "cls": canonical_from_prompt(p.get("cls"), classes)}
            for p in preds]


def _ultralytics_predictor(
    cfg: ComponentEvalConfig,
    model: Any,
    classes: list[str],
) -> Predictor:
    # A fine-tuned checkpoint already carries its learned class indices;
    # overriding them would discard the fine-tuning being measured.
    if cfg.zero_shot:
        model.set_classes(prompt_vocabulary(classes))

    def predict(image_path: Path, scene: Scene) -> list[dict[str, Any]]:
        res = model.predict(str(image_path), conf=cfg.conf, iou=0.5,
                            agnostic_nms=True, verbose=False)[0]
        out: list[dict[str, Any]] = []
        if res.boxes is None:
            return out
        for xyxy, cls_idx, conf in zip(res.boxes.xyxy.tolist(),
                                       res.boxes.cls.tolist(),
                                       res.boxes.conf.tolist()):
            out.append({"bbox": list(xyxy), "cls": res.names[int(cls_idx)],
                        "conf": conf})
        h, w = res.orig_shape[0], res.orig_shape[1]
        out = dedup_contained_boxes(out, image_size=(w, h))
        return normalise_predictions(out, classes)

    return predict


def _grounding_dino_predictor(
    cfg: ComponentEvalConfig,
    model: Any,
    processor: Any,
    classes: list[str],
) -> Predictor:
    import torch
    from PIL import Image

    # Grounding DINO expects a lowercased, period-separated phrase list.
    prompt = ". ".join(prompt_vocabulary(classes)) + "."
    device = next(model.parameters()).device

    def predict(image_path: Path, scene: Scene) -> list[dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=prompt,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=cfg.conf, text_threshold=cfg.conf,
            target_sizes=[image.size[::-1]],
        )[0]
        out = [
            {"bbox": [float(v) for v in box], "cls": label,
             "conf": float(score)}
            for box, label, score in zip(results["boxes"].tolist(),
                                         results["text_labels"],
                                         results["scores"].tolist())
        ]
        out = dedup_contained_boxes(out, image_size=image.size)
        return normalise_predictions(out, classes)

    return predict


def build_predictor(
    cfg: ComponentEvalConfig,
    model: Any | None = None,
    processor: Any | None = None,
    classes: list[str] | None = None,
) -> Predictor:
    """Build the predictor for one arm. ``model``/``processor``/``classes``
    are injected in tests; in production they are loaded lazily from the
    config."""
    # Validate the arm BEFORE loading anything: a typo'd detector name must
    # surface as itself, not as a downstream failure to read data.yaml.
    if cfg.detector not in ULTRALYTICS_DETECTORS + (GROUNDING_DINO,):
        raise ValueError(
            f"unknown detector {cfg.detector!r}; expected one of "
            f"{ULTRALYTICS_DETECTORS + (GROUNDING_DINO,)}")

    if classes is None:
        if cfg.data_yaml is None:
            raise ValueError(
                f"arm {cfg.name!r} has no data_yaml: the class vocabulary "
                "must come from the dataset the detectors were trained "
                "against, so every arm is scored in one label space")
        classes = taxonomy_classes(cfg.data_yaml)

    if cfg.detector in ULTRALYTICS_DETECTORS:
        if model is None:
            model = _load_ultralytics(cfg)
        return _ultralytics_predictor(cfg, model, classes)

    if cfg.detector == GROUNDING_DINO:
        if model is None or processor is None:
            model, processor = _load_grounding_dino()
        return _grounding_dino_predictor(cfg, model, processor, classes)

    raise AssertionError("unreachable: detector validated above")


def _load_ultralytics(cfg: ComponentEvalConfig) -> Any:
    from ultralytics import YOLO, YOLOWorld
    # Zero-shot arms load pretrained weights by name; fine-tuned arms load
    # their own checkpoint.
    weights = cfg.weights or f"{cfg.detector}.pt"
    if "world" in cfg.detector.lower():
        return YOLOWorld(weights)
    try:
        return YOLO(weights)
    except TypeError:
        # A fine-tuned YOLO-World checkpoint's filename need not say "world"
        # (the deployed one is plain best.pt); plain YOLO() raises TypeError
        # on it — see affdet.detect.yolo_boxes._ultralytics_predictor.
        return YOLOWorld(weights)


def _load_grounding_dino() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        _GROUNDING_DINO_MODEL_ID).to(device)
    model.eval()
    return model, processor
