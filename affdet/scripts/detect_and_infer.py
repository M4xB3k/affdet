"""End-to-end single-photo inference: raw image -> YOLO boxes -> synthetic
Scene -> DINOv2 crop features -> AffordanceMLP -> Predictions.

This is the one thing the staged pipeline (render/detect/extract/model, each
with its own CLI in affdet/scripts/) doesn't cover: there is no gold Scene for
an arbitrary photo, so YOLO detections themselves become the scene's objects
directly (affdet.detect.scene_from_predictions) instead of being matched
against gold boxes (affdet.detect.yolo_boxes, used for the ablation).

Every real step is delegated to its own already-tested module:
  - affdet.detect.yolo_boxes._ultralytics_predictor  (YOLO forward pass)
  - affdet.detect.scene_from_predictions              (raw preds -> Scene)
  - affdet.extract.dinov2.DINOv2Extractor             (crop -> DINOv2 features)
  - affdet.model.infer.Inferencer                     (features -> Predictions)
This file only wires them together; it holds no detection/model logic itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import typer
from PIL import Image

from affdet.detect.scene_from_predictions import scene_from_predictions
from affdet.detect.yolo_boxes import Predictor
from affdet.extract.dinov2 import DINOv2Extractor, _build_backbone, _preprocess
from affdet.extract.dinov2_classifier import DINOv2Classifier, relabel_predictions
from affdet.model.infer import Inferencer
from affdet.schemas import Predictions

app = typer.Typer(help="Single-photo inference: image -> YOLO -> DINOv2 -> AffordanceMLP.")

# DINOv2 classification is the default (2026-07-23): YOLO now only proposes
# class-agnostic boxes; DINOv2 assigns the class. Diagnostic justification:
# probe 98.76% vs YOLO's own class head 93.12% on identical predicted crops,
# full synthetic corpus (dinov2_probe_beats_yolo_classhead memory). Override
# with --dinov2-probe to point at a different probe, or --no-dinov2-probe to
# fall back to YOLO's own class head entirely.
from affdet import paths as _paths
DEFAULT_DINOV2_PROBE = _paths.CLASS_PROBE


def resolve_classifier_path(
    dinov2_probe: Path | None, no_dinov2_probe: bool, default_path: Path,
) -> Path | None:
    """--no-dinov2-probe always wins (explicit opt-out); otherwise an explicit
    --dinov2-probe path is used, and an unset one falls back to default_path
    (DINOv2 classification is the default, not opt-in)."""
    if no_dinov2_probe:
        return None
    return dinov2_probe if dinov2_probe is not None else default_path


def _relabel_with_classifier(
    image_path: Path,
    preds: list[dict[str, Any]],
    classifier: DINOv2Classifier,
    backbone: Any,
) -> list[dict[str, Any]]:
    """Override each prediction's "cls" with the DINOv2 probe's prediction from
    that same predicted box's own crop (not an oracle crop) -- apples-to-apples
    with how the probe was scored in the diagnostic that motivated this."""
    if not preds:
        return preds

    resolved_backbone = backbone if backbone is not None else _build_backbone()
    device = next(resolved_backbone.parameters()).device

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        crops = torch.stack([
            _preprocess(img.crop(tuple(p["bbox"]))) for p in preds
        ]).to(device)

    with torch.no_grad():
        features = resolved_backbone.forward_features(crops)["x_norm_clstoken"]
    features = features.cpu().float().numpy()

    return relabel_predictions(preds, features, classifier)


def run_single_image(
    image_path: Path,
    ckpt_dir: Path,
    out_dir: Path,
    predictor: Predictor,
    backbone: Any = None,
    classifier: DINOv2Classifier | None = None,
    conf_thresh: float = 0.05,
    resolved_instances: Path | str | None = None,
    all_pairs: bool = False,
) -> Predictions:
    """Run the full chain for one photo, writing intermediate + final
    artifacts under out_dir/{scenes,features,predictions} so each stage stays
    inspectable, and returning the parsed Predictions.

    scenes_dir and features_dir are kept as separate subdirectories (matching
    the convention every other stage CLI uses) rather than reusing out_dir for
    both: DINOv2Extractor writes a "<scene_id>.meta.json" sidecar into
    features_dir, and Scene discovery's glob ("scene_*.json") is not anchored
    past the wildcard, so it would also match that sidecar if scenes and
    features shared a directory.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    scenes_dir = out_dir / "scene"
    features_dir = out_dir / "features"
    preds_dir = out_dir / "_predictions"
    labels_dir = out_dir / "_no_labels"  # Inferencer.predict() never reads this
    for d in (scenes_dir, features_dir, preds_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    scene_image_path = scenes_dir / f"image{image_path.suffix}"
    if not scene_image_path.exists():
        scene_image_path.write_bytes(image_path.read_bytes())

    with Image.open(image_path) as img:
        image_size = img.size  # (W, H)

    preds = predictor(image_path, None)
    if classifier is not None:
        preds = _relabel_with_classifier(image_path, preds, classifier, backbone)
    # "scene_0001" matches the scene_*.json glob every downstream stage
    # (DINOv2Extractor, Inferencer) uses to discover scenes.
    scene_id = "scene_0001"
    scene = scene_from_predictions(
        scene_id=scene_id, image_path=scene_image_path.name,
        image_size=image_size, preds=preds, conf_thresh=conf_thresh,
    )
    (scenes_dir / f"{scene_id}.json").write_text(scene.model_dump_json())

    DINOv2Extractor(backbone=backbone).run(scenes_dir, features_dir)

    Inferencer(resolved_instances=resolved_instances, all_pairs=all_pairs).predict(
        scenes_dir, features_dir, labels_dir, ckpt_dir, preds_dir,
    )
    predictions_path = out_dir / "predictions.json"
    predictions_path.write_bytes((preds_dir / f"{scene_id}.json").read_bytes())

    return Predictions.model_validate_json(predictions_path.read_text())


@app.command()
def main(
    image: Path = typer.Option(..., help="Path to the input photo"),
    yolo_weights: Path = typer.Option(..., help="Trained YOLO detector weights (best.pt)"),
    affordance_ckpt: Path = typer.Option(..., help="Dir containing the trained AffordanceMLP best.pt"),
    out: Path = typer.Option(..., help="Output directory for scene/features/predictions"),
    conf: float = typer.Option(0.05, help="YOLO confidence threshold"),
    resolved_instances: Path = typer.Option(
        None, help="Optional resolved_instances.json for effective-class gating"),
    all_pairs: bool = typer.Option(
        False, help="Score every predicate for every ordered pair, skipping the "
                     "class-based candidate gate (robust to a wrong detector "
                     "class label, e.g. a real dish mislabeled 'pan')"),
    dinov2_probe: Path = typer.Option(
        None, help="Fitted DINOv2 probe pickle path (see dinov2_vs_yolo_classhead.py); "
                    "defaults to DEFAULT_DINOV2_PROBE. Each object's class comes from "
                    "this probe instead of the YOLO class head by default (diagnostic: "
                    "probe 98.76% vs YOLO 93.12% accuracy on identical predicted crops, "
                    "full synthetic corpus)."),
    no_dinov2_probe: bool = typer.Option(
        False, help="Disable DINOv2 classification, falling back to YOLO's own class head."),
):
    from affdet.detect.yolo_boxes import _ultralytics_predictor
    from affdet.extract.dinov2_classifier import load_probe

    predictor = _ultralytics_predictor(str(yolo_weights), conf=conf)
    classifier_path = resolve_classifier_path(dinov2_probe, no_dinov2_probe, DEFAULT_DINOV2_PROBE)
    classifier = load_probe(classifier_path) if classifier_path is not None else None
    preds = run_single_image(
        image_path=image, ckpt_dir=affordance_ckpt, out_dir=out,
        predictor=predictor, classifier=classifier, conf_thresh=conf,
        resolved_instances=resolved_instances, all_pairs=all_pairs,
    )

    if not preds.tier1:
        typer.echo("No affordances predicted (no detections, or no candidate pairs).")
        return

    for e in sorted(preds.tier1, key=lambda e: -e.score):
        verdict = "YES" if e.verdict else "no"
        typer.echo(f"  {e.actor:<20} {e.predicate:<12} {e.recipient:<20} "
                    f"score={e.score:.3f}  [{verdict}]")
    typer.echo(f"\nWritten to {out}")


if __name__ == "__main__":
    app()
