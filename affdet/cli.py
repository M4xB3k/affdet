"""Command line entry point.

Two commands carry the public surface:

    affdet predict --image photo.jpg     an image in, affordance predicates out
    affdet train   --scenes ... --labels ...   retrain the pairwise head

Everything else the thesis used lives in affdet/scripts/ and is driven
directly; those are experiment drivers, not a stable interface.
"""
from __future__ import annotations

from pathlib import Path

import typer

from affdet import paths

app = typer.Typer(
    help="Detect inter-object affordances in a photograph.",
    no_args_is_help=True,
)


@app.command()
def predict(
    image: Path = typer.Option(..., "--image", "-i", help="Input photograph."),
    out: Path = typer.Option(
        Path("affdet_out"), "--out", "-o",
        help="Directory for the scene, features and predictions written along the way."),
    conf: float = typer.Option(0.5, help="Detector confidence threshold."),
    all_pairs: bool = typer.Option(
        False, "--all-pairs",
        help="Score every predicate for every ordered pair, skipping the class-based "
             "candidate gate. Useful when a detector class label is wrong."),
    detector: Path = typer.Option(None, help="Override the bundled detector weights."),
    student: Path = typer.Option(None, help="Override the bundled affordance head."),
    probe: Path = typer.Option(None, help="Override the bundled DINOv2 class probe."),
    no_probe: bool = typer.Option(
        False, "--no-probe",
        help="Name objects with the detector's own class head instead of the probe."),
    json_out: bool = typer.Option(False, "--json", help="Print predictions as JSON."),
):
    """Run detection, naming and affordance prediction on one image."""
    from affdet.detect.yolo_boxes import _ultralytics_predictor
    from affdet.extract.dinov2_classifier import load_probe
    from affdet.scripts.detect_and_infer import run_single_image

    if not image.exists():
        raise typer.BadParameter(f"No such image: {image}")

    det_path = paths.require(detector or paths.DETECTOR, "Detector weights")
    stu_path = paths.require(student or paths.STUDENT, "Affordance head directory")

    classifier = None
    if not no_probe:
        classifier = load_probe(paths.require(probe or paths.CLASS_PROBE, "Class probe"))

    predictor = _ultralytics_predictor(str(det_path), conf=conf)
    preds = run_single_image(
        image_path=image,
        ckpt_dir=stu_path,
        out_dir=out,
        predictor=predictor,
        classifier=classifier,
        conf_thresh=conf,
        resolved_instances=None,
        all_pairs=all_pairs,
    )

    if json_out:
        typer.echo(preds.model_dump_json(indent=2))
        return

    if not preds.tier1:
        typer.echo("No affordances predicted: nothing detected, or no admissible pairs.")
        raise typer.Exit(code=0)

    typer.echo(f"{'actor':<22}{'predicate':<14}{'recipient':<22}{'score':>7}")
    typer.echo("-" * 65)
    for e in sorted(preds.tier1, key=lambda e: -e.score):
        mark = "*" if e.verdict else " "
        typer.echo(f"{e.actor:<22}{e.predicate:<14}{e.recipient:<22}{e.score:>7.3f} {mark}")
    typer.echo(f"\n* = predicate holds at the calibrated threshold. Written to {out}/")


@app.command()
def train(
    scenes: Path = typer.Option(..., help="Directory of rendered scenes."),
    features: Path = typer.Option(
        ..., help="Directory of extracted DINOv2 features (see `affdet extract`)."),
    labels: Path = typer.Option(..., help="Directory of per-scene label JSON."),
    out: Path = typer.Option(Path("runs/local"), help="Where to write the checkpoint."),
    seed: int = typer.Option(0),
    unbounded: bool = typer.Option(
        False, "--unbounded",
        help="Train structurally impossible pairs as hard negatives instead of "
             "excluding them from the loss. Pairs with `predict --all-pairs`."),
):
    """Train the pairwise affordance head on a rendered corpus.

    Features must be extracted first; the backbone is frozen, so this reads
    cached vectors rather than running DINOv2 per epoch.
    """
    from affdet.model.train import Trainer

    out.mkdir(parents=True, exist_ok=True)
    Trainer(seed=seed, unbounded=unbounded).fit(scenes, features, labels, out)
    typer.echo(f"Checkpoint written to {out}")


@app.command()
def extract(
    scenes: Path = typer.Option(..., help="Directory of rendered scenes."),
    out: Path = typer.Option(..., help="Directory to write feature vectors into."),
    backbone: str = typer.Option("vitb14", help="DINOv2 variant."),
):
    """Extract frozen DINOv2 features for a corpus, ready for `affdet train`."""
    from affdet.extract.dinov2 import DINOv2Extractor

    out.mkdir(parents=True, exist_ok=True)
    DINOv2Extractor(variant=backbone).run(scenes, out)
    typer.echo(f"Features written to {out}")


if __name__ == "__main__":
    app()
