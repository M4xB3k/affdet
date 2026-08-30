import json
import typer
from pathlib import Path

app = typer.Typer(help="Object detector: dataset conversion + fine-tuning.")


@app.command()
def convert(
    scenes: Path = typer.Option(..., "--in", help="Scenes directory (Scene JSONs)"),
    out: Path = typer.Option(..., help="Output dataset directory"),
    val_fraction: float = typer.Option(0.1, help="Fraction of scenes for val split"),
    seed: int = 0,
    curation: Path = typer.Option(
        None, help="curation.json — validates obj_class against curated categories"),
    exclude: Path = typer.Option(
        None, help="Text file with one scene_id per line to exclude"),
):
    """Convert Scene JSONs to an Ultralytics detection dataset."""
    from affdet.detect.yolo_dataset import convert as convert_dataset

    allowed = None
    if curation is not None:
        allowed = {k for k in json.loads(curation.read_text())
                   if not k.startswith("_")}
    excluded = None
    if exclude is not None:
        excluded = {l.strip() for l in exclude.read_text().splitlines() if l.strip()}

    data_yaml = convert_dataset(
        scenes, out, val_fraction=val_fraction, seed=seed,
        allowed_classes=allowed, exclude_scene_ids=excluded,
    )
    typer.echo(f"Dataset written: {data_yaml}")


@app.command()
def train(
    data: Path = typer.Option(..., help="data.yaml from `detect convert`"),
    model: str = typer.Option("yolo11s.pt", help="Weights: yolo11{n,s,m}.pt or rtdetr-l.pt"),
    epochs: int = 100,
    imgsz: int = 640,
    seed: int = 0,
    project: Path = typer.Option(None, help="Ultralytics runs directory"),
    name: str = typer.Option(None, help="Run name"),
    device: str = typer.Option(None, help="e.g. 0, cpu"),
):
    """Fine-tune a detector; per-class mAP is reported by Ultralytics at the end."""
    from affdet.detect.train import train_detector

    train_detector(
        data, model=model, epochs=epochs, imgsz=imgsz, seed=seed,
        project=project, name=name, device=device,
    )


@app.command()
def boxes(
    scenes: Path = typer.Option(..., "--in", help="Scenes directory (Scene JSONs)"),
    out: Path = typer.Option(..., help="Output scenes dir with YOLO-matched boxes"),
    weights: Path = typer.Option(..., help="Trained detector best.pt"),
    iou: float = typer.Option(0.5, help="IoU threshold for gold<->pred matching"),
    conf: float = typer.Option(0.05, help="Detector confidence threshold"),
):
    """Rewrite Scene bboxes with IoU-matched YOLO predictions (the ablation's
    "yolo arm"). Objects the detector missed are dropped."""
    from affdet.detect.yolo_boxes import rewrite_scenes, _ultralytics_predictor

    predictor = _ultralytics_predictor(str(weights), conf=conf)
    stats = rewrite_scenes(scenes, out, predictor, iou_thresh=iou)
    typer.echo(
        f"Rewrote {stats['n_scenes']} scenes: "
        f"{stats['n_matched']}/{stats['n_gold_objects']} objects matched, "
        f"{stats['n_dropped']} dropped (recall "
        f"{stats['n_matched'] / max(stats['n_gold_objects'], 1):.3f})"
    )
    typer.echo(f"YOLO-box scenes written: {out}")


if __name__ == "__main__":
    app()
