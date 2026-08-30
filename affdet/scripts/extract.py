import typer
from pathlib import Path

app = typer.Typer(help="Extract DINOv2 features per object.")


@app.command()
def main(
    in_dir: Path = typer.Option(..., "--in", help="Input scenes directory"),
    out: Path = typer.Option(..., help="Output features directory"),
    mock: bool = typer.Option(False, help="Use MockExtractor"),
    bbox_source: str = typer.Option("auto", help="oracle | yolo | auto"),
    seed: int = 0,
):
    if mock:
        from affdet.extract.mock import MockExtractor
        MockExtractor(seed=seed).run(in_dir, out)
        typer.echo(f"Mock features written to {out}")
    else:
        from affdet.extract.dinov2 import DINOv2Extractor
        # The bbox is baked into each Scene JSON already (oracle boxes in the
        # normal set, YOLO-matched boxes in the ablation's yolo set); this flag
        # only tags the features' provenance. "auto" == the scenes as-given.
        source = "oracle" if bbox_source == "auto" else bbox_source
        DINOv2Extractor().run(in_dir, out, bbox_source=source)
        typer.echo(f"DINOv2 features written to {out} (bbox_source={source})")


if __name__ == "__main__":
    app()
