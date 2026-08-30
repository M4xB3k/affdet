import typer
from pathlib import Path

app = typer.Typer(help="Run inference, emit predictions.")


@app.command()
def main(
    scenes: Path = typer.Option(..., help="Scenes directory (for pair geometry)"),
    features: Path = typer.Option(..., help="Features directory"),
    labels: Path = typer.Option(..., help="Labels (used for key alignment)"),
    ckpt: Path = typer.Option(..., help="Checkpoint directory"),
    out: Path = typer.Option(..., help="Predictions output directory"),
    mock: bool = typer.Option(False, help="Use MockInferencer"),
    seed: int = 0,
    accuracy: float = 0.8,
    resolved_instances: Path = typer.Option(
        None, help="resolved_instances.json for effective-class gating"),
):
    if mock:
        from affdet.model.mock import MockInferencer
        MockInferencer(seed=seed, accuracy=accuracy).predict(features, labels, ckpt, out)
        typer.echo(f"Mock predictions written to {out}")
    else:
        from affdet.model.infer import Inferencer
        Inferencer(resolved_instances=resolved_instances).predict(scenes, features, labels, ckpt, out)
        typer.echo(f"Predictions written to {out}")


if __name__ == "__main__":
    app()
