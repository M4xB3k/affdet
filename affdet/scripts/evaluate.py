import typer
from pathlib import Path
from affdet.eval.metrics import write_metrics_csv

app = typer.Typer(help="Compute eval metrics.")


@app.command()
def main(
    predictions: Path = typer.Option(..., help="Predictions directory"),
    labels: Path = typer.Option(..., help="Labels directory"),
    out: Path = typer.Option(..., help="Results output directory"),
):
    csv_path = write_metrics_csv(predictions, labels, out)
    typer.echo(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    app()
