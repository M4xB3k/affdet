import typer
from pathlib import Path

app = typer.Typer(help="Train the pairwise affordance head.")


@app.command()
def main(
    scenes: Path = typer.Option(..., help="Rendered scenes directory"),
    features: Path = typer.Option(..., help="Features directory"),
    labels: Path = typer.Option(..., help="Labels directory"),
    out: Path = typer.Option(..., help="Checkpoint output directory"),
    mock: bool = typer.Option(False, help="Use MockTrainer"),
    seed: int = 0,
    resolved_instances: Path = typer.Option(
        None, help="resolved_instances.json for effective-class gating"),
    unbounded: bool = typer.Option(
        False, help="Train structurally-impossible pairs as hard negatives "
                     "instead of excluding them from the loss (see "
                     "AffordanceDataset.unbounded) -- pairs with inference's "
                     "--all-pairs flag"),
    unbounded_neg_ratio: int = typer.Option(
        None, help="Only with --unbounded: cap hard-negative pairs kept per "
                    "scene to ratio x (candidate pairs in that scene), "
                    "instead of keeping every impossible pair uncapped"),
):
    if mock:
        from affdet.model.mock import MockTrainer
        MockTrainer(seed=seed).fit(features, labels, out)
        typer.echo(f"Mock checkpoint written to {out}")
    else:
        from affdet.model.train import Trainer
        Trainer(seed=seed, resolved_instances=resolved_instances,
                unbounded=unbounded, unbounded_neg_ratio=unbounded_neg_ratio).fit(
            scenes, features, labels, out)
        typer.echo(f"Checkpoint written to {out}")


if __name__ == "__main__":
    app()
