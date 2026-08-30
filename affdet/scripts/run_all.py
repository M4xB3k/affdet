"""End-to-end pipeline driver. Phase 0: all 6 stages on mocks."""
import typer
from pathlib import Path

app = typer.Typer(help="Run the full affdet pipeline end-to-end.")


@app.command()
def main(
    out: Path = typer.Option(..., help="Output root; subdirs created per stage"),
    mock: bool = typer.Option(False, help="Use mocks for every stage that has one"),
    n_scenes: int = 5,
    seed: int = 0,
):
    out.mkdir(parents=True, exist_ok=True)
    scenes = out / "scenes"
    features = out / "features"
    labels = out / "labels"
    ckpt = out / "checkpoints"
    preds = out / "predictions"
    pddl = out / "pddl"
    results = out / "results"

    if mock:
        from affdet.render.mock import MockRenderer
        from affdet.extract.mock import MockExtractor
        from affdet.model.mock import MockTrainer, MockInferencer

        MockRenderer(seed=seed, n_scenes=n_scenes).run(scenes)
        MockExtractor(seed=seed).run(scenes, features)
        MockTrainer(seed=seed).fit(features, labels, ckpt)
        MockInferencer(seed=seed, accuracy=0.8).predict(features, labels, ckpt, preds)
    else:
        raise NotImplementedError(
            "Non-mock run-all requires real implementations. Use --mock for testing."
        )

    # adapter and eval are real even in --mock runs (pure transformations, no mock)
    from affdet.adapter.pddl_emitter import emit_pddl
    from affdet.eval.metrics import write_metrics_csv

    emit_pddl(preds, pddl)
    csv_path = write_metrics_csv(preds, labels, results)
    typer.echo(f"Pipeline complete. Metrics: {csv_path}")


if __name__ == "__main__":
    app()
