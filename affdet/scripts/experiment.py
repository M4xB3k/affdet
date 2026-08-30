import typer
from pathlib import Path

app = typer.Typer(help="Config-driven experiment harness.")


@app.command()
def run(
    config: Path = typer.Option(..., help="Experiment config YAML"),
    root: Path = typer.Option(..., help="Experiments root (features/runs/ledger)"),
    mock: bool = typer.Option(False, help="Use mock stages"),
):
    from affdet.experiment.config import ExperimentConfig
    from affdet.experiment.runner import run_experiment
    cfg = ExperimentConfig.from_yaml(config)
    rec = run_experiment(cfg, root, mock=mock)
    cal = rec['metrics']['calibrated']['macro_f1']
    raw = rec['metrics']['raw']['macro_f1']
    typer.echo(f"run_id={rec['run_id']} macro_f1(raw)={raw:.4f} "
               f"macro_f1(calibrated)={cal:.4f}")
    typer.echo(f"ledger: {root}/runs.jsonl")


@app.command()
def sweep(
    configs: list[Path] = typer.Argument(..., help="Config YAMLs"),
    root: Path = typer.Option(..., help="Experiments root"),
    mock: bool = typer.Option(False, help="Use mock stages"),
):
    from affdet.experiment.sweep import run_sweep
    recs = run_sweep(configs, root, mock=mock)
    for r in recs:
        typer.echo(f"{r['run_id']}  "
                   f"macro_f1(raw)={r['metrics']['raw']['macro_f1']:.4f}  "
                   f"macro_f1(calibrated)={r['metrics']['calibrated']['macro_f1']:.4f}")


@app.command()
def report(
    root: Path = typer.Option(..., help="Experiments root holding runs.jsonl"),
    tables: Path = typer.Option(None, help="Directory to write .tex tables into"),
    block: str = typer.Option("calibrated", help="'calibrated' or 'raw'"),
):
    """Aggregate the ledger by seed family and emit a LaTeX results table.

    The thesis \\input's the generated .tex, so reported numbers can never
    drift from the numbers actually on disk.
    """
    from affdet.experiment.aggregate import (
        load_and_aggregate, write_results_table, write_aggregate_json,
    )
    families = load_and_aggregate(root / "runs.jsonl")
    if not families:
        typer.echo(f"no runs found in {root}/runs.jsonl")
        raise typer.Exit(1)

    for fam in families:
        blk = fam.get(block, {})
        macro = blk.get("macro_f1") or {}
        mean, std = macro.get("mean"), macro.get("std") or 0.0
        shown = f"{mean:.4f} +/- {std:.4f}" if mean is not None else "--"
        typer.echo(f"{fam['name']}  seeds={fam['seeds']}  "
                   f"macro_f1({block})={shown}")

    if tables:
        tex = write_results_table(families, tables / f"results_{block}.tex",
                                  block=block)
        js = write_aggregate_json(families, tables / f"aggregate_{block}.json")
        typer.echo(f"wrote {tex}")
        typer.echo(f"wrote {js}")


if __name__ == "__main__":
    app()
