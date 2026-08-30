"""CLI — run the DINOv2-probe vs. detector-class-head comparison (E5).

    python -m affdet.scripts.classification_eval run \\
        --config experiments/configs/e5_component/e5-yoloworld-ft.yaml \\
        --root /scratch/$USER/affdet/experiments --seed 0

Unlike the detection arms, the probe is FITTED, so this is run over three
seeds and reported as mean +/- std.
"""
from __future__ import annotations

from pathlib import Path

import typer

from affdet.experiment.classification_runner import run_classification_eval
from affdet.experiment.component_config import ComponentEvalConfig

# A Typer app with exactly ONE command collapses into a bare CLI, so the
# subcommand name stops being accepted and `... run --config X` fails with
# "unexpected extra argument (run)". Pinning the group keeps the invocation
# identical in shape to component_eval's.
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    config: Path = typer.Option(..., help="Arm config YAML"),
    root: Path = typer.Option(..., help="Experiment root (holds runs.jsonl)"),
    seed: int = typer.Option(None, help="Override the config's seed"),
    max_scenes: int = typer.Option(None, help="Cap eval scenes (smoke runs)"),
    max_train_scenes: int = typer.Option(
        None, help="Cap scenes used to fit the probe (smoke runs). Without "
                   "it a smoke run still fits on the whole train split."),
) -> None:
    cfg = ComponentEvalConfig.from_yaml(config)
    updates = {}
    if seed is not None:
        updates["seed"] = seed
    if max_scenes is not None:
        updates["max_scenes"] = max_scenes
    if max_train_scenes is not None:
        updates["max_train_scenes"] = max_train_scenes
    if updates:
        cfg = cfg.model_copy(update=updates)

    typer.echo(f"[e5-cls] arm={cfg.name} detector={cfg.detector} seed={cfg.seed}")
    r = run_classification_eval(cfg, root)

    c = r["classification"]
    typer.echo(f"[e5-cls] train_scenes={r['n_train_scenes']} "
               f"eval_scenes={r['n_eval_scenes']}")
    typer.echo(f"[e5-cls] matched={r['n_matched_objects']} / "
               f"{r['n_gold_objects']} gold "
               f"({r['n_detector_misses']} detector misses, not scored)")
    typer.echo(f"[e5-cls] DINOv2 probe    accuracy = "
               f"{c['probe']['accuracy']:.4f}")
    typer.echo(f"[e5-cls] detector head   accuracy = "
               f"{c['classhead']['accuracy']:.4f}")
    typer.echo(f"[e5-cls] delta           = {c['delta_accuracy']:+.4f}")
    typer.echo(f"[e5-cls] ledger -> {root / 'runs.jsonl'}")


@app.command()
def summarise(
    root: Path = typer.Option(..., help="Experiment root (holds runs.jsonl)"),
) -> None:
    """Print every classification row in the ledger, newest last.

    Also keeps this app at two commands, which is what stops Typer from
    collapsing the group and making `run` unreachable — see
    test_cli_accepts_its_subcommand_name.
    """
    import json

    rows = [json.loads(l) for l in
            (root / "runs.jsonl").read_text().splitlines() if l.strip()]
    for r in [r for r in rows if r.get("kind") == "classification"]:
        c = r["metrics"]["classification"]
        typer.echo(f"{r['config']['name']:24s} seed={r['config'].get('seed')} "
                   f"probe={c['probe']['accuracy']:.4f} "
                   f"head={c['classhead']['accuracy']:.4f} "
                   f"delta={c['delta_accuracy']:+.4f}")


if __name__ == "__main__":
    app()
