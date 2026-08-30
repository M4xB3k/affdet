"""CLI — run one E5 component-evaluation arm from a committed config.

    python -m affdet.scripts.component_eval run \\
        --config experiments/configs/e5_component/e5-yoloworld-ft.yaml \\
        --root /scratch/$USER/affdet/experiments

Results land in the ledger at <root>/runs.jsonl as `kind: "component"` rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from affdet.experiment.component_config import ComponentEvalConfig
from affdet.experiment.component_runner import run_component_eval

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config: Path = typer.Option(..., help="Arm config YAML"),
    root: Path = typer.Option(..., help="Experiment root (holds runs.jsonl)"),
    max_scenes: int = typer.Option(
        None, help="Override the config's scene cap (smoke runs)"),
) -> None:
    cfg = ComponentEvalConfig.from_yaml(config)
    if max_scenes is not None:
        cfg = cfg.model_copy(update={"max_scenes": max_scenes})

    typer.echo(f"[e5] arm={cfg.name} detector={cfg.detector} "
               f"zero_shot={cfg.zero_shot} split={cfg.split}")
    result = run_component_eval(cfg, root)

    det = result["detection"]
    agn = result["detection_class_agnostic"]
    typer.echo(f"[e5] scenes={result['n_scenes_scored']} "
               f"gold={det['n_gold']} pred={det['n_pred']}")
    typer.echo(f"[e5] class-aware     P={det['precision']:.4f} "
               f"R={det['recall']:.4f} F1={det['f1']:.4f} "
               f"macroF1={det['macro_f1']:.4f}")
    typer.echo(f"[e5] class-agnostic  P={agn['precision']:.4f} "
               f"R={agn['recall']:.4f} F1={agn['f1']:.4f}")

    # The subgroup split is the whole point of the instance-disjoint arm, so
    # it belongs in the log and not only in the ledger. Recall is the honest
    # column here: subgroup precision is an upper bound, since a prediction
    # matching no gold object cannot be attributed to a group.
    for label, block in (("class-aware", det), ("box-only", agn)):
        for name, g in sorted(block.get("subgroups", {}).items()):
            typer.echo(f"[e5]   {label:13s} {name:5s} "
                       f"n={g['n_gold']:6d} tp={g['tp']:6d} fn={g['fn']:6d} "
                       f"recall={g['recall']:.4f}")

    typer.echo(f"[e5] ledger -> {root / 'runs.jsonl'}")


@app.command()
def summarise(
    root: Path = typer.Option(..., help="Experiment root (holds runs.jsonl)"),
) -> None:
    """Print every component row in the ledger, newest last."""
    rows = [json.loads(l) for l in
            (root / "runs.jsonl").read_text().splitlines() if l.strip()]
    for r in [r for r in rows if r.get("kind") == "component"]:
        det = r["metrics"]["detection"]
        typer.echo(f"{r['config']['name']:24s} "
                   f"F1={det['f1']:.4f} macroF1={det['macro_f1']:.4f} "
                   f"n={r['n_scenes_scored']}")


if __name__ == "__main__":
    app()
