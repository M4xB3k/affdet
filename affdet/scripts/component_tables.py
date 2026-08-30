"""CLI — export E5 ledger rows to committed results + the thesis table.

    python -m affdet.scripts.component_tables export \\
        --ledger experiments/ledger/runs.jsonl

Writes, in this order (results committed as md/json first, the table
generated from the same rows):

  data/eval/E5_COMPONENT_QUALITY.md
  data/eval/e5_component_quality.json
  tools/docs/thesis/tables/component_quality.tex
"""
from __future__ import annotations

from pathlib import Path

import typer

from affdet.experiment.component_tables import (
    load_classification_runs,
    load_component_runs,
    write_component_json,
    write_component_markdown,
    write_component_tex,
)

app = typer.Typer(add_completion=False)


@app.command()
def export(
    ledger: Path = typer.Option(..., help="Path to runs.jsonl"),
    eval_dir: Path = typer.Option(Path("data/eval"),
                                  help="Committed results directory"),
    tables_dir: Path = typer.Option(Path("tools/docs/thesis/tables"),
                                    help="Thesis tables directory"),
    per_class: bool = typer.Option(
        False, help="Include the per-class breakdown in the LaTeX table"),
    include: str = typer.Option(
        "e5-Da-,e5-Db-",
        help="Comma-separated arm-name prefixes to table. Defaults to the "
             "current ladder; pass '' to table every component row."),
) -> None:
    prefixes = tuple(p for p in include.split(",") if p) or None
    runs = load_component_runs(ledger, include=prefixes)
    if not runs:
        typer.echo(f"[e5] no component runs in {ledger}"
                   + (f" matching {list(prefixes)}" if prefixes else ""),
                   err=True)
        raise typer.Exit(1)

    if prefixes:
        skipped = [r["config"]["name"]
                   for r in load_component_runs(ledger)
                   if not str(r["config"]["name"]).startswith(prefixes)]
        if skipped:
            # Loud, not silent: these are usually superseded arms whose name
            # changed, which "newest wins" cannot retire on its own.
            typer.echo(f"[e5] skipping {len(skipped)} arm(s) outside "
                       f"{list(prefixes)}: {', '.join(sorted(skipped))}")

    typer.echo(f"[e5] {len(runs)} arm(s): "
               f"{', '.join(r['config']['name'] for r in runs)}")

    cls = load_classification_runs(ledger, include=prefixes)
    if cls:
        typer.echo(f"[e5] {len(cls)} classification arm(s): "
                   f"{', '.join(r['config']['name'] for r in cls)}")
    md = write_component_markdown(runs, eval_dir / "E5_COMPONENT_QUALITY.md",
                                  classification=cls)
    js = write_component_json(runs, eval_dir / "e5_component_quality.json")
    tex = write_component_tex(runs, tables_dir / "component_quality.tex",
                              per_class=per_class)

    for p in (md, js, tex):
        typer.echo(f"[e5] wrote {p}")


@app.command()
def arms(
    ledger: Path = typer.Option(..., help="Path to runs.jsonl"),
) -> None:
    """List the component arms the export would use, without writing files.

    Also keeps this app at two commands, which is what stops Typer from
    collapsing the group and making `export` unreachable — see
    test_cli_accepts_its_subcommand_name.
    """
    runs = load_component_runs(ledger)
    if not runs:
        typer.echo(f"[e5] no component runs in {ledger}", err=True)
        raise typer.Exit(1)
    for r in runs:
        det = r["metrics"]["detection"]
        typer.echo(f"{r['config']['name']:26s} "
                   f"F1={det['f1']:.4f} n={r['n_scenes_scored']} "
                   f"run_id={r.get('run_id')}")


if __name__ == "__main__":
    app()
