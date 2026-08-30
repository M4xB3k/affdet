import typer
from pathlib import Path
from affdet.adapter.pddl_emitter import emit_pddl

app = typer.Typer(help="Convert predictions to PDDL problem files.")


@app.command()
def main(
    in_dir: Path = typer.Option(..., "--in", help="Predictions directory"),
    out: Path = typer.Option(..., help="PDDL output directory"),
):
    emit_pddl(in_dir, out)
    typer.echo(f"PDDL files written to {out}")


if __name__ == "__main__":
    app()
