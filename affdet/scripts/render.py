import typer
from pathlib import Path

app = typer.Typer(help="Render scenes (mock, YCB Blender, or OmniObject3D Blender).")


@app.command()
def main(
    out: Path = typer.Option(Path("data/scenes"), help="Output scenes directory"),
    mock: bool = typer.Option(False, help="Use MockRenderer instead of Blender"),
    omni: bool = typer.Option(False, help="Render OmniObject3D scenes from label JSONL"),
    labels: Path = typer.Option(None, help="[omni] Path to omni_vlm_raw.jsonl"),
    omni_root: Path = typer.Option(None, help="[omni] Path to raw_scans root dir"),
    tex_dir: Path = typer.Option(Path("data/table_textures"), help="Table texture directory"),
    seed: int = 0,
    n_scenes: int = 10,
    score_threshold: int = 0,
):
    if mock:
        from affdet.render.mock import MockRenderer
        MockRenderer(seed=seed, n_scenes=n_scenes).run(out)
        typer.echo(f"Wrote {n_scenes} mock scenes to {out}")
    elif omni:
        if labels is None or omni_root is None:
            typer.echo("--omni requires --labels and --omni-root", err=True)
            raise typer.Exit(1)
        # Omni renderer must run inside Blender's Python — print the command to use
        typer.echo(
            "Run OmniObject3D rendering with:\n\n"
            f"  blender --background --python affdet/render/omni_blender.py -- \\\n"
            f"    --labels {labels} \\\n"
            f"    --omni-root {omni_root} \\\n"
            f"    --out {out} \\\n"
            f"    --tex-dir {tex_dir} \\\n"
            f"    --n-scenes {n_scenes} \\\n"
            f"    --seed {seed} \\\n"
            f"    --score-threshold {score_threshold}\n"
        )
    else:
        typer.echo(
            "YCB Blender renderer not yet wired into CLI. "
            "Use: blender --background --python affdet/render/blender.py -- --help"
        )


if __name__ == "__main__":
    app()
