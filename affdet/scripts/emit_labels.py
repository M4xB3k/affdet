import typer
from pathlib import Path

app = typer.Typer(help="Emit affordance Labels artifacts.")


@app.command()
def legacy(
    scenes: Path = typer.Option(..., help="Rendered scenes directory"),
    out: Path = typer.Option(Path("data/labels/scenes"), help="Output labels directory"),
    threshold: int = typer.Option(6, help="Score >= threshold → verdict=1"),
):
    """Old path: read an omni_label block baked into each scene JSON."""
    from affdet.label.scene_labels import emit_scene_labels
    n = emit_scene_labels(scenes_dir=scenes, labels_dir=out, threshold=threshold)
    typer.echo(f"Wrote {n} label files to {out}")


@app.command()
def resolve(
    curation: Path = typer.Option(Path("scripts/omni/curation.json")),
    out: Path = typer.Option(Path("data/omni/labels/resolved_instances.json")),
):
    """Normalize curation.json → resolved_instances.json."""
    from affdet.label.resolved_instances import write_resolved
    n = write_resolved(curation, out)
    typer.echo(f"Wrote {n} resolved instances to {out}")


@app.command(name="omni-score")
def omni_score(
    scenes: Path = typer.Option(..., help="Omni scenes directory"),
    vlm: Path = typer.Option(Path("data/omni/labels/omni_vlm_raw.jsonl")),
    resolved: Path = typer.Option(Path("data/omni/labels/resolved_instances.json")),
    out: Path = typer.Option(Path("data/omni/labels/scenes")),
):
    """Derive per-scene soft labels from the VLM score table."""
    from affdet.label.vlm_resolver import VlmScoreResolver
    from affdet.label.omni_score_labels import derive_scene_labels
    r = VlmScoreResolver.from_files(vlm, resolved)
    cov = derive_scene_labels(scenes, out, r)
    typer.echo(f"Emitted {cov['n_emitted']} labels, dropped {cov['n_dropped']}. "
               f"Coverage: {out}/label_coverage.json")


@app.command(name="pair-score")
def pair_score(
    scenes: Path = typer.Option(..., help="Omni scenes directory"),
    vlm: Path = typer.Option(..., help="Path to labels_omni.jsonl (pair-labeling corpus: "
                             "{actor, recipient, predicate, score} rows keyed by specific "
                             "instance ids, from affdet/scripts/label.py's OmniLabeler)"),
    resolved: Path = typer.Option(Path("data/omni/labels/resolved_instances.json")),
    curation: Path = typer.Option(Path("scripts/omni/curation.json")),
    out: Path = typer.Option(Path("data/omni/labels/scenes")),
):
    """Derive per-scene soft labels from the pair-labeling VLM corpus, via
    placement-aware (actor_instance, recipient_category, placement) lookup."""
    from affdet.label.pair_vlm_resolver import PairVlmResolver
    from affdet.label.omni_score_labels import derive_scene_labels
    r = PairVlmResolver.from_files(vlm, resolved, curation)
    cov = derive_scene_labels(scenes, out, r)
    typer.echo(f"Emitted {cov['n_emitted']} labels, dropped {cov['n_dropped']}. "
               f"Coverage: {out}/label_coverage.json")


if __name__ == "__main__":
    app()
