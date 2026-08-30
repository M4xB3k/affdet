import typer
from pathlib import Path

app = typer.Typer(help="Label paired-scene affordance renders with a VLM.")


@app.command()
def main(
    pairs: Path = typer.Option(..., help="Path to pairs.json ([{actor, recipient}, ...])"),
    renders: Path = typer.Option(..., help="Path to the paired-render output dir "
                                  "(from affdet/render/omni_pair_render.py)"),
    out: Path = typer.Option(Path("data/labels/omni_vlm_paired.jsonl"), help="Output JSONL path"),
    model: str = typer.Option(
        "gpt-5-mini",
        help="Model / Azure deployment name to use. Set AZURE_OPENAI_ENDPOINT "
             "and AZURE_OPENAI_API_KEY env vars to route through Azure OpenAI "
             "instead of api.openai.com.",
    ),
    dry_run: bool = typer.Option(False, help="Print pair/predicate counts without calling API"),
    limit: int = typer.Option(None, help="Max pairs to label (for testing)"),
    offset: int = typer.Option(0, help="Skip first N pairs"),
    chunk_size: int = typer.Option(500, help="Pairs per chunk before cooldown"),
    chunk_cooldown: int = typer.Option(60, help="Seconds to sleep between chunks"),
    max_workers: int = typer.Option(1, help="Concurrent API calls (1 = sequential, "
                                     "original behavior)"),
    tokens_per_minute: int = typer.Option(
        900_000, help="Token budget/min for the rate limiter, only used when "
                       "max_workers > 1. Keep under your actual Azure/OpenAI TPM "
                       "quota to leave headroom."),
    tiles_dir: Path = typer.Option(
        None, help="Solo-object tile renders root (<tiles_dir>/<class>/tile_<NNN>.png). "
                    "When set, canCut-only jobs use a composed 2x2 grid (2 paired "
                    "angles + 2 solo tiles) instead of the standard 4-angle paired "
                    "grid, for clearer knife-type classification."),
):
    from affdet.label.vlm import OmniLabeler
    from affdet.label.rate_limiter import TokenRateLimiter

    rate_limiter = TokenRateLimiter(tokens_per_minute=tokens_per_minute) if max_workers > 1 else None
    OmniLabeler(
        model=model, chunk_size=chunk_size, chunk_cooldown=chunk_cooldown,
        rate_limiter=rate_limiter, tiles_dir=tiles_dir,
    ).run(
        pairs_path=pairs,
        renders_dir=renders,
        out_path=out,
        offset=offset,
        limit=limit,
        dry_run=dry_run,
        max_workers=max_workers,
    )
    if not dry_run:
        typer.echo(f"Labels written to {out}")


if __name__ == "__main__":
    app()
