"""JSONL run ledger — one line per experiment run (config + git SHA + metrics)."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def append_run(ledger_path: Path | str, record: dict) -> None:
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_runs(ledger_path: Path | str) -> list[dict]:
    p = Path(ledger_path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
