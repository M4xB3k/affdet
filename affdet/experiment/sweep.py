"""run_sweep — run several configs (one arm each) and collect their records."""
from __future__ import annotations
from pathlib import Path

from affdet.experiment.config import ExperimentConfig
from affdet.experiment.runner import run_experiment


def run_sweep(config_paths, root: Path | str, mock: bool = False) -> list[dict]:
    records = []
    for p in config_paths:
        cfg = ExperimentConfig.from_yaml(p)
        records.append(run_experiment(cfg, root, mock=mock))
    return records
