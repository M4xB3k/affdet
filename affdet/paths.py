"""Locations of the bundled pretrained weights.

Resolved relative to the installed package so that the CLI works from a
checkout, an installed wheel, or any working directory. Each can be
overridden by an environment variable, which is what the cluster runs do.
"""
from __future__ import annotations

import os
from pathlib import Path

# affdet/paths.py -> affdet/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.environ.get("AFFDET_WEIGHTS", _REPO_ROOT / "weights"))

DETECTOR = WEIGHTS_DIR / "yolo11s_detector.pt"
CLASS_PROBE = WEIGHTS_DIR / "dinov2_class_probe.pkl"
STUDENT = WEIGHTS_DIR / "student"   # dir containing best.pt (infer.py loads ckpt_dir/"best.pt")


def require(path: Path, what: str) -> Path:
    """Fail with a message that says what is missing and how to supply it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"Bundled weights live in {WEIGHTS_DIR}; set AFFDET_WEIGHTS to point "
            f"elsewhere, or pass an explicit path on the command line."
        )
    return path
