"""ExperimentConfig — one config names every knob for one pipeline arm."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    scenes: str
    labels: str
    bbox_source: Literal["oracle", "yolo"] = "oracle"
    crop_padding: float = 0.0
    backbone: Literal["vitb14", "vits14", "vitl14"] = "vitb14"
    head: Literal["mlp", "mlp-big", "attention"] = "mlp"
    #: Generalization level to evaluate at (E1 ladder): "scene" holds out
    #: arrangements, "pair" holds out mesh combinations, "instance" holds
    #: out the meshes themselves.
    split_strategy: Literal["scene", "pair", "instance"] = "scene"
    #: Pair-level instance holdout (E1c/E1c'), mirroring AffordanceDataset and
    #: Inferencer. Takes precedence over split_strategy when set.
    holdout_mode: Literal["strict", "actor_only"] | None = None
    holdout_frac: float = 0.20
    seed: int = 0
    epochs: int = 50
    lr: float = 1e-3
    resolved_instances: str | None = None
    unbounded: bool = False
    unbounded_neg_ratio: int | None = None
    #: Score every predicate for every ordered pair at eval time, skipping the
    #: intersection-matrix class gate (Inferencer.all_pairs). Not comparable
    #: to the gated ladder -- different denominator -- report as its own arm.
    all_pairs: bool = False
    #: Downsample the majority class per predicate to a true 50/50 split
    #: (AffordanceDataset.balance_masks). Independent of pos_weight, which
    #: reweights the loss but leaves the data skewed. Train-time only.
    balanced: bool = False

    def extract_key(self) -> dict:
        return {"scenes": self.scenes, "bbox_source": self.bbox_source,
                "crop_padding": self.crop_padding, "backbone": self.backbone}

    def extract_hash(self) -> str:
        blob = json.dumps(self.extract_key(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))
