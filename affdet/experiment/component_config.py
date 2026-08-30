"""ComponentEvalConfig — one config names one detector arm of the E5
component evaluation.

Deliberately separate from ``ExperimentConfig``, which describes an
*affordance-model* run (backbone, head, epochs, lr, label corpus, holdout
mode). E5 trains nothing and has no affordance labels in it; folding these
knobs into that model would leave every E5 row carrying a dozen meaningless
fields and every E1 row carrying a detector it never used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class ComponentEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    scenes: str
    #: Base architecture, as an Ultralytics weights string
    #: ("yolo11s", "yolov8s-worldv2") or "grounding-dino" for the
    #: transformers-hosted zero-shot detector.
    detector: str
    #: Fine-tuned checkpoint. Required unless this is a zero-shot arm.
    weights: str | None = None
    #: Run the pretrained detector with the taxonomy as its vocabulary,
    #: without any fine-tuning on the render corpus. The controlled
    #: comparison: same architecture, only the training changed.
    zero_shot: bool = False
    iou_thresh: float = 0.5
    #: Held-out portion of the seeded scene split (the same split machinery
    #: the affordance experiments use, so "test" means the same scenes).
    split: Literal["train", "val", "test"] = "test"
    #: Detector confidence floor. Low by default for the same reason the
    #: shipped predictor uses 0.05: a missed detection cannot be recovered
    #: downstream, a spurious one can be filtered.
    conf: float = 0.05
    #: Ultralytics dataset descriptor the detectors were trained against.
    #: The class list is read from here rather than hardcoded, so every arm
    #: predicts over the same vocabulary and it cannot drift from training.
    data_yaml: str | None = None
    #: Take the eval scenes from ``data_yaml``'s own val split instead of a
    #: seeded scene shuffle. REQUIRED for the instance-disjoint arm, whose
    #: test set is defined by mesh identity at dataset-build time: a seeded
    #: shuffle there would score the detector on scenes it trained on.
    split_from_dataset: bool = False
    #: Report a separate score for gold objects whose mesh never appears in
    #: the dataset's TRAIN split ("novel") against everything else ("rest").
    #: The instance-disjoint arm needs this: it reserves meshes but pulls
    #: whole scenes into test, so only ~29% of its scored objects are
    #: actually novel and the pooled F1 does not mean what the arm claims.
    split_novel_meshes: bool = False
    #: Cap on scored scenes. Detector inference is the slow step; None runs
    #: the whole split.
    max_scenes: int | None = None
    #: Crops per feature-extraction call (classification arm only). The
    #: extractor used to stack EVERY crop into one tensor: on the real train
    #: split (~43k crops) that is ~26 GB of float32 before the GPU is
    #: touched, and it OOM-killed all three seeds at 24 GB.
    feature_batch: int = 256
    #: Cap on scenes used to FIT the probe (classification arm only). Without
    #: it a smoke run still pushes the entire train split through DINOv2 and
    #: costs nearly as much as the real run, which defeats the point of
    #: smoke-testing before committing a GPU job. None fits on the whole
    #: train split, which is what a reportable run must do.
    max_train_scenes: int | None = None
    seed: int = 0

    @model_validator(mode="after")
    def _finetuned_arm_needs_weights(self) -> "ComponentEvalConfig":
        if not self.zero_shot and not self.weights:
            raise ValueError(
                f"arm {self.name!r} is not zero-shot and has no weights: a "
                "fine-tuned arm must name its checkpoint, otherwise it would "
                "silently score pretrained weights as if they were fine-tuned"
            )
        return self

    def arm_key(self) -> dict:
        """Identity of this arm for aggregation across seeds. Everything that
        changes what is being measured belongs here; ``seed`` deliberately
        does not, since seeds are what gets aggregated over."""
        return {"scenes": self.scenes, "detector": self.detector,
                "weights": self.weights, "zero_shot": self.zero_shot,
                "iou_thresh": self.iou_thresh, "split": self.split,
                "conf": self.conf}

    def arm_hash(self) -> str:
        blob = json.dumps(self.arm_key(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ComponentEvalConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))
