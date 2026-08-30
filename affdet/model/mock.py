"""MockTrainer + MockInferencer — for Phase 0 plumbing tests only.

Trainer writes a junk checkpoint. Inferencer reads labels and emits predictions
with controlled noise so eval has something non-trivial to grade.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from affdet.schemas import Labels, Predictions, PredictionEntry


@dataclass
class MockTrainer:
    seed: int = 0

    def fit(self, features_dir: Path | str, labels_dir: Path | str,
            ckpt_dir: Path | str) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "checkpoint.json").write_text(
            json.dumps({"seed": self.seed, "kind": "mock"})
        )


@dataclass
class MockInferencer:
    seed: int = 0
    accuracy: float = 0.8

    def predict(self, features_dir: Path | str, labels_dir: Path | str,
                ckpt_dir: Path | str, predictions_dir: Path | str) -> None:
        labels_dir = Path(labels_dir)
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(self.seed)

        for label_path in sorted(labels_dir.glob("scene_*.json")):
            labels = Labels.model_validate_json(label_path.read_text())
            entries: list[PredictionEntry] = []
            for lab in labels.tier1:
                flip = rng.random() > self.accuracy
                pred_verdict = (1 - lab.verdict) if flip else lab.verdict
                score = float(rng.uniform(0.5, 1.0)) if pred_verdict == 1 \
                    else float(rng.uniform(0.0, 0.5))
                entries.append(PredictionEntry(
                    actor=lab.actor,
                    recipient=lab.recipient,
                    predicate=lab.predicate,
                    verdict=pred_verdict,
                    actor_part=lab.actor_part if pred_verdict == 1 else None,
                    score=score,
                    actor_part_logits=None,
                ))
            preds = Predictions(scene_id=labels.scene_id, tier1=entries)
            (predictions_dir / label_path.name).write_text(
                preds.model_dump_json(indent=2)
            )
