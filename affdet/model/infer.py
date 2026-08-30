"""Inferencer — run the trained AffordanceMLP over scene pairs (a forward pass).

NOT a VLM call. Loads best.pt, rebuilds the same per-pair feature vectors the
dataset uses, applies sigmoid, and writes Predictions with score + verdict
(sigmoid >= 0.5). Output contract matches MockInferencer so eval is source-agnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from affdet.schemas import Scene, Features, Predictions, PredictionEntry
from affdet.intersection_matrix import TIER1_PREDICATES, is_candidate_pair
from affdet.model.mlp import AffordanceMLP
from affdet.model.dataset import (
    _geo, build_split_map, instance_holdout, pair_split_for,
)
from affdet.label.resolved_instances import (
    load_effective_class_map, load_dropped_set, effective_class,
)

_PRED_IDX = {p: i for i, p in enumerate(TIER1_PREDICATES)}
_N_GEO = 8


def _infer_input_dim(state: dict) -> int:
    """Recover the head's trained input width (backbone-dependent) from a
    checkpoint state_dict, so the head is rebuilt at the right shape before
    load_state_dict. Handles both heads:
      - AffordanceMLP: 'trunk.0.weight' is [hidden, input_dim] -> take dim 1.
      - AttentionHead: 'proj.weight' is [embed, feat_dim] -> input = 2*feat + geo.
    """
    if "trunk.0.weight" in state:
        return int(state["trunk.0.weight"].shape[1])
    if "proj.weight" in state:
        return int(state["proj.weight"].shape[1]) * 2 + _N_GEO
    raise KeyError("cannot infer input_dim: no known first-layer key in checkpoint")


def _find_scene_jsons(scenes_dir: Path) -> list[Path]:
    for pat in ("scene_*.json", "scene_*/scene_*.json",
                "worker_*/scene_*/scene_*.json"):
        found = sorted(scenes_dir.glob(pat))
        if found:
            return found
    return []


@dataclass
class Inferencer:
    threshold: float = 0.5
    resolved_instances: Path | str | None = None
    model_factory: "Callable[[], object] | None" = None
    all_pairs: bool = False
    """Score every predicate for every ordered object pair, skipping the
    intersection-matrix class gate. The trained model never saw structurally
    impossible pairs as explicit negatives, so this is an extrapolation — but
    it protects against a wrong detector class label silently dropping a real
    affordance (e.g. a real dish mislabeled "pan" still gets canContain
    scored). Off by default: the gate is what training/eval assume."""
    gate_classifier: "object | None" = None
    """Object exposing sklearn-style predict(features) -> labels (e.g.
    DINOv2Classifier). When set, the intersection-matrix gate is built from
    THIS classifier's prediction on each object's own DINOv2 features instead
    of scene.obj_class (ground truth) -- the probe-in-the-loop gate, as
    opposed to every other split/eval in this codebase which gates on the
    renderer's real label. Lets a gated eval measure what the REAL probe's
    error rate costs, instead of assuming perfect upstream classification.
    Ignored when all_pairs=True (no gate to feed). Does not affect the
    features fed to the affordance model itself -- those are still the real
    extracted features regardless of what class the gate believes."""
    split: Literal["train", "val", "test"] | None = None
    split_strategy: Literal["scene", "pair", "instance"] = "scene"
    """Score only scenes in this split, using AffordanceDataset's own seeded
    70/15/15 scene split (scene_split_map) as the single source of truth --
    matching "test" here to the SAME held-out scenes the model never trained
    or early-stopped on. None (default) scores every scene, matching the
    pre-existing behavior (used by callers that already pass an
    already-split scenes_dir)."""
    seed: int = 0
    """Only used with `split` — must match the seed AffordanceDataset/Trainer
    used, or "test" here won't line up with the model's real held-out set."""
    holdout_mode: str | None = None
    """Pair-level instance holdout ("strict" or "actor_only"), mirroring
    AffordanceDataset. MUST match the value used to train, or the scored pairs
    are not the ones the model held out. Takes precedence over
    `split_strategy`; every scene is visited and each PAIR is routed."""
    holdout_frac: float = 0.20

    def predict(self, scenes_dir, features_dir, labels_dir, ckpt_dir,
                predictions_dir) -> None:
        scenes_dir = Path(scenes_dir); features_dir = Path(features_dir)
        ckpt_dir = Path(ckpt_dir); predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        eff_map = (load_effective_class_map(self.resolved_instances)
                   if self.resolved_instances is not None else None)
        dropped = (load_dropped_set(self.resolved_instances)
                   if self.resolved_instances is not None else set())
        # eff_map must be passed here too: the holdout draw has to be
        # IDENTICAL to the one the trainer used, or inference would score
        # against a different set of held-out meshes.
        holdout = (instance_holdout(scenes_dir, seed=self.seed,
                                    frac=self.holdout_frac,
                                    effective_classes=eff_map)
                   if self.holdout_mode is not None else None)
        # With a pair-level holdout every scene is visited; the split filter
        # applies per pair below, not per scene.
        split_map = (build_split_map(scenes_dir, self.split_strategy, seed=self.seed)
                     if self.split is not None and holdout is None else None)

        state = torch.load(ckpt_dir / "best.pt", map_location="cpu")
        # Reconstruct the head at the width it was TRAINED at (backbone-dependent),
        # read from the checkpoint's first-layer weight — assuming 1544 (vitb14)
        # would make load_state_dict fail on a vits14/vitl14 checkpoint.
        input_dim = _infer_input_dim(state["model_state"])
        model = (self.model_factory(input_dim) if self.model_factory is not None
                 else AffordanceMLP(input_dim=input_dim))
        model.load_state_dict(state["model_state"])
        model.eval()

        for scene_json in _find_scene_jsons(scenes_dir):
            if split_map is not None and split_map[scene_json.stem] != self.split:
                continue
            scene = Scene.model_validate_json(scene_json.read_text())
            # A scene containing ANY curation-dropped instance is untrustworthy
            # as a whole (see AffordanceDataset) -- skip it entirely, writing
            # no predictions file for it, rather than excise only the bad
            # instance and still predict on the scene's other objects.
            if any(o.obj_id in dropped for o in scene.objects):
                continue
            feat_path = features_dir / f"{scene_json.stem}.npz"
            if not feat_path.exists():
                continue
            feats = Features.load(feat_path)
            feat_by_id = {oid: feats.features[i] for i, oid in enumerate(feats.obj_ids)}
            W, H = float(scene.image_size[0]), float(scene.image_size[1])

            # One predict() call per scene, not per pair: gate_classifier's
            # prediction depends only on the object's own features, so every
            # pair involving that object reuses the same predicted label.
            gate_cls_by_id: dict[str, str] | None = None
            if self.gate_classifier is not None and not self.all_pairs:
                ids = [o.obj_id for o in scene.objects]
                X = np.stack([feat_by_id[oid] for oid in ids]).astype("float32")
                gate_cls_by_id = dict(zip(ids, self.gate_classifier.predict(X)))

            entries: list[PredictionEntry] = []
            for actor in scene.objects:
                a_eff = (gate_cls_by_id[actor.obj_id] if gate_cls_by_id is not None
                         else effective_class(actor.obj_id, actor.obj_class, eff_map))
                for recip in scene.objects:
                    if actor.obj_id == recip.obj_id:
                        continue
                    if holdout is not None and self.split is not None:
                        if pair_split_for(actor.obj_id, recip.obj_id, holdout,
                                          mode=self.holdout_mode) != self.split:
                            continue
                    r_class = (gate_cls_by_id[recip.obj_id] if gate_cls_by_id is not None
                               else recip.obj_class)
                    cand = (list(TIER1_PREDICATES) if self.all_pairs else
                            [p for p in TIER1_PREDICATES
                             if is_candidate_pair(a_eff, r_class, p)])
                    if not cand:
                        continue
                    x = np.concatenate([
                        feat_by_id[actor.obj_id].astype("float32"),
                        feat_by_id[recip.obj_id].astype("float32"),
                        np.array(_geo(actor, recip, W, H), dtype="float32"),
                    ])
                    with torch.no_grad():
                        logits = model(torch.from_numpy(x).unsqueeze(0))[0]
                        probs = torch.sigmoid(logits)
                    for p in cand:
                        s = float(probs[_PRED_IDX[p]])
                        entries.append(PredictionEntry(
                            actor=actor.obj_id, recipient=recip.obj_id,
                            predicate=p, verdict=1 if s >= self.threshold else 0,
                            score=s, actor_part=None, actor_part_logits=None))
            (predictions_dir / scene_json.name).write_text(
                Predictions(scene_id=scene.scene_id, tier1=entries).model_dump_json(indent=2))
