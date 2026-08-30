"""Trainer — training loop for AffordanceMLP.

Usage:
    trainer = Trainer(epochs=50, seed=0)
    trainer.fit(scenes_dir, features_dir, labels_dir, ckpt_dir)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from affdet.intersection_matrix import TIER1_PREDICATES
from affdet.model.dataset import AffordanceDataset
from affdet.model.mlp import AffordanceMLP

_N_PRED = len(TIER1_PREDICATES)
_POS_WEIGHT_CAP = 30.0


def _compute_pos_weights(dataset: AffordanceDataset) -> torch.Tensor:
    """Per-predicate pos_weight = min(n_neg / n_pos, cap)."""
    all_idx = list(range(len(dataset._xs)))
    weights = torch.ones(_N_PRED)
    for i in range(_N_PRED):
        valid = [j for j in all_idx if dataset._masks[j, i] > 0]
        if not valid:
            continue
        # Targets are soft VLM scores in [0,1] (kept soft for the BCE loss),
        # but class-imbalance counting is inherently binary, so threshold at
        # 0.5 (the same boundary used to derive `verdict`) here only.
        ys = (dataset._ys[valid, i] >= 0.5).float()
        n_pos = ys.sum().item()
        n_neg = len(valid) - n_pos
        if n_pos > 0:
            weights[i] = min(n_neg / n_pos, _POS_WEIGHT_CAP)
    return weights


def _f1_per_predicate(logits: torch.Tensor, targets: torch.Tensor,
                      masks: torch.Tensor, threshold: float = 0.0,
                      per_pred: bool = False):
    """Macro F1 over valid positions.

    The model emits raw logits (no sigmoid in forward), so the decision
    boundary is logit >= 0.0  (equivalently sigmoid(logit) >= 0.5), NOT 0.5
    on the logit. Thresholding logits at 0.5 == prob 0.62, which silently
    under-counts positives — worst on the positive-dominant predicates.

    Returns the macro F1 (float). If per_pred=True, also returns a dict
    {predicate: (f1, n_valid)} — predicates with no valid pairs are omitted.
    """
    f1s = []
    breakdown: dict[str, tuple[float, int]] = {}
    for i in range(_N_PRED):
        valid = masks[:, i] > 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        preds = (logits[valid, i] >= threshold).float()
        # Soft VLM scores are kept for the BCE loss (distillation), but F1 is
        # an inherently binary notion, so threshold at 0.5 (the same boundary
        # used to derive `verdict`) for this computation only.
        tgts  = (targets[valid, i] >= 0.5).float()
        tp = (preds * tgts).sum().item()
        fp = (preds * (1 - tgts)).sum().item()
        fn = ((1 - preds) * tgts).sum().item()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        f1s.append(f1)
        breakdown[TIER1_PREDICATES[i]] = (f1, n_valid)
    macro = float(np.mean(f1s)) if f1s else 0.0
    return (macro, breakdown) if per_pred else macro


@dataclass
class Trainer:
    epochs:       int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    neg_ratio:    int   = 5
    batch_size:   int   = 64
    seed:         int   = 0
    patience:     int   = 15
    verbose:      bool  = True
    resolved_instances: Path | str | None = None
    model_factory: "Callable[[], object] | None" = None
    unbounded: bool = False
    """Train AffordanceDataset(unbounded=True): structurally impossible pairs
    become hard negatives instead of being excluded from the loss, so the
    model learns to actually reject them (e.g. knife-canCut-knife) rather
    than never seeing them. Eval macro-F1 stays scoped to the original
    candidate mask (dataset._candidate_masks) either way, so it remains
    comparable to non-unbounded runs; a separate rejection-rate diagnostic
    covers the new hard-negative region."""
    split_strategy: str = "scene"
    """Generalization level the train/val split is drawn at (E1 ladder):
    scene / pair / instance. Must match the Inferencer's, or "val" would
    mean different scenes during training and evaluation."""
    holdout_mode: str | None = None
    """Pair-level instance holdout ("strict" or "actor_only"), mirroring
    AffordanceDataset/Inferencer (E1c/E1c'). Takes precedence over
    split_strategy. MUST match the Inferencer's holdout_mode/holdout_frac/seed,
    or "test" there won't be the pairs this model actually held out."""
    holdout_frac: float = 0.20
    unbounded_neg_ratio: int | None = None
    """Only meaningful when unbounded=True. Caps hard-negative (impossible)
    pairs kept per scene to ratio x (real candidate pairs in that scene) --
    an uncapped hard-negative flood was observed to hurt predicates with few
    structural candidates (canCut, canPierce) by diluting their positive
    signal. See AffordanceDataset.unbounded_neg_ratio."""
    balanced: bool = False
    """Downsample the majority class per predicate to a true 50/50 split
    (AffordanceDataset.balance_masks). Orthogonal to pos_weight, which
    reweights the loss but leaves the DATA skewed (canSupport is ~89%
    positive). When on, pos_weight is skipped -- the sample is already
    balanced, so applying both would over-correct toward the minority class.
    TRAIN split only: val/test are left untouched so eval stays comparable
    to unbalanced runs."""

    def fit(self, scenes_dir: Path | str, features_dir: Path | str,
            labels_dir: Path | str, ckpt_dir: Path | str) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)

        train_ds = AffordanceDataset(scenes_dir, features_dir, labels_dir,
                                     split="train", seed=self.seed,
                                     split_strategy=self.split_strategy,
                                     holdout_mode=self.holdout_mode,
                                     holdout_frac=self.holdout_frac,
                                     resolved_instances=self.resolved_instances,
                                     unbounded=self.unbounded,
                                     unbounded_neg_ratio=self.unbounded_neg_ratio)
        # val stays uncapped (no ratio) -- the rejection-rate diagnostic wants
        # every impossible pair available to measure against, not a subsample.
        val_ds   = AffordanceDataset(scenes_dir, features_dir, labels_dir,
                                     split="val",   seed=self.seed,
                                     split_strategy=self.split_strategy,
                                     holdout_mode=self.holdout_mode,
                                     holdout_frac=self.holdout_frac,
                                     resolved_instances=self.resolved_instances,
                                     unbounded=self.unbounded)

        if self.balanced:
            # Balance the DATA, then leave the loss unweighted -- doing both
            # would over-correct past 50/50 toward the minority class.
            train_ds.balance_masks(seed=self.seed)
            pos_weights = torch.ones(_N_PRED)
        else:
            pos_weights = _compute_pos_weights(train_ds)
        # Feature width depends on the DINOv2 backbone (vits14=384/obj,
        # vitb14=768, vitl14=1024) => x = 2*feat + 8 geo. Derive it from the
        # data so the head matches the backbone knob instead of assuming vitb14.
        input_dim = int(train_ds._xs.shape[1]) if len(train_ds._xs) else 1544
        model = (self.model_factory(input_dim) if self.model_factory is not None
                 else AffordanceMLP(input_dim=input_dim))
        opt   = torch.optim.Adam(model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=5, factor=0.5
        )

        best_f1    = -1.0
        no_improve = 0
        best_breakdown: dict[str, tuple[float, int]] = {}

        if self.verbose:
            print(f"[train] {len(train_ds)} train / {len(val_ds)} val pairs, "
                  f"{self.epochs} epochs max, patience {self.patience}", flush=True)
            print(f"[train] pos_weights: " + ", ".join(
                f"{p}={pos_weights[i]:.3f}" for i, p in enumerate(TIER1_PREDICATES)), flush=True)

        for epoch in range(self.epochs):
            # --- train ---
            # NOTE: no resample(). Class imbalance is per-PREDICATE and handled
            # by per-predicate pos_weight in the loss below. resample() rebalanced
            # per-PAIR (pair positive if y.max()>0); since canSupport/canCut are
            # 85-98% positive, almost every pair counted as positive, so it kept
            # ~everything and was a no-op. Per-pair rebalancing is also wrong for
            # multi-label data (a pair is shared across heads). Train on all pairs.
            model.train()
            loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
            epoch_loss, n_batches = 0.0, 0
            for x, y, mask in loader:
                opt.zero_grad()
                scores = model(x)
                loss = torch.tensor(0.0)
                for i in range(_N_PRED):
                    valid = mask[:, i] > 0
                    if valid.sum() == 0:
                        continue
                    criterion = nn.BCEWithLogitsLoss(
                        pos_weight=torch.full((int(valid.sum()),), pos_weights[i])
                    )
                    loss = loss + criterion(scores[valid, i], y[valid, i])
                if loss.requires_grad:
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss); n_batches += 1

            # --- val ---
            model.eval()
            all_scores, all_ys = [], []
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)  # no shuffle -> _active_idx order
            with torch.no_grad():
                for x, y, mask in val_loader:
                    all_scores.append(model(x))
                    all_ys.append(y)
            if not all_scores:
                continue
            all_scores_t, all_ys_t = torch.cat(all_scores), torch.cat(all_ys)
            # F1 stays scoped to the original candidate mask (not _masks) so
            # it's comparable across unbounded and non-unbounded runs — the
            # extra hard-negative pairs unbounded=True adds are diagnosed
            # separately below, not folded into the headline metric.
            candidate_masks = val_ds._candidate_masks[val_ds._active_idx]
            val_f1, val_breakdown = _f1_per_predicate(
                all_scores_t, all_ys_t, candidate_masks, per_pred=True,
            )
            scheduler.step(val_f1)

            if self.unbounded and self.verbose:
                hard_neg = (val_ds._masks[val_ds._active_idx] > 0) & (candidate_masks == 0)
                if hard_neg.any():
                    rejected = (all_scores_t[hard_neg] < 0.0).float().mean().item()
                    print(f"epoch {epoch:3d}  impossible-pair rejection rate "
                          f"{rejected:.4f}  (n={int(hard_neg.sum())})", flush=True)

            avg_loss = epoch_loss / max(n_batches, 1)
            improved = val_f1 > best_f1
            if self.verbose:
                lr_now = opt.param_groups[0]["lr"]
                print(f"epoch {epoch:3d}  loss {avg_loss:.4f}  val_macroF1 {val_f1:.4f}"
                      f"  lr {lr_now:.1e}{'  *best' if improved else ''}", flush=True)

            if improved:
                best_f1    = val_f1
                best_breakdown = val_breakdown
                no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "val_f1":      val_f1,
                    "epoch":       epoch,
                    "config": {
                        "epochs":    self.epochs,
                        "lr":        self.lr,
                        "neg_ratio": self.neg_ratio,
                        "seed":      self.seed,
                    },
                }, ckpt_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"[train] early stop at epoch {epoch} "
                              f"(no val improvement for {self.patience})", flush=True)
                    break

        if self.verbose:
            print(f"\n[train] best val macro-F1 = {best_f1:.4f}", flush=True)
            print("[train] per-predicate val F1 (at best epoch):", flush=True)
            for p in TIER1_PREDICATES:
                if p in best_breakdown:
                    f1, n = best_breakdown[p]
                    print(f"    {p:14s} F1={f1:.3f}  (n_valid={n})", flush=True)
                else:
                    print(f"    {p:14s} —  (no candidate pairs)", flush=True)
