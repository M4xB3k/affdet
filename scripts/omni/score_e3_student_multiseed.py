"""Multi-seed: score the 146 E3 pairs against all 8 e1c-actor_only-v4fix seeds.

Each seed draws its OWN mesh holdout (instance_holdout resamples per seed --
see dataset.py's instance_holdout docstring), so a pair's split membership
("train"/"val"/"test") is seed-dependent, not just the model weights. This
scores each pair against every seed's checkpoint using that seed's own
holdout draw, so the "is this pair in the test split" question is answered
correctly per seed rather than reusing seed 0's draw for all of them.

Not a reusable pipeline module -- one-shot, cluster-only. See
docs/HANDOFF_E3_BBOX_RERENDER.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path("/home/maksym.bek/affdet")
sys.path.insert(0, str(REPO_ROOT))

from affdet.extract.dinov2 import _build_backbone, _pad_bbox, _preprocess
from affdet.intersection_matrix import TIER1_PREDICATES, is_candidate_pair
from affdet.label.resolved_instances import (
    effective_class, load_dropped_set, load_effective_class_map,
)
from affdet.model.dataset import _geo, instance_holdout, pair_split_for
from affdet.model.infer import _infer_input_dim
from affdet.model.mlp import AffordanceMLP

_PRED_IDX = {p: i for i, p in enumerate(TIER1_PREDICATES)}

RENDERS_DIR = Path("/scratch/maksym.bek/affdet/pair_renders_bbox_v2")
COMPARISON_PATH = REPO_ROOT / "data/eval/e3_full_147pair_comparison.json"
SCENES_DIR = Path("/scratch/maksym.bek/omni_scenes_all_merged")
RESOLVED_INSTANCES = Path("/scratch/maksym.bek/affdet/resolved_instances_v2.json")
OUT_PATH = REPO_ROOT / "data/eval/e3_full_147pair_comparison_with_student_v4_8seed.json"
HOLDOUT_FRAC = 0.20
HOLDOUT_MODE = "actor_only"
SEEDS = list(range(8))

RUN_DIRS = {
    0: "e1c-actor_only-s0-v4fix-5b02adc6e725-f0d41e5",
    1: "e1c-actor_only-s1-v4fix-5b02adc6e725-f0d41e5",
    2: "e1c-actor_only-s2-v4fix-5b02adc6e725-f0d41e5",
    3: "e1c-actor_only-s3-v4fix-5b02adc6e725-2f47961",
    4: "e1c-actor_only-s4-v4fix-5b02adc6e725-2f47961",
    5: "e1c-actor_only-s5-v4fix-5b02adc6e725-2f47961",
    # filled in after seeds 6-7 finish training -- see fill_seed67_rundirs()
}


def fill_seed67_rundirs():
    """Seeds 6-7 were submitted after this script was drafted; their run dir
    names (which embed a git sha) aren't known until the job actually runs.
    Discover them by globbing rather than hardcoding a guess."""
    base = Path("/scratch/maksym.bek/affdet/experiments/runs")
    for seed in (6, 7):
        matches = sorted(base.glob(f"e1c-actor_only-s{seed}-v4fix-*"))
        if not matches:
            raise FileNotFoundError(f"no run dir found for seed {seed} yet")
        RUN_DIRS[seed] = matches[-1].name


def load_model(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu")
    input_dim = _infer_input_dim(state["model_state"])
    model = AffordanceMLP(input_dim=input_dim)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def class_of(obj_id: str) -> str:
    return obj_id.rsplit("_", 1)[0]


def score_pair(model, backbone, device, actor_id: str, recipient_id: str,
               eff_map: dict[str, str]) -> dict[str, float] | None:
    pair_dir = RENDERS_DIR / f"{actor_id}__{recipient_id}"
    manifest_path = pair_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    bboxes = manifest["bboxes"][0]
    if actor_id not in bboxes or recipient_id not in bboxes:
        return None

    W, H = manifest["resolution"]
    img = Image.open(pair_dir / manifest["angle_files"][0]).convert("RGB")

    crops = torch.stack([
        _preprocess(img.crop(tuple(_pad_bbox(bboxes[oid], W, H, 0.0))))
        for oid in (actor_id, recipient_id)
    ]).to(device)

    with torch.no_grad():
        cls_tokens = backbone.forward_features(crops)["x_norm_clstoken"]
    f_actor, f_recipient = cls_tokens.cpu().float().numpy()

    class Obj:
        def __init__(self, bbox):
            self.bbox = bbox
    geo = _geo(Obj(bboxes[actor_id]), Obj(bboxes[recipient_id]), float(W), float(H))

    x = np.concatenate([f_actor, f_recipient, np.array(geo, dtype="float32")])
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype("float32")).unsqueeze(0))[0]
        probs = torch.sigmoid(logits)

    actor_cls = effective_class(actor_id, class_of(actor_id), eff_map)
    recipient_cls = class_of(recipient_id)
    scores = {}
    for p in TIER1_PREDICATES:
        if is_candidate_pair(actor_cls, recipient_cls, p):
            scores[p] = float(probs[_PRED_IDX[p]])
    return scores


def main():
    fill_seed67_rundirs()
    print("run dirs:", RUN_DIRS)

    comparison = json.loads(COMPARISON_PATH.read_text())
    pairs = sorted({(r["actor"], r["recipient"]) for r in comparison})
    print(f"{len(comparison)} rows, {len(pairs)} unique pairs, {len(SEEDS)} seeds")

    eff_map = load_effective_class_map(RESOLVED_INSTANCES)
    dropped = load_dropped_set(RESOLVED_INSTANCES)

    print("building shared DINOv2 backbone (reused across all seeds)...")
    backbone = _build_backbone("vitb14")
    device = next(backbone.parameters()).device

    # Feature extraction is seed-independent -- cache it once per pair instead
    # of recomputing DINOv2 features 8x for the same crops.
    print("extracting DINOv2 features once per pair (cached across seeds)...")
    feat_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[float], int, int] | None] = {}
    for actor_id, recipient_id in pairs:
        pair_dir = RENDERS_DIR / f"{actor_id}__{recipient_id}"
        manifest_path = pair_dir / "manifest.json"
        if not manifest_path.exists():
            feat_cache[(actor_id, recipient_id)] = None
            continue
        manifest = json.loads(manifest_path.read_text())
        bboxes = manifest["bboxes"][0]
        if actor_id not in bboxes or recipient_id not in bboxes:
            feat_cache[(actor_id, recipient_id)] = None
            continue
        W, H = manifest["resolution"]
        img = Image.open(pair_dir / manifest["angle_files"][0]).convert("RGB")
        crops = torch.stack([
            _preprocess(img.crop(tuple(_pad_bbox(bboxes[oid], W, H, 0.0))))
            for oid in (actor_id, recipient_id)
        ]).to(device)
        with torch.no_grad():
            cls_tokens = backbone.forward_features(crops)["x_norm_clstoken"]
        f_actor, f_recipient = cls_tokens.cpu().float().numpy()

        class Obj:
            def __init__(self, bbox):
                self.bbox = bbox
        geo = _geo(Obj(bboxes[actor_id]), Obj(bboxes[recipient_id]), float(W), float(H))
        feat_cache[(actor_id, recipient_id)] = (f_actor, f_recipient, geo, W, H)
    n_feat_missing = sum(1 for v in feat_cache.values() if v is None)
    print(f"features cached for {len(pairs) - n_feat_missing}/{len(pairs)} pairs "
          f"({n_feat_missing} missing render)")

    per_seed_scores: dict[int, dict[tuple[str, str], dict[str, float]]] = {}
    per_seed_split: dict[int, dict[tuple[str, str], str | None]] = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        ckpt_path = (Path("/scratch/maksym.bek/affdet/experiments/runs")
                     / RUN_DIRS[seed] / "ckpt" / "best.pt")
        model = load_model(ckpt_path)

        holdout = instance_holdout(
            SCENES_DIR, seed=seed, frac=HOLDOUT_FRAC, effective_classes=eff_map,
        )
        print(f"  holdout: {len(holdout.test_meshes)} test meshes, "
              f"{len(holdout.val_meshes)} val meshes")

        scores_this_seed: dict[tuple[str, str], dict[str, float]] = {}
        split_this_seed: dict[tuple[str, str], str | None] = {}
        for actor_id, recipient_id in pairs:
            key = (actor_id, recipient_id)
            if actor_id in dropped or recipient_id in dropped:
                split_this_seed[key] = None
                continue
            split_this_seed[key] = pair_split_for(
                actor_id, recipient_id, holdout, mode=HOLDOUT_MODE,
            )
            cached = feat_cache.get(key)
            if cached is None:
                continue
            f_actor, f_recipient, geo, W, H = cached
            x = np.concatenate([f_actor, f_recipient, np.array(geo, dtype="float32")])
            with torch.no_grad():
                logits = model(torch.from_numpy(x.astype("float32")).unsqueeze(0))[0]
                probs = torch.sigmoid(logits)
            actor_cls = effective_class(actor_id, class_of(actor_id), eff_map)
            recipient_cls = class_of(recipient_id)
            scores = {
                p: float(probs[_PRED_IDX[p]])
                for p in TIER1_PREDICATES
                if is_candidate_pair(actor_cls, recipient_cls, p)
            }
            scores_this_seed[key] = scores

        per_seed_scores[seed] = scores_this_seed
        per_seed_split[seed] = split_this_seed

        split_counts: dict[str, int] = {}
        for v in split_this_seed.values():
            k = v if v is not None else "excluded"
            split_counts[k] = split_counts.get(k, 0) + 1
        print(f"  split membership: {split_counts}")

    out_rows = []
    for r in comparison:
        key = (r["actor"], r["recipient"])
        pred = r["predicate"]
        row = dict(r)
        per_seed = {}
        for seed in SEEDS:
            scores = per_seed_scores[seed].get(key)
            per_seed[str(seed)] = {
                "student_score_0to1": scores.get(pred) if scores is not None else None,
                "split_membership": per_seed_split[seed].get(key),
            }
        row["per_seed"] = per_seed
        test_scores = [
            v["student_score_0to1"] for v in per_seed.values()
            if v["split_membership"] == "test" and v["student_score_0to1"] is not None
        ]
        row["student_score_test_mean"] = float(np.mean(test_scores)) if test_scores else None
        row["student_score_test_std"] = float(np.std(test_scores)) if len(test_scores) > 1 else None
        row["n_test_seeds"] = len(test_scores)
        out_rows.append(row)

    OUT_PATH.write_text(json.dumps(out_rows, indent=2))
    print(f"\nwrote {OUT_PATH}")

    n_any_test = sum(1 for r in out_rows if r["n_test_seeds"] > 0)
    print(f"rows with >=1 seed's test membership + score: {n_any_test}/{len(out_rows)}")


if __name__ == "__main__":
    main()
