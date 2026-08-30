"""Diagnostics for an all_pairs=True (ungated) eval run: per predicate, how
many extra (structurally-impossible) pairs the ungated eval adds over the
gated candidate set, and the model's rejection rate on them."""
from __future__ import annotations
from pathlib import Path

from affdet.intersection_matrix import TIER1_PREDICATES, is_candidate_pair
from affdet.label.resolved_instances import load_effective_class_map, effective_class
from affdet.schemas import Scene, Predictions


def _scene_path_by_name(scenes_dir: Path) -> dict[str, Path]:
    """Map scene filename -> path, across every layout scenes are stored in
    (flat scene_*.json, scene_*/scene_*.json, worker_*/scene_*/scene_*.json).
    Predictions are always written flat (predictions_dir/<scene>.json,
    matching Inferencer), so a flat name lookup is required regardless of how
    the source scenes are nested."""
    by_name: dict[str, Path] = {}
    for pat in ("scene_*.json", "scene_*/scene_*.json",
                "worker_*/scene_*/scene_*.json"):
        for p in scenes_dir.glob(pat):
            by_name.setdefault(p.name, p)
    return by_name


def report_all_pairs_diagnostics(
    predictions_dir: Path | str, scenes_dir: Path | str,
    resolved_instances: Path | str | None = None,
) -> dict[str, dict[str, float | int | None]]:
    predictions_dir = Path(predictions_dir); scenes_dir = Path(scenes_dir)
    eff_map = (load_effective_class_map(resolved_instances)
               if resolved_instances is not None else None)
    scene_by_name = _scene_path_by_name(scenes_dir)

    extra_pairs = {p: 0 for p in TIER1_PREDICATES}
    rejected = {p: 0 for p in TIER1_PREDICATES}

    for pred_json in sorted(predictions_dir.glob("*.json")):
        scene_json = scene_by_name.get(pred_json.name)
        if scene_json is None:
            continue
        scene = Scene.model_validate_json(scene_json.read_text())
        class_by_id = {o.obj_id: effective_class(o.obj_id, o.obj_class, eff_map)
                       for o in scene.objects}
        preds = Predictions.model_validate_json(pred_json.read_text())
        for entry in preds.tier1:
            actor_class = class_by_id.get(entry.actor)
            recip_class = class_by_id.get(entry.recipient)
            if actor_class is None or recip_class is None:
                continue
            if is_candidate_pair(actor_class, recip_class, entry.predicate):
                continue
            extra_pairs[entry.predicate] += 1
            if entry.verdict == 0:
                rejected[entry.predicate] += 1

    return {
        p: {
            "extra_pairs": extra_pairs[p],
            "rejection_rate": (rejected[p] / extra_pairs[p]
                                if extra_pairs[p] > 0 else None),
        }
        for p in TIER1_PREDICATES
    }
