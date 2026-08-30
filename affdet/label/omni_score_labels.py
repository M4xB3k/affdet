"""Derive per-scene soft affordance labels from the VLM score resolver.

For each scene, for each (actor, recipient) pair and each candidate predicate
(gated by the intersection matrix on the resolver's EFFECTIVE class), emit a
LabelEntry carrying the soft score (VLM score/9, or 0/1 for canPourInto). Pairs
with no teacher signal are dropped and counted in label_coverage.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from affdet.schemas import Scene, Labels, LabelEntry
from affdet.intersection_matrix import TIER1_PREDICATES, is_candidate_pair
from affdet.label.vlm_resolver import VlmScoreResolver


def _find_scene_jsons(scenes_dir: Path) -> list[Path]:
    for pat in ("scene_*.json", "scene_*/scene_*.json",
                "worker_*/scene_*/scene_*.json"):
        found = sorted(scenes_dir.glob(pat))
        if found:
            return found
    return []


def derive_scene_labels(scenes_dir: Path | str, labels_dir: Path | str,
                        resolver: VlmScoreResolver) -> dict:
    scenes_dir = Path(scenes_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    n_dropped = 0
    per_pred: dict[str, int] = defaultdict(int)

    for scene_json in _find_scene_jsons(scenes_dir):
        scene = Scene.model_validate_json(scene_json.read_text())
        entries: list[LabelEntry] = []
        for actor in scene.objects:
            a_eff = resolver.effective_class(actor.obj_id, actor.obj_class)
            for recip in scene.objects:
                if actor.obj_id == recip.obj_id:
                    continue
                for pred in TIER1_PREDICATES:
                    if not is_candidate_pair(a_eff, recip.obj_class, pred):
                        continue
                    t = resolver.soft_target(actor.obj_id, actor.obj_class,
                                             recip.obj_class, pred,
                                             recip_id=recip.obj_id)
                    if t is None:
                        n_dropped += 1
                        continue
                    entries.append(LabelEntry(
                        actor=actor.obj_id, recipient=recip.obj_id,
                        predicate=pred, verdict=1 if t >= 0.5 else 0,
                        score=t, confidence=t, source="vlm_distill_v1",
                    ))
                    per_pred[pred] += 1
                    n_emitted += 1

        (labels_dir / f"{scene.scene_id}.json").write_text(
            Labels(scene_id=scene.scene_id, tier1=entries).model_dump_json(indent=2))

    coverage = {"n_emitted": n_emitted, "n_dropped": n_dropped,
                "per_predicate": dict(per_pred)}
    (labels_dir / "label_coverage.json").write_text(json.dumps(coverage, indent=2))
    return coverage
