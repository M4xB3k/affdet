"""Convert omni_label scores embedded in scene JSONs into Labels artifacts.

Reads every scene_XXXX.json that has an `omni_label` block, applies a score
threshold to produce a binary verdict, and writes a matching scene_XXXX.json
into the labels directory (Labels schema).

Usage:
    python -m affdet.scripts.emit_labels \
        --scenes data/scenes/omni \
        --out data/labels/scenes \
        --threshold 6
"""
from __future__ import annotations

import json
from pathlib import Path

from affdet.schemas import Labels, LabelEntry


def emit_scene_labels(
    scenes_dir: Path,
    labels_dir: Path,
    threshold: int = 6,
) -> int:
    """Write one Labels JSON per scene that has an omni_label block.

    Returns the number of label files written.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    scene_jsons = sorted(scenes_dir.glob("scene_*/scene_*.json"))
    if not scene_jsons:
        scene_jsons = sorted(scenes_dir.glob("scene_*.json"))

    for scene_json in scene_jsons:
        out_path = labels_dir / scene_json.name
        if out_path.exists():
            continue

        raw = json.loads(scene_json.read_text())
        omni_label = raw.get("omni_label")
        if omni_label is None:
            continue

        scene_id = raw["scene_id"]
        actor_id = omni_label["actor"]       # e.g. "knife_005"
        food_id  = omni_label["food"]        # e.g. "apple_001"
        affordance = omni_label["affordance"]
        score = omni_label["score"]
        verdict = 1 if score >= threshold else 0

        # food_id is the instance id; recipient key for the label uses obj_id
        # which matches instance_id written by omni_blender
        entry = LabelEntry(
            actor=actor_id,
            recipient=food_id,
            predicate=affordance,
            verdict=verdict,
            actor_part=None,
            confidence=score / 10.0,
            source=f"vlm_gpt4o_mini_t{threshold}",
        )

        labels_obj = Labels(scene_id=scene_id, tier1=[entry])
        out_path.write_text(labels_obj.model_dump_json(indent=2))
        written += 1

    return written
