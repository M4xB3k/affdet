"""make_pair_list.py real_run() must read the ACTUAL label corpus shape:
labels_all_merged_v3 is a DIRECTORY of per-scene {scene_id, tier1: [...]}
files (the affdet.schemas.Labels shape Inferencer/AffordanceDataset use), not
a flat JSONL of {actor, recipient, predicate, score} rows. Without this,
--labels pointed at the real corpus crashes with IsADirectoryError.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.make_pair_list import load_labels


def test_loads_flat_jsonl_file(tmp_path):
    """Existing behaviour must not regress: a flat JSONL of label rows."""
    p = tmp_path / "labels.jsonl"
    p.write_text(
        json.dumps({"actor": "bowl_001", "recipient": "apple_001",
                    "predicate": "canContain", "score": 0.8}) + "\n"
    )
    labels = load_labels(p)
    assert labels[("bowl_001", "apple_001", "canContain")] == 0.8


def test_loads_directory_of_per_scene_label_files(tmp_path):
    """The real corpus: one Labels-schema JSON per scene, key = scene_id,
    tier1 = [{actor, recipient, predicate, score, ...}]."""
    labels_dir = tmp_path / "labels_all_merged_v3"
    labels_dir.mkdir()
    (labels_dir / "scene_v1_0001.json").write_text(json.dumps({
        "scene_id": "scene_v1_0001",
        "tier1": [
            {"actor": "bowl_016", "recipient": "sausage_054",
             "predicate": "canContain", "verdict": 0, "score": 0.22,
             "confidence": 0.22, "source": "vlm_distill_v1"},
            {"actor": "knife_003", "recipient": "sausage_054",
             "predicate": "canCut", "verdict": 1, "score": 1.0,
             "confidence": 1.0, "source": "vlm_distill_v1"},
        ],
    }))
    labels = load_labels(labels_dir)
    assert labels[("bowl_016", "sausage_054", "canContain")] == 0.22
    assert labels[("knife_003", "sausage_054", "canCut")] == 1.0


def test_directory_skips_non_label_files(tmp_path):
    """label_coverage.json (a corpus-wide summary, not a per-scene label
    file) sits alongside the real per-scene files and has no `tier1` key --
    must be skipped, not crash the whole load."""
    labels_dir = tmp_path / "labels_all_merged_v3"
    labels_dir.mkdir()
    (labels_dir / "label_coverage.json").write_text(json.dumps(
        {"n_emitted": 59779, "n_dropped": 3428}))
    (labels_dir / "scene_v1_0001.json").write_text(json.dumps({
        "scene_id": "scene_v1_0001",
        "tier1": [{"actor": "bowl_016", "recipient": "sausage_054",
                   "predicate": "canContain", "verdict": 0, "score": 0.22,
                   "confidence": 0.22, "source": "vlm_distill_v1"}],
    }))
    labels = load_labels(labels_dir)
    assert labels[("bowl_016", "sausage_054", "canContain")] == 0.22
