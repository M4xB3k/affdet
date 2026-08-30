"""make_pair_list.py --holdout-json — read an E1c/E1c' InstanceHoldout record
(test_meshes/val_meshes/all_meshes, as recorded in the experiment ledger)
directly, instead of requiring a separately-built flat instance->split map.

Without this, real_run() only accepts --split-map's flat {instance: split}
shape, but the actual thing produced by a holdout_mode run
(runner.py's `record["holdout"]`, via InstanceHoldout.to_dict()) is
{"test_meshes": [...], "val_meshes": [...], "all_meshes": [...]}. The E3
pair sample needs to draw from exactly the same test meshes the model was
evaluated on, or a human-labeled pair may have no matching model prediction
to compare against.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.make_pair_list import holdout_json_to_split_map


def test_test_meshes_map_to_test_split():
    holdout = {"test_meshes": ["knife_001", "apple_003"],
               "val_meshes": ["bowl_002"],
               "all_meshes": ["knife_001", "apple_003", "bowl_002", "bowl_005"]}
    split_map = holdout_json_to_split_map(holdout)
    assert split_map["knife_001"] == "test"
    assert split_map["apple_003"] == "test"


def test_val_meshes_map_to_val_split():
    holdout = {"test_meshes": ["knife_001"],
               "val_meshes": ["bowl_002"],
               "all_meshes": ["knife_001", "bowl_002", "bowl_005"]}
    split_map = holdout_json_to_split_map(holdout)
    assert split_map["bowl_002"] == "val"


def test_remaining_meshes_map_to_train():
    """all_meshes minus (test_meshes | val_meshes) = train -- the meshes the
    model actually trained on, needed so real_run()'s existing test_instances
    filter (which only checks membership, not the full 3-way split) still
    works correctly against this map."""
    holdout = {"test_meshes": ["knife_001"],
               "val_meshes": ["bowl_002"],
               "all_meshes": ["knife_001", "bowl_002", "bowl_005"]}
    split_map = holdout_json_to_split_map(holdout)
    assert split_map["bowl_005"] == "train"


def test_reads_ledger_style_file(tmp_path):
    """The real input is a ledger record's `holdout` field written to its own
    JSON file, not a hand-built dict."""
    p = tmp_path / "holdout.json"
    p.write_text(json.dumps({"test_meshes": ["knife_001"],
                             "val_meshes": [],
                             "all_meshes": ["knife_001", "bowl_005"]}))
    from scripts.make_pair_list import load_holdout_split_map
    split_map = load_holdout_split_map(p)
    assert split_map["knife_001"] == "test"
    assert split_map["bowl_005"] == "train"
