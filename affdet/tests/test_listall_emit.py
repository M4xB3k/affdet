"""Emitting list-all labels in the per-scene shape the trainer consumes.

`labels_all_merged_v4/` is one JSON per scene with a `tier1` list, and its
`score` is normalised to 0-1 as raw/9 -- NOT the 0-10 the VLM emits. Getting
that scale wrong would silently shrink every training target by ~10x and the
model would look broken for reasons unrelated to the labels.
"""
import json

import pytest

from affdet.eval.listall_emit import emit_scene_labels, to_unit_score


def test_score_is_normalised_by_nine():
    """v4 uses raw/9: a 2 becomes 0.2222, matching the observed corpus."""
    assert to_unit_score(2) == pytest.approx(2 / 9)
    assert to_unit_score(9) == pytest.approx(1.0)
    assert to_unit_score(0) == 0.0


def test_score_of_ten_is_clamped_to_one():
    """The VLM's scale runs to 10 but v4's normaliser tops out at 1.0."""
    assert to_unit_score(10) == 1.0


def test_verdict_threshold_matches_the_corpus_convention():
    scenes = emit_scene_labels(
        {("s1", "knife_1", "apple_1", "canCut"): 8.0,
         ("s1", "knife_1", "apple_1", "canPierce"): 2.0})
    by = {e["predicate"]: e for e in scenes["s1"]["tier1"]}
    assert by["canCut"]["verdict"] == 1
    assert by["canPierce"]["verdict"] == 0


def test_emitted_rows_carry_the_v4_fields():
    scenes = emit_scene_labels({("s1", "knife_1", "apple_1", "canCut"): 8.0})
    row = scenes["s1"]["tier1"][0]
    for k in ("actor", "recipient", "predicate", "verdict", "actor_part",
              "confidence", "source", "score", "rationale"):
        assert k in row


def test_scene_id_is_carried_and_grouping_is_per_scene():
    scenes = emit_scene_labels({
        ("s1", "knife_1", "apple_1", "canCut"): 8.0,
        ("s2", "bowl_1", "apple_1", "canContain"): 9.0,
    })
    assert set(scenes) == {"s1", "s2"}
    assert scenes["s1"]["scene_id"] == "s1"
    assert len(scenes["s1"]["tier1"]) == 1


def test_source_marks_the_listall_provenance():
    """So a ledger entry can tell which teacher produced a checkpoint."""
    scenes = emit_scene_labels({("s1", "a", "b", "canCut"): 5.0})
    assert "listall" in scenes["s1"]["tier1"][0]["source"]


def test_tier2_placeholder_present_for_schema_parity():
    scenes = emit_scene_labels({("s1", "a", "b", "canCut"): 5.0})
    assert "tier2_placeholder" in scenes["s1"]


def test_empty_input_emits_nothing():
    assert emit_scene_labels({}) == {}


def test_rows_are_sorted_deterministically():
    scores = {("s1", "b_1", "x_1", "canCut"): 5.0,
              ("s1", "a_1", "x_1", "canCut"): 5.0}
    a = emit_scene_labels(scores)["s1"]["tier1"]
    b = emit_scene_labels(scores)["s1"]["tier1"]
    assert [r["actor"] for r in a] == [r["actor"] for r in b]
    assert [r["actor"] for r in a] == sorted(r["actor"] for r in a)


def test_written_files_are_valid_json_per_scene(tmp_path):
    from affdet.eval.listall_emit import write_label_dir
    n = write_label_dir(
        {("s1", "knife_1", "apple_1", "canCut"): 8.0}, tmp_path)
    assert n == 1
    d = json.loads((tmp_path / "s1.json").read_text())
    assert d["scene_id"] == "s1"
    assert d["tier1"][0]["score"] == pytest.approx(8 / 9)
