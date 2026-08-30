"""Tests for the E3 re-render verification tool (affdet/label/rerender_verify.py).

Unlike manual_tool.py (which must stay blind to VLM/model scores during a
real labelling session), this tool exists to answer a narrower question after
the fact: does a freshly re-rendered pair image still show the same scene a
human already judged? So it deliberately DOES show the human's own score next
to the VLM's score -- there is no fresh judgment being collected, only a
match/mismatch/unsure verdict on image identity.
"""
from __future__ import annotations

import json

import pytest

from affdet.label.rerender_verify import (
    append_verify_record, build_items, build_verify_payload,
    find_old_render_dir, group_judgments_by_pair, load_verify_session,
    make_verify_record, remaining_verify_items,
)


def test_find_old_render_dir_returns_first_existing_match(tmp_path):
    d1 = tmp_path / "renders_a"
    d2 = tmp_path / "renders_b"
    (d1 / "knife_002__litchi_001").mkdir(parents=True)
    (d2 / "knife_002__litchi_001").mkdir(parents=True)
    (d1 / "knife_002__litchi_001" / "angle_00.png").write_bytes(b"fake")

    found = find_old_render_dir([d1, d2], "knife_002", "litchi_001")

    assert found == d1 / "knife_002__litchi_001"


def test_find_old_render_dir_skips_dirs_without_the_pair(tmp_path):
    d1 = tmp_path / "renders_a"
    d2 = tmp_path / "renders_b"
    d1.mkdir()
    (d2 / "knife_002__litchi_001").mkdir(parents=True)
    (d2 / "knife_002__litchi_001" / "angle_00.png").write_bytes(b"fake")

    found = find_old_render_dir([d1, d2], "knife_002", "litchi_001")

    assert found == d2 / "knife_002__litchi_001"


def test_find_old_render_dir_returns_none_when_missing_everywhere(tmp_path):
    d1 = tmp_path / "renders_a"
    d1.mkdir()

    found = find_old_render_dir([d1], "knife_002", "litchi_001")

    assert found is None


# --------------------------------------------------------------------------
# Grouping comparison rows by pair -- one pair can carry multiple predicates
# --------------------------------------------------------------------------

def test_group_judgments_by_pair_groups_same_pair_different_predicates():
    rows = [
        {"actor": "knife_002", "recipient": "litchi_001", "predicate": "canCut",
         "human_score_0to9": 7, "vlm_score_0to9": 6.5},
        {"actor": "knife_002", "recipient": "litchi_001", "predicate": "canPierce",
         "human_score_0to9": 3, "vlm_score_0to9": 2.0},
        {"actor": "bowl_002", "recipient": "durian_001", "predicate": "canContain",
         "human_score_0to9": 8, "vlm_score_0to9": 7.0},
    ]

    grouped = group_judgments_by_pair(rows)

    assert set(grouped.keys()) == {("knife_002", "litchi_001"), ("bowl_002", "durian_001")}
    assert len(grouped[("knife_002", "litchi_001")]) == 2
    assert grouped[("knife_002", "litchi_001")][0]["predicate"] == "canCut"
    assert grouped[("knife_002", "litchi_001")][0]["human_score_0to9"] == 7
    assert grouped[("bowl_002", "durian_001")][0]["vlm_score_0to9"] == 7.0


def test_group_judgments_by_pair_preserves_row_order_within_a_pair():
    rows = [
        {"actor": "a", "recipient": "b", "predicate": "canCut", "human_score_0to9": 1, "vlm_score_0to9": 1.0},
        {"actor": "a", "recipient": "b", "predicate": "canPeel", "human_score_0to9": 2, "vlm_score_0to9": 2.0},
        {"actor": "a", "recipient": "b", "predicate": "canPierce", "human_score_0to9": 3, "vlm_score_0to9": 3.0},
    ]

    grouped = group_judgments_by_pair(rows)

    assert [r["predicate"] for r in grouped[("a", "b")]] == ["canCut", "canPeel", "canPierce"]


# --------------------------------------------------------------------------
# Building the per-pair review items the server hands to the page
# --------------------------------------------------------------------------

def _make_render_tree(base, name, angles=("angle_00.png",)):
    d = base / name
    d.mkdir(parents=True)
    for a in angles:
        (d / a).write_bytes(b"fake")
    return d


def test_build_items_joins_old_new_and_judgments(tmp_path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    _make_render_tree(old_dir, "knife_002__litchi_001")
    _make_render_tree(new_dir, "knife_002__litchi_001")

    pairs = [{"actor": "knife_002", "recipient": "litchi_001"}]
    judgments = [
        {"actor": "knife_002", "recipient": "litchi_001", "predicate": "canCut",
         "human_score_0to9": 7, "vlm_score_0to9": 6.5},
    ]

    items = build_items(pairs, judgments, old_search_dirs=[old_dir], new_render_dir=new_dir)

    assert len(items) == 1
    item = items[0]
    assert item["pair_id"] == "knife_002__litchi_001"
    assert item["actor"] == "knife_002"
    assert item["recipient"] == "litchi_001"
    assert item["old_dir"] == old_dir / "knife_002__litchi_001"
    assert item["new_dir"] == new_dir / "knife_002__litchi_001"
    assert item["judgments"][0]["predicate"] == "canCut"
    assert item["judgments"][0]["human_score_0to9"] == 7


def test_build_items_flags_missing_old_render(tmp_path):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    new_dir = tmp_path / "new"
    _make_render_tree(new_dir, "knife_002__litchi_001")

    pairs = [{"actor": "knife_002", "recipient": "litchi_001"}]
    items = build_items(pairs, [], old_search_dirs=[old_dir], new_render_dir=new_dir)

    assert items[0]["old_dir"] is None


def test_build_items_flags_missing_new_render(tmp_path):
    old_dir = tmp_path / "old"
    _make_render_tree(old_dir, "knife_002__litchi_001")
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    pairs = [{"actor": "knife_002", "recipient": "litchi_001"}]
    items = build_items(pairs, [], old_search_dirs=[old_dir], new_render_dir=new_dir)

    assert items[0]["new_dir"] is None


def test_build_items_pair_with_no_judgments_gets_empty_list(tmp_path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    _make_render_tree(old_dir, "a__b")
    _make_render_tree(new_dir, "a__b")

    pairs = [{"actor": "a", "recipient": "b"}]
    items = build_items(pairs, [], old_search_dirs=[old_dir], new_render_dir=new_dir)

    assert items[0]["judgments"] == []


# --------------------------------------------------------------------------
# Records -- match / mismatch / unsure verdicts, not fresh 0-9 judgments
# --------------------------------------------------------------------------

def test_make_verify_record_carries_verdict_and_pair_id():
    rec = make_verify_record("knife_002__litchi_001", "mismatch", note="pose changed")
    assert rec["pair_id"] == "knife_002__litchi_001"
    assert rec["verdict"] == "mismatch"
    assert rec["note"] == "pose changed"
    assert "timestamp_utc" in rec


def test_make_verify_record_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        make_verify_record("p1", "definitely", note="")


@pytest.mark.parametrize("verdict", ["match", "mismatch", "unsure"])
def test_make_verify_record_accepts_the_three_verdicts(verdict):
    rec = make_verify_record("p1", verdict, note="")
    assert rec["verdict"] == verdict


def test_append_then_load_verify_session_roundtrips(tmp_path):
    out = tmp_path / "session.jsonl"
    append_verify_record(out, make_verify_record("p1", "match", note=""))
    append_verify_record(out, make_verify_record("p2", "mismatch", note="pose"))

    recs = load_verify_session(out)

    assert [r["pair_id"] for r in recs] == ["p1", "p2"]
    assert recs[1]["verdict"] == "mismatch"


def test_load_verify_session_of_missing_file_is_empty(tmp_path):
    assert load_verify_session(tmp_path / "nope.jsonl") == []


def test_remaining_verify_items_skips_already_judged(tmp_path):
    items = [{"pair_id": "p1"}, {"pair_id": "p2"}, {"pair_id": "p3"}]
    out = tmp_path / "session.jsonl"
    append_verify_record(out, make_verify_record("p1", "match", note=""))

    remaining = remaining_verify_items(items, load_verify_session(out))

    assert [i["pair_id"] for i in remaining] == ["p2", "p3"]


# --------------------------------------------------------------------------
# Payload -- what the page actually receives, image paths kept relative
# --------------------------------------------------------------------------

def test_build_verify_payload_marks_missing_renders(tmp_path):
    items = [{
        "pair_id": "a__b", "actor": "a", "recipient": "b",
        "old_dir": None, "new_dir": tmp_path / "new" / "a__b",
        "judgments": [],
    }]

    payload = build_verify_payload(items, done=0, total=1)

    assert payload["items"][0]["old_available"] is False
    assert payload["items"][0]["new_available"] is True
    assert payload["done"] == 0
    assert payload["total"] == 1


def test_build_verify_payload_carries_judgments_through():
    items = [{
        "pair_id": "a__b", "actor": "a", "recipient": "b",
        "old_dir": None, "new_dir": None,
        "judgments": [{"predicate": "canCut", "human_score_0to9": 7, "vlm_score_0to9": 6.5}],
    }]

    payload = build_verify_payload(items, done=0, total=1)

    assert payload["items"][0]["judgments"] == [
        {"predicate": "canCut", "human_score_0to9": 7, "vlm_score_0to9": 6.5}
    ]
