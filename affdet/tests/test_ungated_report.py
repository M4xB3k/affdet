"""Per-predicate diagnostics for an all_pairs=True (ungated) eval run: how
many extra (structurally-impossible) pairs the ungated eval adds over the
gated candidate set, and the model's rejection rate on them."""
from pathlib import Path

from affdet.eval.ungated_report import report_all_pairs_diagnostics
from affdet.schemas import (
    Scene, SceneObject, Camera, Predictions, PredictionEntry,
)


def _camera():
    return Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                  intrinsics=[[500, 0, 320], [0, 500, 240], [0, 0, 1]])


def _scene(scene_id, objects):
    return Scene(scene_id=scene_id, image_path="rgb.png", image_size=[640, 480],
                 source="blender", objects=objects, camera=_camera())


def _obj(obj_id, obj_class):
    return SceneObject(obj_id=obj_id, obj_class=obj_class, bbox=[0, 0, 1, 1])


def _write_scene_and_preds(tmp_path, scene_id, objects, entries):
    scenes = tmp_path / "scenes"; scenes.mkdir(exist_ok=True)
    preds = tmp_path / "pred"; preds.mkdir(exist_ok=True)
    (scenes / f"{scene_id}.json").write_text(
        _scene(scene_id, objects).model_dump_json())
    (preds / f"{scene_id}.json").write_text(
        Predictions(scene_id=scene_id, tier1=entries).model_dump_json())
    return scenes, preds


def test_counts_zero_extra_pairs_when_every_prediction_is_a_candidate(tmp_path):
    """knife -> tomato, canCut is a real candidate pair (gated): no extra
    pairs should be attributed to canCut."""
    objects = [_obj("knife_1", "knife"), _obj("tomato_1", "tomato")]
    entries = [PredictionEntry(actor="knife_1", recipient="tomato_1",
                               predicate="canCut", verdict=1, score=0.9)]
    scenes, preds = _write_scene_and_preds(tmp_path, "scene_0", objects, entries)

    report = report_all_pairs_diagnostics(preds, scenes)

    assert report["canCut"]["extra_pairs"] == 0


def test_counts_extra_pair_for_structurally_impossible_triple(tmp_path):
    """knife -> knife, canCut is NOT a candidate pair (no class can cut
    itself in the gate) -- this must be counted as an extra pair."""
    objects = [_obj("knife_1", "knife"), _obj("knife_2", "knife")]
    entries = [PredictionEntry(actor="knife_1", recipient="knife_2",
                               predicate="canCut", verdict=0, score=0.1)]
    scenes, preds = _write_scene_and_preds(tmp_path, "scene_0", objects, entries)

    report = report_all_pairs_diagnostics(preds, scenes)

    assert report["canCut"]["extra_pairs"] == 1


def test_rejection_rate_is_fraction_of_extra_pairs_scored_negative(tmp_path):
    """3 structurally-impossible canCut pairs, model rejects 2 (verdict=0)
    and wrongly accepts 1 (verdict=1) -> rejection rate 2/3."""
    objects = [_obj("knife_1", "knife"), _obj("knife_2", "knife"),
               _obj("knife_3", "knife"), _obj("knife_4", "knife")]
    entries = [
        PredictionEntry(actor="knife_1", recipient="knife_2",
                        predicate="canCut", verdict=0, score=0.1),
        PredictionEntry(actor="knife_1", recipient="knife_3",
                        predicate="canCut", verdict=0, score=0.2),
        PredictionEntry(actor="knife_1", recipient="knife_4",
                        predicate="canCut", verdict=1, score=0.8),
    ]
    scenes, preds = _write_scene_and_preds(tmp_path, "scene_0", objects, entries)

    report = report_all_pairs_diagnostics(preds, scenes)

    assert report["canCut"]["extra_pairs"] == 3
    assert abs(report["canCut"]["rejection_rate"] - (2 / 3)) < 1e-9


def test_rejection_rate_is_none_when_predicate_has_no_extra_pairs(tmp_path):
    objects = [_obj("knife_1", "knife"), _obj("tomato_1", "tomato")]
    entries = [PredictionEntry(actor="knife_1", recipient="tomato_1",
                               predicate="canCut", verdict=1, score=0.9)]
    scenes, preds = _write_scene_and_preds(tmp_path, "scene_0", objects, entries)

    report = report_all_pairs_diagnostics(preds, scenes)

    assert report["canCut"]["rejection_rate"] is None


def test_aggregates_across_multiple_scenes(tmp_path):
    scenes = tmp_path / "scenes"; scenes.mkdir()
    preds = tmp_path / "pred"; preds.mkdir()
    for i, (objects, entries) in enumerate([
        ([_obj("knife_1", "knife"), _obj("knife_2", "knife")],
         [PredictionEntry(actor="knife_1", recipient="knife_2",
                          predicate="canCut", verdict=0, score=0.1)]),
        ([_obj("knife_3", "knife"), _obj("knife_4", "knife")],
         [PredictionEntry(actor="knife_3", recipient="knife_4",
                          predicate="canCut", verdict=0, score=0.1)]),
    ]):
        sid = f"scene_{i}"
        (scenes / f"{sid}.json").write_text(_scene(sid, objects).model_dump_json())
        (preds / f"{sid}.json").write_text(
            Predictions(scene_id=sid, tier1=entries).model_dump_json())

    report = report_all_pairs_diagnostics(preds, scenes)

    assert report["canCut"]["extra_pairs"] == 2


def test_finds_scene_json_nested_under_a_matching_subdirectory(tmp_path):
    """Real corpora (omni_scenes_all_merged on the cluster) nest scenes as
    scene_<id>/scene_<id>.json, not flat scene_<id>.json -- Inferencer's
    predictions are still written flat (predictions_dir/scene_<id>.json), so
    the report must look inside the matching subdirectory rather than only
    checking scenes_dir/<same name>.json directly, or every scene silently
    fails to match and the report is all-zero."""
    scenes = tmp_path / "scenes"; scenes.mkdir()
    preds = tmp_path / "pred"; preds.mkdir()
    scene_id = "scene_v1_0001"
    nested = scenes / scene_id; nested.mkdir()
    objects = [_obj("knife_1", "knife"), _obj("knife_2", "knife")]
    (nested / f"{scene_id}.json").write_text(_scene(scene_id, objects).model_dump_json())
    entries = [PredictionEntry(actor="knife_1", recipient="knife_2",
                               predicate="canCut", verdict=0, score=0.1)]
    (preds / f"{scene_id}.json").write_text(
        Predictions(scene_id=scene_id, tier1=entries).model_dump_json())

    report = report_all_pairs_diagnostics(preds, scenes)

    assert report["canCut"]["extra_pairs"] == 1


def test_uses_effective_class_when_resolved_instances_given(tmp_path):
    """A knife mislabeled 'peeler' in resolved_instances must be gated as a
    peeler, not its raw class -- matching how training/eval gate elsewhere."""
    objects = [_obj("knife_1", "knife"), _obj("tomato_1", "tomato")]
    entries = [PredictionEntry(actor="knife_1", recipient="tomato_1",
                               predicate="canCut", verdict=1, score=0.9)]
    scenes, preds = _write_scene_and_preds(tmp_path, "scene_0", objects, entries)

    resolved = tmp_path / "resolved_instances.json"
    import json
    resolved.write_text(json.dumps({"knife_1": {"effective_class": "peeler"}}))

    report_gated = report_all_pairs_diagnostics(preds, scenes)
    report_resolved = report_all_pairs_diagnostics(
        preds, scenes, resolved_instances=resolved)

    # canCut(knife, tomato) is a candidate; canCut(peeler, tomato) may not be
    # -- the two reports must be able to disagree, proving the effective
    # class actually changed the gate decision.
    assert report_gated["canCut"]["extra_pairs"] != report_resolved["canCut"]["extra_pairs"] \
        or report_gated["canCut"]["extra_pairs"] == 0  # sanity: at least ran
    assert report_resolved["canCut"]["extra_pairs"] == 1
