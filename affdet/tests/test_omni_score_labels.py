import json
from pathlib import Path

from affdet.schemas import Scene, SceneObject, Camera, Labels
from affdet.label.vlm_resolver import VlmScoreResolver
from affdet.label.omni_score_labels import derive_scene_labels


def _obj(cls, oid):
    return SceneObject(obj_id=oid, obj_class=cls, bbox=[0, 0, 10, 10])


def _write_scene(d: Path, sid, objs):
    d.mkdir(parents=True, exist_ok=True)
    s = Scene(scene_id=sid, image_path=f"{sid}.png", image_size=[128, 96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                            intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
    (d / f"{sid}.json").write_text(s.model_dump_json())


def _resolver():
    resolved = {
        "knife_001": {"effective_class": "knife", "dropped": False, "pourable": None},
        "knife_011": {"effective_class": "peeler", "dropped": False, "pourable": None},
    }
    rows = [
        {"actor": "knife_001", "food": "apple", "affordance": "canCut", "score": 9},
        {"actor": "peeler_011", "food": "apple", "affordance": "canPeel", "score": 8},
    ]
    return VlmScoreResolver.from_dicts(rows, resolved)


def test_emits_soft_score_label(tmp_path):
    scenes = tmp_path / "scenes"
    _write_scene(scenes, "scene_0001", [_obj("knife", "knife_001"), _obj("apple", "apple_002")])
    out = tmp_path / "labels"
    derive_scene_labels(scenes, out, _resolver())

    labels = Labels.model_validate_json((out / "scene_0001.json").read_text())
    cut = [e for e in labels.tier1 if e.predicate == "canCut"]
    assert cut and abs(cut[0].score - 1.0) < 1e-6


def test_peeler_gated_on_canpeel_not_cancut(tmp_path):
    # knife_011 is effectively a peeler -> canPeel candidate, NOT canCut
    scenes = tmp_path / "scenes"
    _write_scene(scenes, "scene_0002", [_obj("knife", "knife_011"), _obj("apple", "apple_002")])
    out = tmp_path / "labels"
    derive_scene_labels(scenes, out, _resolver())
    labels = Labels.model_validate_json((out / "scene_0002.json").read_text())
    preds = {e.predicate for e in labels.tier1}
    assert "canPeel" in preds
    assert "canCut" not in preds


def test_missing_score_drops_pair_and_reports(tmp_path):
    scenes = tmp_path / "scenes"
    # knife_099 has no resolved entry and no score row -> dropped
    _write_scene(scenes, "scene_0003", [_obj("knife", "knife_099"), _obj("apple", "apple_002")])
    out = tmp_path / "labels"
    cov = derive_scene_labels(scenes, out, _resolver())
    labels = Labels.model_validate_json((out / "scene_0003.json").read_text())
    assert all(e.predicate != "canCut" for e in labels.tier1)
    assert cov["n_dropped"] >= 1


def test_coverage_report_written(tmp_path):
    scenes = tmp_path / "scenes"
    _write_scene(scenes, "scene_0001", [_obj("knife", "knife_001"), _obj("apple", "apple_002")])
    out = tmp_path / "labels"
    derive_scene_labels(scenes, out, _resolver())
    cov = json.loads((out / "label_coverage.json").read_text())
    assert "n_emitted" in cov and "per_predicate" in cov
