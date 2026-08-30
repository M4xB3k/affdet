import json

from typer.testing import CliRunner

from affdet.schemas import Scene, SceneObject, Camera, Labels
from affdet.scripts.emit_labels import app

runner = CliRunner()


def test_help_lists_resolve_and_omni_score():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "resolve" in result.output
    assert "omni-score" in result.output
    assert "pair-score" in result.output


def _write_scene(d, sid, objs):
    d.mkdir(parents=True, exist_ok=True)
    s = Scene(scene_id=sid, image_path=f"{sid}.png", image_size=[128, 96],
              source="blender", objects=objs,
              camera=Camera(pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    (d / f"{sid}.json").write_text(s.model_dump_json())


def test_pair_score_derives_labels_from_pair_labeling_corpus(tmp_path):
    scenes = tmp_path / "scenes"
    _write_scene(scenes, "scene_0001", [
        SceneObject(obj_id="knife_001", obj_class="knife", bbox=[0, 0, 10, 10]),
        SceneObject(obj_id="apple_001", obj_class="apple", bbox=[0, 0, 10, 10]),
    ])

    vlm = tmp_path / "labels_omni.jsonl"
    vlm.write_text(json.dumps(
        {"actor": "knife_001", "recipient": "apple_001", "predicate": "canCut", "score": 9}
    ) + "\n")

    resolved = tmp_path / "resolved_instances_v2.json"
    resolved.write_text(json.dumps({
        "knife_001": {"effective_class": "knife", "dropped": False,
                      "pourable_from": None, "pourable_into": None},
    }))

    curation = tmp_path / "curation.json"
    curation.write_text(json.dumps({"apple": {"placement": {"whole": ["001"]}}}))

    out = tmp_path / "labels"
    result = runner.invoke(app, [
        "pair-score",
        "--scenes", str(scenes),
        "--vlm", str(vlm),
        "--resolved", str(resolved),
        "--curation", str(curation),
        "--out", str(out),
    ])

    assert result.exit_code == 0, result.output
    labels = Labels.model_validate_json((out / "scene_0001.json").read_text())
    cut = [e for e in labels.tier1 if e.predicate == "canCut"]
    assert cut and abs(cut[0].score - 1.0) < 1e-6
