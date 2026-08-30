from pathlib import Path
from PIL import Image
from affdet.render.mock import MockRenderer
from affdet.schemas import Scene


def test_mock_renderer_writes_n_scenes(tmp_path):
    MockRenderer(seed=42, n_scenes=5, objs_per_scene=(3, 5)).run(tmp_path)
    scene_jsons = sorted(tmp_path.glob("scene_*.json"))
    scene_pngs = sorted(tmp_path.glob("scene_*.png"))
    assert len(scene_jsons) == 5
    assert len(scene_pngs) == 5


def test_mock_scene_validates_against_schema(tmp_path):
    MockRenderer(seed=0, n_scenes=1, objs_per_scene=(3, 3)).run(tmp_path)
    json_path = tmp_path / "scene_0001.json"
    Scene.model_validate_json(json_path.read_text())  # raises on failure


def test_mock_renderer_deterministic(tmp_path):
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    MockRenderer(seed=42, n_scenes=2, objs_per_scene=(3, 3)).run(out1)
    MockRenderer(seed=42, n_scenes=2, objs_per_scene=(3, 3)).run(out2)
    j1 = (out1 / "scene_0001.json").read_text()
    j2 = (out2 / "scene_0001.json").read_text()
    assert j1 == j2


def test_mock_scene_has_gt_attrs_for_blender_source(tmp_path):
    MockRenderer(seed=0, n_scenes=1, objs_per_scene=(3, 3)).run(tmp_path)
    scene = Scene.model_validate_json((tmp_path / "scene_0001.json").read_text())
    assert scene.source == "blender"
    for obj in scene.objects:
        assert obj.gt_attrs is not None
        assert obj.mask_path is not None
