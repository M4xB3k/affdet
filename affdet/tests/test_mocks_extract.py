from pathlib import Path
import numpy as np
from affdet.render.mock import MockRenderer
from affdet.extract.mock import MockExtractor
from affdet.schemas import Features, FeaturesMeta


def test_mock_extractor_writes_one_npz_per_scene(tmp_path):
    scenes_dir = tmp_path / "scenes"
    feats_dir = tmp_path / "features"
    MockRenderer(seed=0, n_scenes=3, objs_per_scene=(3, 3)).run(scenes_dir)
    MockExtractor(seed=0).run(scenes_dir, feats_dir)
    assert len(sorted(feats_dir.glob("scene_*.npz"))) == 3


def test_mock_extractor_features_have_correct_shape(tmp_path):
    scenes_dir = tmp_path / "scenes"
    feats_dir = tmp_path / "features"
    MockRenderer(seed=0, n_scenes=1, objs_per_scene=(4, 4)).run(scenes_dir)
    MockExtractor(seed=0).run(scenes_dir, feats_dir)
    feats = Features.load(feats_dir / "scene_0001.npz")
    assert feats.features.shape == (4, 768)


def test_mock_extractor_deterministic(tmp_path):
    scenes_dir = tmp_path / "scenes"
    MockRenderer(seed=0, n_scenes=1, objs_per_scene=(3, 3)).run(scenes_dir)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    MockExtractor(seed=42).run(scenes_dir, out_a)
    MockExtractor(seed=42).run(scenes_dir, out_b)
    fa = Features.load(out_a / "scene_0001.npz")
    fb = Features.load(out_b / "scene_0001.npz")
    np.testing.assert_array_equal(fa.features, fb.features)


def test_mock_extractor_writes_meta_sidecar(tmp_path):
    scenes_dir = tmp_path / "scenes"
    feats_dir = tmp_path / "features"
    MockRenderer(seed=0, n_scenes=1, objs_per_scene=(3, 3)).run(scenes_dir)
    MockExtractor(seed=0).run(scenes_dir, feats_dir)
    meta = FeaturesMeta.model_validate_json((feats_dir / "scene_0001.meta.json").read_text())
    assert meta.bbox_source == "oracle"
