"""Tests for DINOv2Extractor.

The real DINOv2 model is never loaded — we inject a fake backbone via
`DINOv2Extractor(backbone=fake)`. This keeps tests fully offline and fast
while still exercising all logic: crop/resize, batching, save/load, no-overwrite.

The fake backbone returns a deterministic (n, 768) tensor given any input batch.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import pytest

from affdet.render.mock import MockRenderer
from affdet.schemas import Features, FeaturesMeta


# ---------------------------------------------------------------------------
# Fake backbone — returns deterministic (B, 768) CLS tokens, no GPU needed
# ---------------------------------------------------------------------------

class _FakeBackbone(torch.nn.Module):
    """Faithful to the real DINOv2 ViT two-method contract:

    - ``forward(x)`` returns a **tensor** ``(B, 768)`` — mirrors the real model's
      default ``forward`` (``self.head(x_norm_clstoken)``, Identity head for the
      pretrained backbone, so a bare tensor).
    - ``forward_features(x)`` returns the **dict** with ``x_norm_clstoken``.

    Encoding both faithfully is the point: a fake that only returned the dict
    masked a real bug (code called ``backbone(x)["x_norm_clstoken"]`` →
    IndexError on the 2-D tensor). With this fake, calling the wrong method
    crashes the unit test, guarding against regressions.
    """
    def _cls(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.arange(B, dtype=torch.float32).unsqueeze(1).expand(B, 768) * 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cls(x)  # (B, 768) tensor, like the real default forward

    def forward_features(self, x: torch.Tensor) -> dict:
        return {"x_norm_clstoken": self._cls(x)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenes(tmp_path: Path, n: int = 3, objs: int = 4) -> Path:
    scenes_dir = tmp_path / "scenes"
    MockRenderer(seed=0, n_scenes=n, objs_per_scene=(objs, objs)).run(scenes_dir)
    return scenes_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dinov2_extractor_writes_one_npz_per_scene(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    scenes_dir = _make_scenes(tmp_path, n=3)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(scenes_dir, feats_dir)
    assert len(sorted(feats_dir.glob("scene_*.npz"))) == 3


def test_dinov2_extractor_feature_shape(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    scenes_dir = _make_scenes(tmp_path, n=1, objs=4)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(scenes_dir, feats_dir)
    feats = Features.load(feats_dir / "scene_0001.npz")
    assert feats.features.shape == (4, 768)
    assert feats.features.dtype == np.float32


def test_dinov2_extractor_obj_ids_match_scene(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    from affdet.schemas import Scene
    scenes_dir = _make_scenes(tmp_path, n=1, objs=3)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(scenes_dir, feats_dir)
    scene = Scene.model_validate_json((scenes_dir / "scene_0001.json").read_text())
    feats = Features.load(feats_dir / "scene_0001.npz")
    assert feats.obj_ids == [o.obj_id for o in scene.objects]


def test_dinov2_extractor_writes_meta_sidecar(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    scenes_dir = _make_scenes(tmp_path, n=1)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(scenes_dir, feats_dir)
    meta = FeaturesMeta.model_validate_json(
        (feats_dir / "scene_0001.meta.json").read_text()
    )
    assert meta.bbox_source == "oracle"
    assert meta.dinov2_variant == "vitb14"


def test_dinov2_extractor_no_overwrite(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    scenes_dir = _make_scenes(tmp_path, n=1, objs=3)
    feats_dir = tmp_path / "features"
    extractor = DINOv2Extractor(backbone=_FakeBackbone())
    extractor.run(scenes_dir, feats_dir)
    mtime_before = (feats_dir / "scene_0001.npz").stat().st_mtime
    extractor.run(scenes_dir, feats_dir)
    mtime_after = (feats_dir / "scene_0001.npz").stat().st_mtime
    assert mtime_before == mtime_after


def test_dinov2_extractor_features_are_float32(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    scenes_dir = _make_scenes(tmp_path, n=1, objs=5)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(scenes_dir, feats_dir)
    feats = Features.load(feats_dir / "scene_0001.npz")
    assert feats.features.dtype == np.float32


def test_dinov2_extractor_no_scenes_raises(tmp_path):
    from affdet.extract.dinov2 import DINOv2Extractor
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No scenes found"):
        DINOv2Extractor(backbone=_FakeBackbone()).run(empty, tmp_path / "out")


def test_dinov2_extractor_handles_scene_with_zero_objects(tmp_path):
    """A scene with no objects (e.g. a live photo with zero detections) must
    not crash on torch.stack([]) — it should write an empty Features."""
    from affdet.extract.dinov2 import DINOv2Extractor
    scenes_dir = _make_scenes(tmp_path, n=1, objs=0)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(scenes_dir, feats_dir)
    feats = Features.load(feats_dir / "scene_0001.npz")
    assert feats.obj_ids == []
    assert feats.features.shape == (0, 768)


def test_dinov2_extractor_walks_nested_layout(tmp_path):
    """scene_*/scene_*.json layout (cluster output) is also walked."""
    from affdet.extract.dinov2 import DINOv2Extractor
    from affdet.schemas import Scene
    # build flat scenes, then move each into a subdir
    flat = tmp_path / "flat"
    MockRenderer(seed=0, n_scenes=2, objs_per_scene=(3, 3)).run(flat)
    nested = tmp_path / "nested"
    nested.mkdir()
    for scene_json in sorted(flat.glob("scene_*.json")):
        sid = scene_json.stem
        sub = nested / sid
        sub.mkdir()
        for f in flat.glob(f"{sid}*"):
            f.rename(sub / f.name)
    feats_dir = tmp_path / "features"
    DINOv2Extractor(backbone=_FakeBackbone()).run(nested, feats_dir)
    assert len(sorted(feats_dir.glob("scene_*.npz"))) == 2
