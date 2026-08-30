"""MockExtractor — random 768-d gaussians per object, deterministic by scene_id."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

from affdet.schemas import Scene, Features, FeaturesMeta


def _scene_seed(base_seed: int, scene_id: str) -> int:
    h = hashlib.sha256(scene_id.encode()).digest()
    return (base_seed + int.from_bytes(h[:4], "big")) % (2**32)


@dataclass
class MockExtractor:
    seed: int = 0

    def run(self, scenes_dir: Path | str, features_dir: Path | str) -> None:
        scenes_dir = Path(scenes_dir)
        features_dir = Path(features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)

        for scene_json in sorted(scenes_dir.glob("scene_*.json")):
            scene = Scene.model_validate_json(scene_json.read_text())
            rng = np.random.RandomState(_scene_seed(self.seed, scene.scene_id))
            n = len(scene.objects)
            feats = rng.randn(n, 768).astype(np.float32)
            Features(
                obj_ids=[o.obj_id for o in scene.objects],
                features=feats,
            ).save(features_dir / f"{scene.scene_id}.npz")
            meta = FeaturesMeta(bbox_source="oracle", dinov2_variant="vitb14",
                                extractor_version="mock-v1")
            (features_dir / f"{scene.scene_id}.meta.json").write_text(
                meta.model_dump_json()
            )
