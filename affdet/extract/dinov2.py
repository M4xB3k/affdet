"""DINOv2Extractor — frozen ViT-B/14 CLS-token features per object crop.

For each scene: crops each object bbox from the RGB image, resizes to 224×224,
runs all crops through the backbone in one forward pass, saves (N, 768) float32
features as an NPZ alongside a JSON sidecar.

The backbone is injected so tests can pass a fake without touching the network.
The real backbone is built lazily inside run() when none is injected.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from affdet.schemas import Scene, Features, FeaturesMeta


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _pad_bbox(bbox, W: float, H: float, frac: float) -> list[float]:
    """Expand a [x1,y1,x2,y2] bbox by `frac` of its w/h on each side, clamped."""
    if frac == 0.0:
        return list(bbox)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return [max(0.0, x1 - frac * w), max(0.0, y1 - frac * h),
            min(W, x2 + frac * w), min(H, y2 + frac * h)]


def _preprocess(crop: Image.Image) -> torch.Tensor:
    crop = crop.resize((224, 224), Image.BILINEAR)
    t = torch.from_numpy(np.array(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - _MEAN) / _STD


def _build_backbone(variant: str = "vitb14") -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _find_scene_jsons(scenes_dir: Path) -> list[Path]:
    flat = sorted(scenes_dir.glob("scene_*.json"))
    nested = sorted(scenes_dir.glob("scene_*/scene_*.json"))
    return flat if flat else nested


@dataclass
class DINOv2Extractor:
    backbone: Any = field(default=None)  # injected in tests; built lazily in run()
    variant: str = "vitb14"
    crop_padding: float = 0.0

    def run(self, scenes_dir: Path | str, features_dir: Path | str,
            bbox_source: str = "oracle") -> None:
        scenes_dir = Path(scenes_dir)
        features_dir = Path(features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)

        scene_jsons = _find_scene_jsons(scenes_dir)
        if not scene_jsons:
            raise ValueError(f"No scenes found under {scenes_dir}")

        backbone = self.backbone if self.backbone is not None else _build_backbone(self.variant)
        try:
            device = next(backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        meta = FeaturesMeta(bbox_source=bbox_source, dinov2_variant=self.variant)

        for scene_json in scene_jsons:
            out_npz = features_dir / f"{scene_json.stem}.npz"
            if out_npz.exists():
                continue

            scene = Scene.model_validate_json(scene_json.read_text())

            if not scene.objects:
                # No detections (e.g. a live photo with zero YOLO boxes) —
                # nothing to crop or embed.
                Features(obj_ids=[], features=np.zeros((0, 768), dtype="float32")).save(out_npz)
                (features_dir / f"{scene_json.stem}.meta.json").write_text(
                    meta.model_dump_json()
                )
                continue

            img = Image.open(scene_json.parent / scene.image_path).convert("RGB")

            W_px, H_px = img.size
            crops = torch.stack([
                _preprocess(img.crop(tuple(_pad_bbox(o.bbox, W_px, H_px, self.crop_padding))))
                for o in scene.objects
            ]).to(device)

            with torch.no_grad():
                # DINOv2's default forward() returns self.head(cls) — a (B, 768)
                # tensor (Identity head for the pretrained backbone). The dict
                # with "x_norm_clstoken" comes from forward_features().
                cls_tokens = backbone.forward_features(crops)["x_norm_clstoken"]

            Features(
                obj_ids=[o.obj_id for o in scene.objects],
                features=cls_tokens.cpu().float().numpy(),
            ).save(out_npz)

            (features_dir / f"{scene_json.stem}.meta.json").write_text(
                meta.model_dump_json()
            )
