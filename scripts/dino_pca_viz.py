"""Visualise DINOv2 patch features as an RGB image via PCA.

Colours are the first three principal components of the patch-token features,
mapped to R, G and B. No segmentation head and no labels are involved: the
structure visible in the output is already present in the frozen features.

Used to build the DINOv2 explainer figure in the final presentation.

  python scripts/dino_pca_viz.py --images a.png b.png --out-dir figs/

The background of a render is flat and dominates the first component, so
`--fg-threshold` drops patches whose first component sits on the background
side of the split before fitting the colour PCA, and paints them white.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_PATCH = 14


def _load(path: Path, size: int) -> tuple[torch.Tensor, Image.Image]:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((t - _MEAN) / _STD).unsqueeze(0), img


def _patch_tokens(model, batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model.forward_features(batch)
    return out["x_norm_patchtokens"][0].float().cpu().numpy()


def _pca(x: np.ndarray, k: int) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    # economy SVD: rows are patches, so the right singular vectors are the axes
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:k].T


def _norm(x: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0,1] so a few outliers cannot wash the image out."""
    lo, hi = np.percentile(x, 2, axis=0), np.percentile(x, 98, axis=0)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)


def render(model, path: Path, size: int, fg_threshold: bool) -> Image.Image:
    batch, _ = _load(path, size)
    if torch.cuda.is_available():
        batch = batch.cuda()
    tokens = _patch_tokens(model, batch)
    side = size // _PATCH

    keep = np.ones(len(tokens), dtype=bool)
    if fg_threshold:
        first = _pca(tokens, 1)[:, 0]
        # the larger side of the split is the background in these renders
        keep = first < np.median(first) if (first > np.median(first)).sum() > len(first) / 2 \
            else first > np.median(first)
        if keep.sum() < 16:          # degenerate split: keep everything
            keep = np.ones(len(tokens), dtype=bool)

    rgb = np.ones((len(tokens), 3), dtype=np.float32)
    rgb[keep] = _norm(_pca(tokens[keep], 3))

    grid = rgb.reshape(side, side, 3)
    out = (grid * 255).astype(np.uint8)
    return Image.fromarray(out).resize((size, size), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--size", type=int, default=448, help="multiple of 14")
    ap.add_argument("--variant", default="vitb14")
    ap.add_argument("--fg-threshold", action="store_true",
                    help="drop background patches before fitting the colour PCA")
    args = ap.parse_args()

    assert args.size % _PATCH == 0, f"--size must be a multiple of {_PATCH}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{args.variant}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    for p in args.images:
        img = render(model, p, args.size, args.fg_threshold)
        dst = args.out_dir / f"{p.stem}_dinopca.png"
        img.save(dst)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
