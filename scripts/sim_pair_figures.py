"""Figure for a pair map: A placed on B, with a heatmap on each object.

    python scripts/sim_pair_figures.py PAIR_PKL SHAKER_GLB SHAKER_HEIGHT_M BASE_GLB BASE_HEIGHT_M OUT_PNG [--cache DIR]

Panels: (1) on B, how robustly A can be put down at each point (best over A's faces), with the winning face;
(2) on A, how robustly each face works as the support; (3, 4) A rendered from above and below with that value
painted on the resting patch of each face.
"""
import argparse
import os
import pickle
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from affdet.geometry.prepare import prepare  # noqa: E402
from affdet.sim.pair_map import patch_heat  # noqa: E402
from scripts.sim_map_figures import outline  # noqa: E402
from scripts.sim_map_render import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl"); ap.add_argument("top_glb"); ap.add_argument("top_h", type=float)
    ap.add_argument("base_glb"); ap.add_argument("base_h", type=float); ap.add_argument("out")
    ap.add_argument("--title", default="A on B")
    a = ap.parse_args()
    m = pickle.load(open(a.pkl, "rb"))
    _, top = prepare(a.top_glb, a.top_h, "+y")
    _, base = prepare(a.base_glb, a.base_h, "+y")
    faces = m["faces"]
    vmax = max(1.0, max(m["a_face_margin_mm"].values()))
    fig = plt.figure(figsize=(17, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 0.9, 0.8, 0.8])
    ax = fig.add_subplot(gs[0])
    ext = [m["xs"][0] - m["step"] / 2, m["xs"][-1] + m["step"] / 2, m["ys"][0] - m["step"] / 2, m["ys"][-1] + m["step"] / 2]
    im = ax.imshow(m["b_margin_mm"], origin="lower", extent=ext, cmap="viridis")
    outline(ax, base, color="w")
    ax.set_title(f"on B: robustness of the best face (mm)\n{100 * m['b_any'].mean():.0f}% of points can carry A with some face")
    ax.set_aspect("equal"); fig.colorbar(im, ax=ax, shrink=0.75)
    ax2 = fig.add_subplot(gs[1])
    vals = [m["a_face_margin_mm"][f] for f in faces]
    cols = [cm.viridis(v / vmax) if f not in m["skipped"] else "0.8" for f, v in zip(faces, vals)]
    ax2.barh(range(len(faces)), vals, color=cols)
    ax2.set_yticks(range(len(faces)))
    ax2.set_yticklabels([f + (" (cannot rest)" if f in m["skipped"] else f"  {100 * m['a_face_stable_frac'][f]:.0f}% of B" ) for f in faces], fontsize=8)
    ax2.set_xlabel("best robustness with this face down (mm)"); ax2.invert_yaxis()
    ax2.set_title("on A: which face can support it")
    fv = {f: v for f, v in zip(faces, vals)}

    def col(sub):
        h = patch_heat(sub.vertices, fv)
        rgba = (np.array(cm.viridis(np.clip(h / vmax, 0, 1))) * 255).astype(np.uint8)
        rgba[h <= 0] = [205, 205, 205, 255]
        return rgba

    for k, (el, t) in enumerate(((35, "A from above"), (-35, "A from below"))):
        axk = fig.add_subplot(gs[2 + k])
        axk.imshow(render(top, col, size=420, el=el, dist=2.1)); axk.axis("off"); axk.set_title(t)
    fig.suptitle(a.title)
    fig.tight_layout(); fig.savefig(a.out, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
