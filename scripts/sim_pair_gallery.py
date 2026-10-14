"""Small multiples of pair maps: one object painted for many partners.

    python scripts/sim_pair_gallery.py PICKS_JSON GLB_DIR SHAKER_CSV PAIR_DIR(S...) --base mug:0 --tops shaker:86450daa bowl:1 ... --out OUT.png
    python scripts/sim_pair_gallery.py ... --top shaker:c442c916 --bases mug:0 tray:0 ... --out OUT.png

Mode 1 (--base, --tops): the supporting object B painted with how robustly each top can be put down at every point.
Mode 2 (--top, --bases): the top object A painted with how robustly each of its faces supports it, once per base.
Colours: viridis of the margin in mm, scaled per panel; pink = nothing holds; grey = not a support surface.
"""
import argparse
import glob
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
from affdet.sim.label_export import _lookup  # noqa: E402
from affdet.sim.pair_map import patch_heat  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402
from scripts.sim_map_render import render  # noqa: E402


def find(dirs, name):
    for d in dirs:
        hit = glob.glob(os.path.join(d, "**", name + ".pkl"), recursive=True)
        if hit:
            return pickle.load(open(hit[0], "rb"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base"); ap.add_argument("--tops", nargs="*", default=[])
    ap.add_argument("--top"); ap.add_argument("--bases", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, "/tmp")
    mesh = lambda spec: prepare(*reg.glb(spec), "+y")[1]
    panels = []
    if a.base:
        obj = mesh(a.base)
        for t in a.tops:
            m = find(a.dirs, f"{reg.spec_name(t)}_on_{reg.spec_name(a.base)}")
            if m is None:
                continue
            vb = max(1.0, float(m["b_margin_mm"].max()))

            def col(sub, m=m, vb=vb):
                up = sub.vertex_normals[:, 2] > 0.5
                v = _lookup(m, sub.vertices[:, 0], sub.vertices[:, 1], m["b_margin_mm"], smooth=True)
                rgba = (np.array(cm.viridis(np.clip(v / vb, 0, 1))) * 255).astype(np.uint8)
                rgba[v <= 0.05] = [233, 150, 150, 255]
                rgba[~up] = [200, 200, 200, 255]
                return rgba
            panels.append((render(obj, col, size=360, el=78, dist=1.5), f"{t} on {a.base}\n{100 * m['b_any'].mean():.0f}% of points, best {vb:.0f} mm"))
    else:
        obj = mesh(a.top)
        for b in a.bases:
            m = find(a.dirs, f"{reg.spec_name(a.top)}_on_{reg.spec_name(b)}")
            if m is None:
                continue
            fv = {f: v for f, v in m["a_face_margin_mm"].items()}
            vmax = max(1.0, max(fv.values()))

            def col(sub, fv=fv, vmax=vmax):
                h = patch_heat(sub.vertices, fv)
                rgba = (np.array(cm.viridis(np.clip(h / vmax, 0, 1))) * 255).astype(np.uint8)
                rgba[h <= 0] = [205, 205, 205, 255]
                return rgba
            ok = [f for f, v in fv.items() if v > 0]
            panels.append((render(obj, col, size=360, el=-35, dist=2.1), f"{a.top} on {b}\nfaces that work: {','.join(ok) or 'none'}"))
    n = len(panels)
    cols = min(a.cols, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (img, title) in zip(axs.ravel(), panels):
        ax.imshow(img); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("wrote", a.out, n, "panels")


if __name__ == "__main__":
    main()
