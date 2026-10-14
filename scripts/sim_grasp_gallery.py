"""Small multiples of grasp maps: objects painted with the share of gripper yaws that lift and hold them (top row) and that also keep
them within 30 degrees of level (bottom row).

    python scripts/sim_grasp_gallery.py PICKS_JSON GLB_DIR SHAKER_CSV GRASP_DIR(S...) --specs mug:0 frying_pan:0 ... --out OUT.png
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
from affdet.sim.registry import Registry  # noqa: E402
from scripts.sim_map_render import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--specs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, "/tmp")
    panels = []
    for spec in a.specs:
        hit = [f for d in a.dirs for f in glob.glob(os.path.join(d, "**", f"grasp_{reg.spec_name(spec)}.pkl"), recursive=True)]
        if not hit:
            continue
        g = pickle.load(open(hit[0], "rb"))
        mesh = prepare(*reg.glb(spec), "+y")[1]
        held = g["held"].mean(2); stable = (g["held"] & (g["swing_deg"] < 30)).mean(2)
        row = []
        for grid in (held, stable):
            def col(sub, grid=grid):
                up = sub.vertex_normals[:, 2] > 0.3
                v = _lookup(g, sub.vertices[:, 0], sub.vertices[:, 1], grid, smooth=True)
                rgba = (np.array(cm.viridis(np.clip(v, 0, 1))) * 255).astype(np.uint8)
                rgba[v <= 0.01] = [233, 190, 190, 255]
                rgba[~up] = [205, 205, 205, 255]
                return rgba
            row.append(render(mesh, col, size=340, el=72, dist=1.55))
        panels.append((spec, row, 100 * g["held"].any(2).mean(), 100 * (g["held"] & (g["swing_deg"] < 30)).any(2).mean()))
    n = len(panels)
    fig, axs = plt.subplots(2, n, figsize=(2.8 * n, 6.2), squeeze=False)
    for k, (spec, row, h, s) in enumerate(panels):
        for r in range(2):
            axs[r][k].imshow(row[r]); axs[r][k].axis("off")
        axs[0][k].set_title(f"{spec}\nheld in {h:.0f}% of cells", fontsize=9)
        axs[1][k].set_title(f"level (<30 deg) in {s:.0f}%", fontsize=9)
    fig.suptitle("canGrasp: share of gripper yaws that lift and hold the object (top), and keep it level (bottom); pink = none", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("wrote", a.out, n)


if __name__ == "__main__":
    main()
