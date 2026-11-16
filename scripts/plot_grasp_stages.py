"""Clamp / lift / level grasp maps side by side, one row per object (top-down view of the grid).

    python scripts/plot_grasp_stages.py OUT.png MAP_DIR [--old OLD_MAP_DIR] [--names mug0 bowl0 ...]

Per cell: the share of gripper yaws that pass each stage (clamp, lift, level), and the furthest stage any yaw reached
(fail-stage map). With --old, the old `held` and `success` (held and swing < 30 deg) maps of the same object come first,
for comparison (they may use another grid). Thresholds: affdet.sim.grasp_stages.THRESHOLDS, stored in each map.
"""
import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from affdet.sim.grasp_stages import FAIL  # noqa: E402

STAGE_NAMES = ["off object", "blocked", "force not reached", "clamp unstable", "lift failed", "not level", "level (success)"]
STAGE_COLOURS = ["#ffffff", "#7f7f7f", "#d62728", "#ff7f0e", "#bcbd22", "#17becf", "#2ca02c"]


def _show(ax, m, img, title, cmap="viridis", norm=None, vmax=1.0):
    ext = [m["xs"][0], m["xs"][-1], m["ys"][0], m["ys"][-1]]
    s = m["step"] / 2
    ext = [100 * (ext[0] - s), 100 * (ext[1] + s), 100 * (ext[2] - s), 100 * (ext[3] + s)]
    img = np.where(m["on_object"].any(2), img, np.nan) if norm is None else img
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=8); ax.set_xlabel("x (cm)", fontsize=7); ax.tick_params(labelsize=6)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("maps"); ap.add_argument("--old"); ap.add_argument("--names", nargs="+")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.maps, "grasp_*.pkl")))
    if a.names:
        files = [f for f in files if os.path.basename(f)[6:-4] in a.names]
    ncol = 4 + (2 if a.old else 0)
    fig, axes = plt.subplots(len(files), ncol, figsize=(2.6 * ncol, 2.4 * len(files)), squeeze=False, layout="constrained")
    cmap_stage = ListedColormap(STAGE_COLOURS)
    norm_stage = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), 7)
    for r, f in enumerate(files):
        m = pickle.load(open(f, "rb"))
        name = os.path.basename(f)[6:-4]
        c = 0
        if a.old:
            fo = os.path.join(a.old, os.path.basename(f))
            if os.path.exists(fo):
                o = pickle.load(open(fo, "rb"))
                _show(axes[r, 0], o, o["held"].mean(2), f"{name}\nold: held (share of yaws)")
                _show(axes[r, 1], o, o["success"].mean(2), "old: held and swing < 30 deg")
            c = 2
        for j, k in enumerate(("clamp", "lift", "level")):
            im = _show(axes[r, c + j], m, m[k].mean(2), f"{k} (share of yaws)" + ("" if c or j else f"\n{name}"))
        best = m["fail_stage"].max(2)  # the furthest stage any yaw reached at this cell
        _show(axes[r, c + 3], m, best, "furthest stage reached", cmap=cmap_stage, norm=norm_stage)
    fig.colorbar(im, ax=axes[:, -2], shrink=0.5, label="share of yaws", location="right")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="k", lw=0.3) for col in STAGE_COLOURS]
    fig.legend(handles, STAGE_NAMES, loc="outside lower center", ncol=7, fontsize=7, frameon=False)
    m0 = pickle.load(open(files[0], "rb"))
    p = m0["params"]
    fig.suptitle(f"Grasp stages, settle {p['settle']} s, force {p['force']} N, thresholds as stored in the maps "
                 f"(tilt < {m0['thresholds']['max_tilt_deg']} deg, slip < {100 * m0['thresholds']['max_slip_m']:.0f} cm)", fontsize=9)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    assert FAIL["success"] == len(STAGE_NAMES) - 1
    main()
