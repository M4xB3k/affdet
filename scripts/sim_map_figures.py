"""Figures for the placement and grasp actionability maps.

    python scripts/sim_map_figures.py PICKS_JSON GLB_DIR PLACEMENT_DIR GRASP_DIR OUT_DIR
"""
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from affdet.geometry.prepare import prepare  # noqa: E402


def outline(ax, mesh, color="0.35", lw=0.4):
    tri = mesh.vertices[mesh.faces][:, :, :2]
    for t in tri[:: max(1, len(tri) // 2500)]:
        ax.fill(t[:, 0], t[:, 1], fill=False, ec=color, lw=lw, alpha=0.5)


def placement_figure(pkl, mesh, title, out):
    m = pickle.load(open(pkl, "rb"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ext = [m["xs"][0] - m["step"] / 2, m["xs"][-1] + m["step"] / 2, m["ys"][0] - m["step"] / 2, m["ys"][-1] + m["step"] / 2]
    cm = ListedColormap(["#e9b3b3", "#9bd39b", "#9bbbe3"])
    ax[0].imshow(m["kind"], origin="lower", extent=ext, cmap=cm, vmin=0, vmax=2, interpolation="nearest")
    outline(ax[0], mesh)
    ax[0].plot(0, 0, "k+", ms=12, mew=2, label="centre (UTAMP-2 default)")
    ax[0].plot(*m["best_xy"], "r*", ms=14, label=f"most robust ({m['best_margin_mm']:.0f} mm)")
    ax[0].set_title(f"{title}: where can B be put down?\n(green: stays on top, blue: stays inside, red: falls/tips)")
    ax[0].legend(loc="upper right", fontsize=7)
    im = ax[1].imshow(m["margin_mm"], origin="lower", extent=ext, cmap="viridis")
    outline(ax[1], mesh, color="w")
    ax[1].plot(*m["best_xy"], "r*", ms=14)
    ax[1].set_title("robustness: mm to the nearest failing placement")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return m


def grasp_figure(pkl, mesh, title, out):
    m = pickle.load(open(pkl, "rb"))
    held, swing = m["held"], m["swing_deg"]
    clean = m["success"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ext = [m["xs"][0] - m["step"] / 2, m["xs"][-1] + m["step"] / 2, m["ys"][0] - m["step"] / 2, m["ys"][-1] + m["step"] / 2]
    # 0 nothing lifts, 1 lifts but the object swings (> max swing), 2 lifts and stays put (best over yaws)
    cls = np.where(clean.any(2), 2, np.where(held.any(2), 1, 0))
    ax[0].imshow(cls, origin="lower", extent=ext, cmap=ListedColormap(["#2b2b2b", "#e8b04b", "#7bd88f"]), vmin=0, vmax=2,
                 interpolation="nearest")
    outline(ax[0], mesh, color="w")
    ax[0].plot(m["com"][0], m["com"][1], "c^", ms=11, label="centre of mass")
    ax[0].set_title(f"{title}: best grasp per point\n(green lifts cleanly, amber lifts but swings, dark: nothing lifts)")
    ax[0].legend(fontsize=7, loc="lower left")
    sw = np.where(held, swing, np.inf).min(2)
    sw = np.where(np.isinf(sw), np.nan, sw)
    im = ax[1].imshow(sw, origin="lower", extent=ext, cmap="coolwarm", vmin=0, vmax=120)
    outline(ax[1], mesh, color="0.3")
    ax[1].plot(m["com"][0], m["com"][1], "k^", ms=11)
    ax[1].set_title("smallest swing (deg) among yaws that hold it")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    X, Y = np.meshgrid(m["xs"], m["ys"])
    d = 100 * np.hypot(X - m["com"][0], Y - m["com"][1])
    rng = np.random.default_rng(0)
    for k, (c, lab) in enumerate((("0.6", "nothing lifts"), ("#e8b04b", "lifts, swings"), ("#2ca02c", "lifts cleanly"))):
        sel = cls == k
        ax[2].scatter(d[sel], k + rng.uniform(-0.15, 0.15, sel.sum()), s=12, c=c, label=f"{lab} ({sel.sum()})")
    ax[2].set_yticks([0, 1, 2]); ax[2].set_yticklabels(["nothing", "swings", "clean"])
    ax[2].set_xlabel("distance of grasp centre from centre of mass (cm)"); ax[2].legend(fontsize=7)
    ax[2].set_title("COM distance vs. outcome")
    for a_ in ax[:2]:
        a_.set_aspect("equal"); a_.set_xlabel("x (m)"); a_.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return m


def main():
    picks_json, glb_dir, pl_dir, gr_dir, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    picks = json.load(open(picks_json))

    def mesh_of(role, idx):
        r = picks[role][idx]
        return prepare(os.path.join(glb_dir, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100)[1]

    for name, role in (("shaker_on_board", "cutting_board"), ("shaker_on_mug", "mug"), ("shaker_on_tray", "tray")):
        f = os.path.join(pl_dir, name + ".pkl")
        if os.path.exists(f):
            placement_figure(f, mesh_of(role, 0), name.replace("_", " "), os.path.join(out, name + ".png"))
    for f in sorted(glob.glob(os.path.join(gr_dir, "frying_pan_*.pkl"))):
        uid = os.path.basename(f)[11:19]
        idx = [i for i, r in enumerate(picks["frying_pan"]) if r["object_uid"].startswith(uid)][0]
        grasp_figure(f, mesh_of("frying_pan", idx), f"pan {uid}", os.path.join(out, f"grasp_{uid}.png"))


if __name__ == "__main__":
    main()
