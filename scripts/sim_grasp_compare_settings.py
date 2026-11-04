"""Compare grasp-map variants: one figure (objects x variants, top-down score maps) and a markdown table.

    python grasp_compare.py ITER_DIR OUT_PREFIX [--variants base f70 ...]

ITER_DIR/<variant>/grasp_<name>.pkl as written by scripts/sim_grasp_maps.py (with params, on_object, hold_force_n).
Score = share of gripper yaws that lift, hold without slip and stay within 30 deg of level ("success").
"""
import argparse, glob, json, os, pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("out"); ap.add_argument("--variants", nargs="*")
a = ap.parse_args()
variants = a.variants or sorted(d for d in os.listdir(a.dir) if os.path.isdir(os.path.join(a.dir, d)) and glob.glob(os.path.join(a.dir, d, "grasp_*.pkl")))
maps = {v: {os.path.basename(f)[6:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(a.dir, v, "grasp_*.pkl")))} for v in variants}
objects = sorted({o for v in variants for o in maps[v]})

def stats(m):
    on = m["on_object"].any(2)
    succ = m["success"][on].mean() if on.any() else np.nan          # share of (cell, yaw) on the object that succeed
    held = m["held"][on].mean() if on.any() else np.nan
    cell = (m["score"] > 0)[on].mean() if on.any() else np.nan     # share of object cells with at least one good yaw
    hf = m["hold_force_n"][m["held"]]
    return dict(success=float(succ), held=float(held), cells_any=float(cell), hold_force=float(hf.mean()) if hf.size else 0.0,
                swing_held=float(np.median(m["swing_deg"][m["held"]])) if m["held"].any() else np.nan)

table = {v: {o: stats(maps[v][o]) for o in objects if o in maps[v]} for v in variants}
json.dump({"variants": variants, "params": {v: next(iter(maps[v].values()))["params"] for v in variants if maps[v]}, "stats": table},
          open(a.out + ".json", "w"), indent=1, default=str)

lines = ["Success share of (cell, yaw) grasps on the object (held, no slip, within 30 deg of level). Cells with any good yaw in brackets.", "",
         "| object | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
for o in objects:
    lines.append(f"| {o} | " + " | ".join(f"{table[v][o]['success']:.2f} ({table[v][o]['cells_any']:.2f})" if o in table[v] else "-" for v in variants) + " |")
lines.append("| **mean** | " + " | ".join(f"**{np.nanmean([s['success'] for s in table[v].values()]):.3f}**" for v in variants) + " |")
lines += ["", "Mean pad force during the hold (N), held grasps only:", "",
          "| " + " | ".join(variants) + " |", "|" + "---|" * len(variants),
          "| " + " | ".join(f"{np.nanmean([s['hold_force'] for s in table[v].values() if s['hold_force'] > 0]):.1f}" for v in variants) + " |"]
open(a.out + ".md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, axes = plt.subplots(len(objects), len(variants), figsize=(2.1 * len(variants), 2.1 * len(objects)), squeeze=False)
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
for i, o in enumerate(objects):
    for j, v in enumerate(variants):
        ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
        if o not in maps[v]:
            ax.set_facecolor("#eee"); continue
        m = maps[v][o]
        on = m["on_object"].any(2)
        img = np.where(on, m["score"], np.nan)
        xs, ys, st = m["xs"], m["ys"], m["step"]
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1, extent=[xs[0] - st / 2, xs[-1] + st / 2, ys[0] - st / 2, ys[-1] + st / 2])
        ax.contour(xs, ys, on.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        ax.set_title(f"{table[v][o]['success']:.2f}", fontsize=8)
        if j == 0:
            ax.set_ylabel(o, fontsize=8)
        if i == 0:
            ax.annotate(v, (0.5, 1.35), xycoords="axes fraction", ha="center", fontsize=10, weight="bold")
fig.suptitle("Grasp maps: share of yaws that succeed per cell (yellow = all 6, dark = none, white = off the object)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(a.out + ".png", dpi=110)
print("wrote", a.out + ".png")
