"""Analytic grasp maps for the curated objects, and their agreement with the simulated gripper.

    python scripts/sim_grasp_analytic_report.py PICKS_JSON GLB_DIR WORK_DIR OUT_DIR

WORK_DIR holds the simulated maps (grasp_mug/, ...) and the CoACD cache (placement/coacd/).
Writes grasp_analytic_<role>_<uid>.pkl / .png per object and agreement.json (analytic vs simulated cells).
"""
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

from affdet.sim.grasp_analytic import grasp_analytic  # noqa: E402
from affdet.sim.objects import load_body  # noqa: E402
from scripts.sim_map_figures import outline  # noqa: E402

YAWS = (0, 30, 60, 90, 120, 150)


def classes(a):
    return np.where(a["level"].any(2), 2, np.where(a["fits"].any(2), 1, 0))


def main():
    picks_json, glb_dir, work, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    picks = json.load(open(picks_json))
    agree = {}
    for role, idxs, step in (("frying_pan", (0, 1), 0.01), ("mug", (0,), 0.01)):
        for i in idxs:
            r = picks[role][i]
            uid = r["object_uid"][:8]
            body, mesh = load_body(os.path.join(glb_dir, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                                   float(r["objathor_mass_kg"]), "obj", cache_dir=os.path.join(work, "placement", "coacd"))
            ex = mesh.extents[:2]
            xs = np.arange(-ex[0] / 2, ex[0] / 2 + 1e-9, step)
            ys = np.arange(-ex[1] / 2, ex[1] / 2 + 1e-9, step)
            a = grasp_analytic(body.parts, body.mass, xs, ys, YAWS)
            a["step"] = step
            pickle.dump(a, open(os.path.join(out, f"grasp_analytic_{role}_{uid}.pkl"), "wb"))
            cls = classes(a)
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
            ext = [xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2]
            ax[0].imshow(cls, origin="lower", extent=ext, cmap=ListedColormap(["#2b2b2b", "#e8b04b", "#7bd88f"]), vmin=0, vmax=2,
                         interpolation="nearest")
            outline(ax[0], mesh, color="w")
            ax[0].plot(a["com"][0], a["com"][1], "c^", ms=11, label="centre of mass")
            ax[0].set_title(f"{role} {uid} ({body.mass} kg): analytic best pinch\n(green holds it level, amber fits but swings, dark: no pinch fits)")
            ax[0].legend(fontsize=7, loc="lower left")
            mg = np.where(a["fits"], a["margin"], 0).max(2)
            im = ax[1].imshow(np.log10(np.clip(mg, 1e-2, 1e2)), origin="lower", extent=ext, cmap="RdYlGn", vmin=-1, vmax=1)
            outline(ax[1], mesh, color="0.3")
            ax[1].plot(a["com"][0], a["com"][1], "k^", ms=11)
            fig.colorbar(im, ax=ax[1], shrink=0.8, label="log10(torque capacity / weight torque)")
            ax[1].set_title(f"torque margin of the best yaw (limit: {100 * a['d_max']:.1f} cm from the COM line)")
            # best yaw per cell: closing direction relative to the line to the COM
            best = np.where(a["fits"], a["margin"], -1).argmax(2)
            X, Y = np.meshgrid(xs, ys)
            ok = a["fits"].any(2)
            ang = np.radians(np.array(YAWS))[best] + np.pi / 2  # closing direction angle (u = yaw + 90 deg)
            ax[2].quiver(X[ok], Y[ok], np.cos(ang)[ok], np.sin(ang)[ok], color="C3", headwidth=0, pivot="mid", scale=60, width=0.003)
            outline(ax[2], mesh, color="0.6")
            ax[2].plot(a["com"][0], a["com"][1], "k^", ms=11)
            ax[2].set_title("closing line of the best yaw (it points at the COM when it can)")
            for x_ in ax:
                x_.set_aspect("equal"); x_.set_xlabel("x (m)"); x_.set_ylabel("y (m)")
            fig.tight_layout(); fig.savefig(os.path.join(out, f"grasp_analytic_{role}_{uid}.png"), dpi=130); plt.close(fig)
            # agreement with the simulated gripper where a simulated map exists
            sim_f = [f for f in (os.path.join(work, "grasp_mug", f"{role}_{uid}.pkl"), os.path.join(work, "grasp", f"{role}_{uid}.pkl"))
                     if os.path.exists(f)]
            if sim_f:
                s = pickle.load(open(sim_f[0], "rb"))
                if s["score"].shape == cls.shape:
                    sim_ok = s["success"].any(2)
                    an_ok = a["level"].any(2)
                    agree[f"{role}_{uid}"] = {
                        "cells": int(sim_ok.size), "both_yes": int((sim_ok & an_ok).sum()), "sim_only": int((sim_ok & ~an_ok).sum()),
                        "analytic_only": int((~sim_ok & an_ok).sum()), "both_no": int((~sim_ok & ~an_ok).sum()),
                        "sim_held_any": int(s["held"].any(2).sum()), "analytic_fits_any": int(a["fits"].any(2).sum())}
    json.dump(agree, open(os.path.join(out, "agreement.json"), "w"), indent=1)
    print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    main()
