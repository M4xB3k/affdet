"""Offline pilot: how few cells of a full sweep must be simulated to reproduce it? Emulates coarse-to-fine strategies on stored
full maps (per-face `stable` arrays), no simulation. For each strategy: fraction simulated, cell agreement, stable cells lost/invented,
and the error of the best margin.

    python scripts/sim_grid_pilot.py MAP_DIR OUT_JSON [--limit N]
"""
import argparse, glob, json, os, pickle
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def margin(stable, step):
    pad = np.pad(stable, 1, constant_values=False)
    return 1000 * step * distance_transform_edt(pad)[1:-1, 1:-1] * stable


def coarse_idx(n, f):
    c = list(range(0, n, f))
    return c if c[-1] == n - 1 else c + [n - 1]


def emulate(full, factor, grow):
    """Simulate every `factor`-th cell, then every cell whose enclosing coarse cells disagree, and (grow>0) every cell within `grow`
    fine cells of a stable coarse cell. Cells not simulated inherit the nearest simulated corner. Returns (verdict, n_simulated)."""
    ny, nx = full.shape
    cx, cy = coarse_idx(nx, factor), coarse_idx(ny, factor)
    done = np.zeros(full.shape, bool)
    for j in cy:
        for i in cx:
            done[j, i] = True
    lo_x = np.array([max(c for c in cx if c <= i) for i in range(nx)]); hi_x = np.array([min(c for c in cx if c >= i) for i in range(nx)])
    lo_y = np.array([max(c for c in cy if c <= j) for j in range(ny)]); hi_y = np.array([min(c for c in cy if c >= j) for j in range(ny)])
    need = np.zeros(full.shape, bool)
    for j in range(ny):
        for i in range(nx):
            if not done[j, i]:
                if len({full[lo_y[j], lo_x[i]], full[lo_y[j], hi_x[i]], full[hi_y[j], lo_x[i]], full[hi_y[j], hi_x[i]]}) > 1:
                    need[j, i] = True
    if grow:
        seeds = np.zeros(full.shape, bool)
        for j in cy:
            for i in cx:
                seeds[j, i] = full[j, i]
        near = binary_dilation(seeds, iterations=grow)
        need |= near & ~done
    sim = done | need
    out = np.zeros(full.shape, bool)
    for j in range(ny):
        for i in range(nx):
            if sim[j, i]:
                out[j, i] = full[j, i]
            else:
                jj = lo_y[j] if (j - lo_y[j]) <= (hi_y[j] - j) else hi_y[j]
                ii = lo_x[i] if (i - lo_x[i]) <= (hi_x[i] - i) else hi_x[i]
                out[j, i] = full[jj, ii]
    return out, int(sim.sum())


STRATS = {"f2": (2, 0), "f2+grow1": (2, 1), "f2+grow2": (2, 2), "f3": (3, 0), "f3+grow1": (3, 1), "f3+grow2": (3, 2), "f4+grow2": (4, 2), "f4+grow3": (4, 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("maps"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.maps, "**", "*.pkl"), recursive=True))
    if a.limit:
        files = files[:a.limit]
    acc = {k: dict(n=0, sim=0, wrong=0, lost=0, invented=0, mdiff=[], best_err=[]) for k in STRATS}
    for fn in files:
        m = pickle.load(open(fn, "rb"))
        if not isinstance(m, dict) or "per_face" not in m:
            continue
        step = m["step"]
        for f, v in m["per_face"].items():
            full = np.asarray(v["stable"], bool)
            if full.size < 16:
                continue
            for k, (fac, gr) in STRATS.items():
                est, ns = emulate(full, fac, gr)
                s = acc[k]; s["n"] += full.size; s["sim"] += ns; s["wrong"] += int((est != full).sum())
                s["lost"] += int((full & ~est).sum()); s["invented"] += int((est & ~full).sum())
                mf, me = margin(full, step), margin(est, step)
                s["best_err"].append(float(mf.max() - me.max()))
                if full.any():
                    s["mdiff"].append(float(np.abs(mf - me)[full | est].mean()))
    res = {}
    for k, s in acc.items():
        res[k] = {"simulated_frac": round(s["sim"] / s["n"], 3), "cell_agreement": round(1 - s["wrong"] / s["n"], 4), "stable_lost": s["lost"],
                  "stable_invented": s["invented"], "mean_abs_margin_err_mm": round(float(np.mean(s["mdiff"])), 2),
                  "best_margin_err_mm_p95": round(float(np.percentile(np.abs(s["best_err"]), 95)), 1), "cells": s["n"]}
        print(k, res[k], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
