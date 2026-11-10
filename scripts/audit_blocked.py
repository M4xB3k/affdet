"""Audit of the grasp simulator's 'blocked' verdict on plates and cutting boards (see grasp_test in affdet/sim/grasp_map.py).

A cell is blocked when the pads cannot start at a height that reaches the pinched surface:
    z0 = max(0.003, z_top - pad_below_top, floor + 0.001); blocked if z0 > z_top - 0.008
with z_top the surface under the grasp centre and floor the highest surface under either pad footprint (3x3 points per pad).
This script re-derives the verdict (1) on the simulator's collision hulls and (2) on the true visual mesh, and varies the 8 mm
rule and the pre-grasp opening; then runs the physics on the cells a narrower opening would unblock.
    python audit_blocked.py MAPS PICKS GLB COACD SHAKER_CSV OUT.json SPEC...
"""
import json, os, pickle, sys, math
from multiprocessing import Pool
import numpy as np
import trimesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import affdet.sim.mujoco_support as ms
from affdet.sim.registry import Registry
from affdet.sim.grasp_map import grasp_test, grasp_params, DEFAULTS

GRID = 0.002
A = {}


def heightfield(mesh, half, zoff):
    xs = np.arange(-half[0], half[0] + 1e-9, GRID); ys = np.arange(-half[1], half[1] + 1e-9, GRID)
    X, Y = np.meshgrid(xs, ys)
    o = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 2.0)]
    loc, ir, _ = mesh.ray.intersects_location(o, np.tile([0, 0, -1.0], (len(o), 1)), multiple_hits=True)
    H = np.zeros(len(o))
    np.maximum.at(H, ir, loc[:, 2] + zoff)
    return xs, ys, H.reshape(X.shape)


def look(xs, ys, H, px, py):
    ix = np.clip(np.round((px - xs[0]) / GRID).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((py - ys[0]) / GRID).astype(int), 0, len(ys) - 1)
    return H[iy, ix]


def blocked(z_top, floor, tol=0.008, pad_below=0.03):
    z0 = np.maximum(np.maximum(0.003, z_top - pad_below), floor + 0.001)
    return z0 > z_top - tol


def cellgeom(cells, xs_c, ys_c, yaws, hf, open_half):
    """z_top and floor for each (iy, ix, iw) with pads opened to open_half (pad centre at open_half + pad_half_y)."""
    hx, hy = DEFAULTS["pad_half"][0], DEFAULTS["pad_half"][1]
    c0 = open_half + hy
    x = xs_c[cells[:, 1]]; y = ys_c[cells[:, 0]]; yaw = np.radians(np.array(yaws)[cells[:, 2]])
    u = np.c_[-np.sin(yaw), np.cos(yaw)]; t = np.c_[-u[:, 1], u[:, 0]]
    z_top = look(*hf, x, y)
    floor = np.zeros(len(x))
    for s in (-1, 1):
        for a in (-hx, 0, hx):
            for b in (-hy, 0, hy):
                px = x + s * c0 * u[:, 0] + a * t[:, 0] + b * u[:, 0]
                py = y + s * c0 * u[:, 1] + a * t[:, 1] + b * u[:, 1]
                floor = np.maximum(floor, look(*hf, px, py))
    return z_top, floor


def init(args):
    A["reg"] = Registry(*args)


def work(a):
    maps, spec, name = a
    reg = A["reg"]
    m = pickle.load(open(os.path.join(maps, f"grasp_{name}.pkl"), "rb"))
    body, mesh = reg.get(spec)
    pos = ms._place(body, np.eye(3))
    hull = trimesh.util.concatenate(body.parts)
    ext = np.abs(np.concatenate([hull.bounds[:, :2], mesh.bounds[:, :2]])).max()
    half = (ext + 0.08, ext + 0.08)
    hh = heightfield(hull, half, pos[2]); hm = heightfield(mesh, half, pos[2])
    on = m["on_object"]
    cells = np.argwhere(on)
    xs_c, ys_c, yaws = m["xs"], m["ys"], m["yaws"]
    sb = m["blocked"][on]
    out = {"spec": spec, "n_cells": int(len(cells)), "stored_blocked": int(sb.sum())}
    res = {}
    for lab, hf in (("hull", hh), ("mesh", hm)):
        zt, fl = cellgeom(cells, xs_c, ys_c, yaws, (hf[0], hf[1], hf[2]), DEFAULTS["open_half"])
        res[lab] = (zt, fl)
        b = blocked(zt, fl)
        out[f"blocked_{lab}"] = int(b.sum())
        if lab == "hull":
            out["agree_stored_vs_hull"] = float((b == sb).mean())
    zt, fl = res["mesh"]; ztH, flH = res["hull"]
    mg = zt - fl
    out["margin_mm_pct"] = {str(q): float(np.percentile(mg, q) * 1000) for q in (5, 25, 50, 75, 95)}
    out["same_or_higher_surface"] = int((mg <= 0.001).sum())          # a pad footprint touches a surface as high as the pinch: no threshold helps
    out["too_thin_to_start_low"] = int((zt < 0.011).sum())             # pad centre cannot go below 3 mm: need 8 mm of surface above it
    for tol in (0.003, 0.005, 0.008, 0.012):
        out[f"blocked_mesh_tol{int(tol*1000)}mm"] = int(blocked(zt, fl, tol=tol).sum())
    out["hull_blocks_but_mesh_not"] = int((blocked(ztH, flH) & ~blocked(zt, fl)).sum())
    out["mesh_blocks_but_hull_not"] = int((~blocked(ztH, flH) & blocked(zt, fl)).sum())
    out["max_hull_minus_mesh_top_mm"] = float(np.max(hh[2] - hm[2]) * 1000)
    out["mean_hull_minus_mesh_top_mm"] = float(np.mean((hh[2] - hm[2])[hm[2] > 0.004]) * 1000)
    flips = {}
    base_b = blocked(zt, fl)
    for oh in (0.005, 0.01, 0.015, 0.025):
        z2, f2 = cellgeom(cells, xs_c, ys_c, yaws, hm, oh)
        nb = blocked(z2, f2)
        flips[oh] = np.where(base_b & ~nb)[0]
        out[f"unblocked_by_open_half_{oh}"] = int(len(flips[oh]))
    # physics on cells a narrower pre-grasp unblocks
    rng = np.random.default_rng(0)
    phys = []
    for oh in (0.025, 0.01, 0.005):
        idx = flips[oh]
        for i in rng.choice(idx, min(8, len(idx)), replace=False) if len(idx) else []:
            iy, ix, iw = cells[i]
            r = grasp_test(body, float(xs_c[ix]), float(ys_c[iy]), float(yaws[iw]), params=grasp_params(open_half=oh))
            phys.append({"open_half": oh, "cell": [int(iy), int(ix), int(iw)], "held": r["held"], "success": r["success"], "swing_deg": r["swing_deg"],
                         "lifted_m": r["lifted_m"], "blocked": bool(r.get("blocked", False))})
    out["physics_narrow"] = phys
    return name, out


if __name__ == "__main__":
    maps, picks, glb, coacd, csv_, outp, *specs = sys.argv[1:]
    reg_args = (picks, glb, csv_, coacd)
    reg = Registry(*reg_args)
    jobs = [(maps, s, reg.spec_name(s)) for s in specs]
    with Pool(int(os.environ.get("PROCS", 16)), initializer=init, initargs=(reg_args,)) as p:
        res = dict(p.imap_unordered(work, jobs))
    json.dump(res, open(outp, "w"), indent=1)
    print("done", len(res))
