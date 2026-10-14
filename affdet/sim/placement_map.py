"""Placement actionability: where on object A can object B be put down so that it stays?

For every offset on a grid over A's footprint, B is set down there (resting on `top_face`) and the stack is
simulated (mujoco_support.stack_test). The result is a map over A's surface, as in O2O-Afford's "where can I
place it" heatmaps, but with labels from physics rather than from point-wise interaction trials on point clouds.

- `stable`: B stayed on A for that offset.
- `margin_mm`: how far (mm) that offset is from the nearest failing offset. A robust placement point sits in the
  middle of a large stable region, not at its edge. The argmax is the suggested placement point.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from scipy.ndimage import distance_transform_edt

from affdet.sim.mujoco_support import Body, stack_test

RELEASE_PENETRATION_M = 0.003  # B may overlap A by at most this much at the moment it is let go
ON_TOP_DEPTH_M = 0.015  # B resting this far or less below A's highest point is 'on' A (a rim, a board); deeper is 'in' A

_G = {}


def _init(base, top, base_face, top_face):
    _G.update(base=base, top=top, base_face=base_face, top_face=top_face)


def _one(xy):
    r = stack_test(_G["base"], _G["top"], _G["base_face"], _G["top_face"], offset=tuple(xy))
    return (r["n_contacts"], r["drop_m"], r["tilt_deg"], r["drift_m"], r["surface_depth_m"], r["excess_drop_m"], r["w_end"],
            r["v_end"], r["release_penetration_m"])


def placement_map(base: Body, top: Body, extent_xy, step=0.01, base_face="under", top_face="under", procs=8,
                  max_tilt=10.0, max_drift=0.03, max_excess_drop=0.01, max_w=1.5, max_v=0.1,
                  max_release_penetration=RELEASE_PENETRATION_M) -> dict:
    """Grid sweep over x, y in [-extent/2, +extent/2] (A's object frame, centred footprint)."""
    xs = np.arange(-extent_xy[0] / 2, extent_xy[0] / 2 + 1e-9, step)
    ys = np.arange(-extent_xy[1] / 2, extent_xy[1] / 2 + 1e-9, step)
    pts = [(x, y) for y in ys for x in xs]
    with Pool(procs, initializer=_init, initargs=(base, top, base_face, top_face)) as p:
        res = p.map(_one, pts, chunksize=4)
    shape = (len(ys), len(xs))
    contacts = np.array([r[0] for r in res]).reshape(shape)
    out = {"xs": xs, "ys": ys, "step": step,
           "drop_m": np.array([r[1] for r in res]).reshape(shape),
           "tilt_deg": np.array([r[2] for r in res]).reshape(shape),
           "drift_m": np.array([r[3] for r in res]).reshape(shape),
           "surface_depth_m": np.array([r[4] for r in res]).reshape(shape),
           "excess_drop_m": np.array([r[5] for r in res]).reshape(shape),
           "w_end": np.array([r[6] for r in res]).reshape(shape), "v_end": np.array([r[7] for r in res]).reshape(shape),
           "release_penetration_m": np.array([r[8] for r in res]).reshape(shape)}
    out["n_contacts"] = contacts
    return finalize(out, max_tilt, max_drift, max_excess_drop, max_w, max_v, max_release_penetration)


def placement_map_adaptive(base: Body, top: Body, extent_xy, step=0.01, factor=3, base_face="under", top_face="under", procs=8,
                           **kw) -> dict:
    """Coarse-to-fine version of `placement_map`: the same fine grid, far fewer simulations.

    Every `factor`-th cell is simulated first. A fine cell whose four enclosing coarse cells all agree (all stable or all
    failing) inherits that value; the rest, the cells near a boundary, are simulated. A stable island narrower than the
    coarse spacing can be missed, so `factor` trades cost against that risk. `n_simulated` / `n_cells` report the saving.
    """
    xs = np.arange(-extent_xy[0] / 2, extent_xy[0] / 2 + 1e-9, step)
    ys = np.arange(-extent_xy[1] / 2, extent_xy[1] / 2 + 1e-9, step)
    ny, nx = len(ys), len(xs)

    def coarse_idx(n):
        c = list(range(0, n, factor))
        return c if c[-1] == n - 1 else c + [n - 1]

    cx, cy = coarse_idx(nx), coarse_idx(ny)
    fields = ("n_contacts", "drop_m", "tilt_deg", "drift_m", "surface_depth_m", "excess_drop_m", "w_end", "v_end", "release_penetration_m")
    arr = {k: np.zeros((ny, nx)) for k in fields}
    done = np.zeros((ny, nx), bool)

    def run(cells):
        with Pool(procs, initializer=_init, initargs=(base, top, base_face, top_face)) as p:
            res = p.map(_one, [(xs[i], ys[j]) for j, i in cells], chunksize=4)
        for (j, i), r in zip(cells, res):
            for k, v in zip(fields, r):
                arr[k][j, i] = v
            done[j, i] = True

    run([(j, i) for j in cy for i in cx])
    probe = finalize({"xs": xs, "ys": ys, "step": step, **{k: v.copy() for k, v in arr.items()}}, **kw)["stable"]  # valid on done cells
    lo_x = np.array([max(c for c in cx if c <= i) for i in range(nx)]); hi_x = np.array([min(c for c in cx if c >= i) for i in range(nx)])
    lo_y = np.array([max(c for c in cy if c <= j) for j in range(ny)]); hi_y = np.array([min(c for c in cy if c >= j) for j in range(ny)])
    need = []
    for j in range(ny):
        for i in range(nx):
            if done[j, i]:
                continue
            corners = {probe[lo_y[j], lo_x[i]], probe[lo_y[j], hi_x[i]], probe[hi_y[j], lo_x[i]], probe[hi_y[j], hi_x[i]]}
            if len(corners) > 1:
                need.append((j, i))
    run(need)
    # inherited cells copy the metrics of their nearest simulated corner, so `finalize` gives them the same verdict
    for j in range(ny):
        for i in range(nx):
            if not done[j, i]:
                jj = lo_y[j] if (j - lo_y[j]) <= (hi_y[j] - j) else hi_y[j]
                ii = lo_x[i] if (i - lo_x[i]) <= (hi_x[i] - i) else hi_x[i]
                for k in fields:
                    arr[k][j, i] = arr[k][jj, ii]
    out = {"xs": xs, "ys": ys, "step": step, **arr}
    out["n_contacts"] = arr["n_contacts"].astype(int)
    out = finalize(out, **kw)
    out["n_simulated"] = int(done.sum()); out["n_cells"] = int(done.size)
    return out


def finalize(out: dict, max_tilt=10.0, max_drift=0.03, max_excess_drop=0.01, max_w=1.5, max_v=0.1,
             max_release_penetration=RELEASE_PENETRATION_M) -> dict:
    """Turn the raw per-cell metrics into `stable`, `kind`, `margin_mm` and the best point (re-runnable offline)."""
    xs, ys, step, contacts = out["xs"], out["ys"], out["step"], out["n_contacts"]
    shape = contacts.shape
    # B stays if it keeps touching A, does not tip, does not slide relative to A and does not fall further than the
    # surface under it requires. The slide limit is generous: soft contacts creep ~1 cm in 2 s; sliding off is metres.
    stable = ((contacts > 0) & (out["tilt_deg"] < max_tilt) & (out["drift_m"] < max_drift)
              & (out["excess_drop_m"] < max_excess_drop) & (out["w_end"] < max_w) & (out["v_end"] < max_v)
              & (out["release_penetration_m"] <= max_release_penetration))
    out["stable"] = stable
    # soft contacts leave ~0.3 rad/s / 1 cm/s of residual jitter on a stack at rest; only violent motion at the end
    # (an object still tumbling off) is rejected by max_w / max_v
    # 0 = falls off / tips, 1 = stays on an upward surface of A (rim, top), 2 = stays but sits deep inside A (cup, tray)
    out["kind"] = np.where(stable, np.where(out["surface_depth_m"] <= ON_TOP_DEPTH_M, 1, 2), 0)
    pad = np.pad(stable, 1, constant_values=False)  # outside the swept area counts as failing
    out["margin_mm"] = 1000 * step * distance_transform_edt(pad)[1:-1, 1:-1] * stable
    k = np.unravel_index(np.argmax(out["margin_mm"]), shape)
    out["best_xy"] = (float(xs[k[1]]), float(ys[k[0]]))
    out["best_margin_mm"] = float(out["margin_mm"][k])
    out["centre_stable"] = bool(stable[np.argmin(abs(ys)), np.argmin(abs(xs))])
    return out
