"""Analytic grasp actionability: a pinch at (x, y) with closing direction u holds the object level iff the pinch
fits and the pads can resist the torque of the weight about the pinch axis.

- Fit: the object's chord along u through (x, y), at pad height, is at most the gripper opening and the pads
  have something to press on.
- Torque: the weight acts at the centre of mass (COM). About the closing axis (the line through the pinch along
  u) its moment is m g d, where d is the horizontal distance of the COM from that line. Pad friction resists at
  most tau_max = 2 mu_t F (torsional friction of two pads at force F).
- So the grasp holds the object level when m g d <= tau_max, and the object swings (pivots about the pinch until
  the COM hangs below it) otherwise. A pan pinched across its handle has the COM far *along* the handle, so d is
  the whole handle-to-COM distance; pinched radially across its rim wall the closing line passes through the
  COM and d is ~0.

The constants match the simulated gripper in grasp_map.py (mu_t = 0.01 m, F = 30 N), and the model is checked
against the simulation on a uniform bar (swing grows with d; m g d = tau_max at 5.1 cm is about 19 deg of
swing) and on a mug (the analytic "level" set is a subset of the simulated successes).
"""
from __future__ import annotations

import math

import numpy as np

from affdet.sim import grasp_map as gm


def com_of(parts) -> np.ndarray:
    """Volume-weighted centre of mass of convex parts (uniform density, as in the simulation)."""
    vols = np.array([p.volume for p in parts])
    return (np.array([p.center_mass for p in parts]) * vols[:, None]).sum(0) / vols.sum()


def _planes(part):
    n = part.face_normals
    d = np.einsum("ij,ij->i", n, part.vertices[part.faces[:, 0]])
    return n, d


def _interval(planes, p0, u):
    """Parameter interval [t0, t1] where p0 + t u is inside the convex part (None if empty)."""
    n, d = planes
    nu = n @ u
    num = d - n @ p0
    lo, hi = -np.inf, np.inf
    for a, b in zip(nu, num):
        if abs(a) < 1e-12:
            if b < 0:
                return None
        elif a > 0:
            hi = min(hi, b / a)
        else:
            lo = max(lo, b / a)
    return (lo, hi) if lo <= hi else None


def _merge_around_zero(intervals):
    """Connected union of the intervals that contains t = 0 (None if 0 is in none of them)."""
    iv = sorted(i for i in intervals if i is not None)
    cur = None
    for lo, hi in iv:
        if cur is None or lo > cur[1] + 1e-6:
            if cur is not None and cur[0] <= 0 <= cur[1]:
                return cur
            cur = [lo, hi]
        else:
            cur[1] = max(cur[1], hi)
    return tuple(cur) if cur is not None and cur[0] <= 0 <= cur[1] else None


def surface_top(planes_list, x, y):
    """Highest z of the object over (x, y), or None if nothing is there."""
    best = None
    for pl in planes_list:
        iv = _interval(pl, np.array([x, y, 0.0]), np.array([0.0, 0.0, 1.0]))
        if iv is not None and np.isfinite(iv[1]):
            best = iv[1] if best is None else max(best, iv[1])
    return best


def grasp_analytic(parts, mass, xs, ys, yaws_deg, force=gm.FORCE_N, mu_t=gm.PAD_FRICTION[1],
                   opening=2 * gm.PAD["open_half"], pad_below_top=gm.PAD_BELOW_TOP_M) -> dict:
    """Maps over (y, x, yaw): `fits`, `d` (m, COM distance from the closing line), `margin` (tau_max / torque,
    large = comfortable), `level` (fits and holds level)."""
    planes = [_planes(p) for p in parts]
    com = com_of(parts)
    tau_max = 2.0 * mu_t * force
    shape = (len(ys), len(xs), len(yaws_deg))
    fits = np.zeros(shape, bool)
    d_arm = np.full(shape, np.nan)
    margin = np.zeros(shape)
    pad_mid = gm.PAD["half"][2]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            top = surface_top(planes, x, y)
            if top is None:
                continue
            z_mid = max(0.003, top - pad_below_top) + pad_mid
            for ik, yaw in enumerate(yaws_deg):
                a = math.radians(yaw)
                u = np.array([-math.sin(a), math.cos(a), 0.0])
                p0 = np.array([x, y, z_mid])
                iv = _merge_around_zero([_interval(pl, p0, u) for pl in planes])
                if iv is None:
                    continue
                width = iv[1] - iv[0]
                if not (0.002 <= width <= opening):
                    continue
                fits[iy, ix, ik] = True
                dx, dy = com[0] - x, com[1] - y
                d = abs(dx * u[1] - dy * u[0])  # horizontal distance of the COM from the closing line
                d_arm[iy, ix, ik] = d
                torque = mass * 9.81 * d
                margin[iy, ix, ik] = tau_max / max(torque, 1e-9)
    level = fits & (margin >= 1.0)
    return {"xs": np.asarray(xs), "ys": np.asarray(ys), "yaws": list(yaws_deg), "fits": fits, "d": d_arm,
            "margin": margin, "level": level, "com": com, "tau_max": tau_max, "d_max": tau_max / (mass * 9.81)}
