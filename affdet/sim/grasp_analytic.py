"""Analytic grasp actionability: a pinch at (x, y) with closing direction u holds the object level iff the pinch
fits and the pads can resist the torque of the weight about both pinch axes.

- Fit: the object's chord along u through (x, y), at pad height, is at most the gripper opening and the pads
  have something to press on.
- Torque: the weight acts at the centre of mass (COM). About the closing axis (the line through the pinch along
  u) its moment is m g d, where d is the horizontal distance of the COM from that line. Pad friction resists at
  most tau_max = 2 mu_t F (torsional friction of two pads at force F).
- The weight also pitches the object about the horizontal axis perpendicular to the closing line when the COM is
  offset *along* the closing direction by e (a pan pinched radially at its rim has e = the whole radius). The pads
  resist that only with a couple of their normal forces: m g e <= F L, with L = PITCH_ARM_M, about half the pad
  height (calibrated on the simulated grasps below).
- So the grasp holds the object level when both torques are within capacity, and the object swings (pivots about
  the pinch until the COM hangs below it) otherwise. A pan pinched across its handle has the COM far *along* the handle, so d is
  the whole handle-to-COM distance (the roll torque); pinched radially across its rim wall d is ~0 but e is the
  whole radius (the pitch torque). Neither holds a heavy pan level.

The constants match the simulated gripper in grasp_map.py (mu_t = 0.01 m, F = 30 N), and the model is checked
against the simulation on a uniform bar (swing grows with d; m g d = tau_max at 5.1 cm is about 19 deg of
swing) and on a mug (the analytic "level" set is a subset of the simulated successes).
"""
from __future__ import annotations

import math

import numpy as np

from affdet.sim import grasp_map as gm
from affdet.sim.objects import centre_of_mass

# Lever arm of the normal-force couple that resists pitch. The pad is 2.5 cm tall, so the couple could be up to F x 2.5 cm,
# but a contact patch on a wall of limited height cannot sit at both pad edges: on 272 simulated grasps (2 pans and a
# mug) every cell whose weight torque exceeded F x 1.25 cm swung 40 deg or more, and all below F x 0.6 cm swung < 30.
PITCH_ARM_M = gm.PAD["half"][2]


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
                   opening=2 * gm.PAD["open_half"], pad_below_top=gm.PAD_BELOW_TOP_M, part_mass=None) -> dict:
    """Maps over (y, x, yaw): `fits`, `d` (m, COM distance from the closing line), `margin` (tau_max / torque,
    large = comfortable), `level` (fits and holds level)."""
    planes = [_planes(p) for p in parts]
    com = com_of(parts) if part_mass is None else centre_of_mass(parts, part_mass)
    tau_max = 2.0 * mu_t * force
    shape = (len(ys), len(xs), len(yaws_deg))
    fits = np.zeros(shape, bool)
    d_arm = np.full(shape, np.nan)
    margin = np.zeros(shape)
    pad_mid = gm.PAD["half"][2]
    c0 = gm.PAD["open_half"] + gm.PAD["half"][1]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            top = surface_top(planes, x, y)
            if top is None:
                continue
            for ik, yaw in enumerate(yaws_deg):
                a = math.radians(yaw)
                u = np.array([-math.sin(a), math.cos(a), 0.0])
                # reach: neither pad may start inside the object, so whatever stands under a pad footprint (a mug's
                # far wall next to a pinch on the floor slab) sets the lowest pad height; above the surface = blocked
                tv = np.array([-u[1], u[0]])
                hx, hy, _ = gm.PAD["half"]
                floor = 0.0
                for sgn in (-1, 1):
                    cx, cy = x + sgn * c0 * u[0], y + sgn * c0 * u[1]
                    for a_ in (-hx, 0.0, hx):
                        for b_ in (-hy, 0.0, hy):
                            t_ = surface_top(planes, cx + a_ * tv[0] + b_ * u[0], cy + a_ * tv[1] + b_ * u[1])
                            floor = max(floor, t_ or 0.0)
                z0 = max(0.003, top - pad_below_top, floor + 0.001)
                if z0 > top - 0.008:
                    continue
                z_mid = z0 + pad_mid
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
                e = abs(dx * u[0] + dy * u[1])  # ... and its offset along the closing direction
                d_arm[iy, ix, ik] = d
                roll = tau_max / max(mass * 9.81 * d, 1e-9)
                pitch = force * PITCH_ARM_M / max(mass * 9.81 * e, 1e-9)
                margin[iy, ix, ik] = min(roll, pitch)
    level = fits & (margin >= 1.0)
    return {"xs": np.asarray(xs), "ys": np.asarray(ys), "yaws": list(yaws_deg), "fits": fits, "d": d_arm,
            "margin": margin, "level": level, "com": com, "tau_max": tau_max, "d_max": tau_max / (mass * 9.81)}
