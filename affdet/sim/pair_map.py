"""Pair-level support labels for "A is placed on B": one heatmap on each object.

`placement_map` answers, for one face of A (the one pointing down), where on B it stays. A pair label needs both
sides, so the sweep runs over every face of A:

- B-side map: for each point of B, the best robustness (mm to the nearest failing placement) over A's faces, and
  which face achieves it. "Where on B can A be supported, and how."
- A-side map: for each face of A, the best robustness of any placement with that face down. On A's surface it is
  painted on the face's resting patch (vertices within `eps` of the table plane when A rests on that face).
  "Which part of A can be the supporting contact."

A face that does not hold on a flat table (`face_supports`: rests on that face or a facet within 20 degrees of it,
survives a 5 degree tilt and keeps its COM 5 mm inside the patch, the same verdict as the shaker study) is skipped by default: it cannot hold on a flat board
either, so only concave supports (a cup holding a round cap) could change its answer. Pass prefilter=False to
sweep it anyway. `face_ok` supplies precomputed verdicts and `reuse` earlier per-face sweeps of the same pair.
`adaptive=k` (k >= 2) sweeps coarse-to-fine (`placement_map_adaptive`): about half the simulations for 96 to 98 % of cells
unchanged, at the price of missing a few cells of a narrow stable region (checked in scripts/sim_adaptive_check.py).
"""
from __future__ import annotations

import numpy as np
import trimesh

from affdet.sim.mujoco_support import FACE_DOWN, Body, face_holds, stack_test
from affdet.sim.placement_map import placement_map, placement_map_adaptive


MIN_SETTLED_MARGIN_M = 0.005  # same as the static labeller's min_margin: the settled COM must sit this far inside the contact patch


def face_supports(top: Body, face: str) -> bool:
    """The shaker-study verdict for one face: it survives the tilt test and rests with its COM 5 mm inside the patch."""
    r = face_holds(top, face)
    return bool(r["holds"] and r["settled_margin_m"] >= MIN_SETTLED_MARGIN_M)


def pair_map(base: Body, top: Body, extent_xy, step=0.01, faces=tuple(FACE_DOWN), prefilter=True, procs=8,
             face_ok: dict | None = None, reuse: dict | None = None, adaptive: int = 0, **kw) -> dict:
    per_face, skipped = {}, []
    for f in faces:
        if prefilter:
            ok = face_ok[f] if face_ok is not None and f in face_ok else face_supports(top, f)
            if not ok:
                skipped.append(f)
                continue
        if reuse and f in reuse:
            per_face[f] = reuse[f]
        elif adaptive >= 2:
            per_face[f] = placement_map_adaptive(base, top, extent_xy, step=step, factor=adaptive, top_face=f, procs=procs, **kw)
        else:
            per_face[f] = placement_map(base, top, extent_xy, step=step, top_face=f, procs=procs, **kw)
    if not per_face:
        raise ValueError("no face of the top object holds on a table")
    return aggregate(per_face, faces, skipped, step)


def aggregate(per_face: dict, faces, skipped, step) -> dict:
    """Combine the per-face placement maps of one pair into the maps on B and on A."""
    any_ = next(iter(per_face.values()))
    shape = any_["stable"].shape
    margin = np.stack([per_face[f]["margin_mm"] if f in per_face else np.zeros(shape) for f in faces])
    best_face = np.where(margin.max(0) > 0, margin.argmax(0), -1)
    return {"faces": list(faces), "skipped": list(skipped), "xs": any_["xs"], "ys": any_["ys"], "step": step,
            "per_face": per_face,
            "b_margin_mm": margin.max(0), "b_best_face": best_face, "b_any": margin.max(0) > 0,
            "b_stable_frac": np.stack([per_face[f]["stable"] if f in per_face else np.zeros(shape, bool) for f in faces]).mean(0),
            "a_face_margin_mm": {f: (per_face[f]["best_margin_mm"] if f in per_face else 0.0) for f in faces},
            "a_face_stable_frac": {f: (float(per_face[f]["stable"].mean()) if f in per_face else 0.0) for f in faces}}


def recheck_release(base: Body, top: Body, m: dict, top_face: str, base_face="under", procs=8) -> dict:
    """Re-apply the release-collision rule to a stored placement map: every cell that was stable is re-checked with a
    zero-step `stack_test` (no physics, ~0.1 s), and the map is finalised again. Needed for maps made before the rule."""
    from multiprocessing import Pool

    from affdet.sim.placement_map import finalize
    ys_i, xs_i = np.nonzero(m["stable"])
    pts = [(float(m["xs"][i]), float(m["ys"][j])) for j, i in zip(ys_i, xs_i)]
    pen = np.zeros(m["stable"].shape)
    if pts:
        with Pool(procs, initializer=_init_recheck, initargs=(base, top, base_face, top_face)) as p:
            vals = p.map(_pen, pts, chunksize=4)
        pen[ys_i, xs_i] = vals
    m = dict(m)
    m["release_penetration_m"] = pen
    return finalize(m)


_R = {}


def _init_recheck(base, top, base_face, top_face):
    _R.update(base=base, top=top, base_face=base_face, top_face=top_face)


def _pen(xy):
    return stack_test(_R["base"], _R["top"], _R["base_face"], _R["top_face"], offset=tuple(xy), seconds=0.0)["release_penetration_m"]


def patch_heat(vertices: np.ndarray, face_values: dict, eps=0.003) -> np.ndarray:
    """Per-vertex heat on A: each face's value on the vertices that touch the table when A rests on that face.

    Where patches of several faces overlap (an edge) the larger value wins."""
    heat = np.zeros(len(vertices))
    for face, val in face_values.items():
        R = trimesh.transformations.rotation_matrix(FACE_DOWN[face][1], FACE_DOWN[face][0])[:3, :3]
        z = vertices @ R.T[:, 2]  # height of each vertex when `face` is down
        heat = np.maximum(heat, val * (z - z.min() < eps))
    return heat
