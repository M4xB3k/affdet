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
"""
from __future__ import annotations

import numpy as np
import trimesh

from affdet.sim.mujoco_support import FACE_DOWN, Body, face_holds
from affdet.sim.placement_map import placement_map


MIN_SETTLED_MARGIN_M = 0.005  # same as the static labeller's min_margin: the settled COM must sit this far inside the contact patch


def face_supports(top: Body, face: str) -> bool:
    """The shaker-study verdict for one face: it survives the tilt test and rests with its COM 5 mm inside the patch."""
    r = face_holds(top, face)
    return bool(r["holds"] and r["settled_margin_m"] >= MIN_SETTLED_MARGIN_M)


def pair_map(base: Body, top: Body, extent_xy, step=0.01, faces=tuple(FACE_DOWN), prefilter=True, procs=8,
             face_ok: dict | None = None, reuse: dict | None = None, **kw) -> dict:
    per_face, skipped = {}, []
    for f in faces:
        if prefilter:
            ok = face_ok[f] if face_ok is not None and f in face_ok else face_supports(top, f)
            if not ok:
                skipped.append(f)
                continue
        per_face[f] = reuse[f] if reuse and f in reuse else placement_map(base, top, extent_xy, step=step, top_face=f, procs=procs, **kw)
    if not per_face:
        raise ValueError("no face of the top object holds on a table")
    any_ = next(iter(per_face.values()))
    shape = any_["stable"].shape
    margin = np.stack([per_face[f]["margin_mm"] if f in per_face else np.zeros(shape) for f in faces])
    best_face = np.where(margin.max(0) > 0, margin.argmax(0), -1)
    out = {"faces": list(faces), "skipped": skipped, "xs": any_["xs"], "ys": any_["ys"], "step": step,
           "per_face": per_face,
           "b_margin_mm": margin.max(0), "b_best_face": best_face, "b_any": margin.max(0) > 0,
           "b_stable_frac": np.stack([per_face[f]["stable"] if f in per_face else np.zeros(shape, bool) for f in faces]).mean(0),
           "a_face_margin_mm": {f: (per_face[f]["best_margin_mm"] if f in per_face else 0.0) for f in faces},
           "a_face_stable_frac": {f: (float(per_face[f]["stable"].mean()) if f in per_face else 0.0) for f in faces}}
    return out


def patch_heat(vertices: np.ndarray, face_values: dict, eps=0.003) -> np.ndarray:
    """Per-vertex heat on A: each face's value on the vertices that touch the table when A rests on that face.

    Where patches of several faces overlap (an edge) the larger value wins."""
    heat = np.zeros(len(vertices))
    for face, val in face_values.items():
        R = trimesh.transformations.rotation_matrix(FACE_DOWN[face][1], FACE_DOWN[face][0])[:3, :3]
        z = vertices @ R.T[:, 2]  # height of each vertex when `face` is down
        heat = np.maximum(heat, val * (z - z.min() < eps))
    return heat
