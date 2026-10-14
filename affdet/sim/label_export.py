"""Export simulation maps as per-vertex heatmap labels on each object's mesh (the training format).

One `.npz` per (object, role in a label) holding the mesh and one or more per-vertex channels. The mesh is the
object's own upright metric frame (the same frame `affdet.geometry.prepare` produces), subdivided to `max_edge`
so that a label has resolution inside large faces. See docs/sim_labels.md for the schema and the score definitions.

- isP2PSupportive, side B (the supporting object): `margin_mm` and `score` at every upward-facing vertex: how far
  (mm) a placement there can be off before it fails, for the best face of A.
- isP2PSupportive, side A (the object put down): the same value per face of A, painted on the vertices that touch
  the table when A rests on that face.
- canGrasp: `held_frac` (share of gripper yaws that lift and hold the object) and `stable_frac` (... and keep it
  within 30 degrees of level) at every upward-facing vertex, by the (x, y) of the grasp.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh

from affdet.sim.pair_map import patch_heat

SCHEMA_VERSION = 1
CAP_MM = 20.0  # a placement 2 cm from failing counts as fully robust (score 1): about a robot's positioning error


def margin_score(margin_mm, cap_mm=CAP_MM):
    """Robustness in mm to a score in [0, 1]: 0 fails, 1 survives `cap_mm` of error."""
    return np.clip(np.asarray(margin_mm, float) / cap_mm, 0.0, 1.0)


def _lookup(m, x, y, grid):
    ix = np.clip(np.round((x - m["xs"][0]) / m["step"]).astype(int), 0, len(m["xs"]) - 1)
    iy = np.clip(np.round((y - m["ys"][0]) / m["step"]).astype(int), 0, len(m["ys"]) - 1)
    return grid[iy, ix]


def subdivide(mesh: trimesh.Trimesh, max_edge=0.006) -> trimesh.Trimesh:
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=12)
    return trimesh.Trimesh(v, f, process=False)


def pair_labels(pair: dict, top_mesh: trimesh.Trimesh, base_mesh: trimesh.Trimesh, names=("A", "B"), max_edge=0.006) -> dict:
    """The two label arrays of an isP2PSupportive pair map: {'A': {...}, 'B': {...}}, each ready for `save`."""
    b = subdivide(base_mesh, max_edge)
    up = b.vertex_normals[:, 2] > 0.5
    margin_b = np.where(up, _lookup(pair, b.vertices[:, 0], b.vertices[:, 1], pair["b_margin_mm"]), 0.0)
    best_face = np.where(up, _lookup(pair, b.vertices[:, 0], b.vertices[:, 1], pair["b_best_face"]), -1)
    out_b = {"vertices": b.vertices, "faces": b.faces, "valid": up, "margin_mm": margin_b, "score": margin_score(margin_b) * up,
             "best_face": best_face.astype(np.int8)}
    a = subdivide(top_mesh, max_edge)
    fv = {f: float(v) for f, v in pair["a_face_margin_mm"].items()}
    margin_a = patch_heat(a.vertices, fv)
    face_of = np.full(len(a.vertices), -1, np.int8)
    for k, f in enumerate(pair["faces"]):
        if f not in pair["skipped"]:  # faces that cannot rest on a flat table are not labelled (neither 0 nor 1)
            face_of[patch_heat(a.vertices, {f: 1.0}) > 0] = k
    out_a = {"vertices": a.vertices, "faces": a.faces, "valid": face_of >= 0, "margin_mm": margin_a, "score": margin_score(margin_a),
             "face": face_of}
    return {"A": out_a, "B": out_b}


def grasp_labels(grasp: dict, mesh: trimesh.Trimesh, max_swing_deg=30.0, max_edge=0.006) -> dict:
    """Per-vertex canGrasp labels from a simulated grasp map (`grasp_map` output)."""
    m = subdivide(mesh, max_edge)
    up = m.vertex_normals[:, 2] > 0.5
    held = grasp["held"].mean(2)
    stable = (grasp["held"] & (grasp["swing_deg"] < max_swing_deg)).mean(2)
    return {"vertices": m.vertices, "faces": m.faces, "valid": up,
            "held_frac": np.where(up, _lookup(grasp, m.vertices[:, 0], m.vertices[:, 1], held), 0.0),
            "stable_frac": np.where(up, _lookup(grasp, m.vertices[:, 0], m.vertices[:, 1], stable), 0.0)}


def slide_labels(slide: dict, mesh: trimesh.Trimesh, max_edge=0.006) -> dict:
    """Per-vertex canSlide labels from a `slide_map`: each side vertex takes the outcome of the nearest (azimuth, height) push.

    `valid` marks vertices whose surface faces sideways (a horizontal push can reach them); `slide_score` is 1 where that push
    slides the object without tipping or spinning it, 0 where it does not, and `moved_frac`/`tilt_deg`/`yaw_deg` carry the outcome."""
    m = subdivide(mesh, max_edge)
    centre = 0.5 * (m.bounds[0] + m.bounds[1])
    height = m.bounds[1][2] - m.bounds[0][2]
    n = m.vertex_normals
    side = np.hypot(n[:, 0], n[:, 1]) > 0.5
    az = np.degrees(np.arctan2(m.vertices[:, 1] - centre[1], m.vertices[:, 0] - centre[0])) % 360.0
    fr = (m.vertices[:, 2] - m.bounds[0][2]) / max(height, 1e-9)
    A = np.asarray(slide["azimuths"], float); F = np.asarray(slide["fracs"], float)
    ia = np.abs(((az[:, None] - A[None]) + 180.0) % 360.0 - 180.0).argmin(1)
    ifr = np.abs(fr[:, None] - F[None]).argmin(1)
    out = {"vertices": m.vertices, "faces": m.faces, "valid": side}
    for key, name in (("success", "slide_score"), ("moved_frac", "moved_frac"), ("tilt_deg", "tilt_deg"), ("yaw_deg", "yaw_deg")):
        out[name] = np.where(side, np.asarray(slide[key], float)[ifr, ia], 0.0)
    return out


def save(path: str, labels: dict, meta: dict):
    meta = {"schema_version": SCHEMA_VERSION, "cap_mm": CAP_MM, **meta}
    np.savez_compressed(path, meta=json.dumps(meta), **labels)


def load(path: str):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))
