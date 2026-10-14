"""Load curated Objaverse objects as simulation bodies (upright, metric, CoACD collision shapes, cached)."""
from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np
import trimesh

from affdet.geometry.prepare import prepare
from affdet.sim.mujoco_support import Body, decompose


def load_body(glb: str, height_m: float, mass_kg: float, name: str, up_axis: str = "+y", cache_dir: str | None = None,
              convex: bool = False, friction: float = 0.6, threshold: float = 0.05, max_hulls: int = 24,
              mass_model: str = "auto"):
    """Return (Body, visual mesh). `convex` uses the single convex hull (exact for table contact, wrong for cups).

    `mass_model` (see `part_masses`): "auto", "shell", "solid" or "uniform" (the old uniform-over-hull-volume)."""
    scene, mesh = prepare(glb, height_m, up_axis)
    if convex:
        parts = [mesh.convex_hull]
    else:
        key = hashlib.md5(f"{os.path.basename(glb)}{height_m}{up_axis}{threshold}{max_hulls}".encode()).hexdigest()[:12]
        f = os.path.join(cache_dir, f"{name}_{key}.pkl") if cache_dir else None
        if f and os.path.exists(f):
            parts = [trimesh.Trimesh(v, fc) for v, fc in pickle.load(open(f, "rb"))]
        else:
            parts = decompose(mesh, threshold=threshold, max_hulls=max_hulls)
            if f:
                os.makedirs(cache_dir, exist_ok=True)
                pickle.dump([(np.asarray(p.vertices), np.asarray(p.faces)) for p in parts], open(f, "wb"))
    pm = None if mass_model == "uniform" else part_masses(mesh, parts, mass_kg, mass_model)
    return Body(name, parts, mass_kg, friction=friction, part_mass=pm), mesh


def part_masses(mesh: trimesh.Trimesh, parts: list, total_kg: float, model: str = "auto", n_samples: int = 6000,
                seed: int = 0) -> list[float]:
    """Split `total_kg` over the convex parts so the centre of mass follows the real material, not the hull volume.

    The CoACD hulls of a pan wall or a mug are chunky and mostly air, so spreading mass uniformly over their volume
    puts the centre of mass in the empty bowl. Two models:
    - "shell": mass proportional to the visual surface area inside each part (uniform wall thickness): right for
      pans, mugs, trays, bottles, anything thin-walled.
    - "solid": mass proportional to part volume: right for boards, blocks, filled objects.
    "auto" picks solid when the mesh is watertight and fills more than half of the summed part volume.
    """
    vols = np.array([p.volume for p in parts])
    if model == "auto":
        solid = mesh.is_watertight and mesh.volume > 0.5 * vols.sum()
        model = "solid" if solid else "shell"
    if model == "solid":
        w = vols
    else:
        pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)  # area-weighted points on the real surface
        # distance to each convex part: the largest signed plane distance (<= 0 inside); the tightest part that
        # contains a point (within 1 mm) owns it
        d = np.stack([((pts[:, None, :] - p.triangles_center[None]) * p.face_normals[None]).sum(2).max(1) for p in parts], 1)
        owner = (np.floor(np.maximum(d, 0) / 0.001) + 0.5 * vols / vols.max()).argmin(1)
        w = np.bincount(owner, minlength=len(parts)).astype(float)
        w += 1e-9
    return list(total_kg * w / w.sum())


def centre_of_mass(parts: list, masses: list[float] | None = None) -> np.ndarray:
    """Mass-weighted centroid of the parts (each part's own centroid, uniform inside a part)."""
    cs = np.array([p.center_mass for p in parts])
    w = np.array(masses if masses is not None else [p.volume for p in parts])
    return (cs * w[:, None]).sum(0) / w.sum()
