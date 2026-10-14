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
              convex: bool = False, friction: float = 0.6, threshold: float = 0.05, max_hulls: int = 24):
    """Return (Body, visual mesh). `convex` uses the single convex hull (exact for table contact, wrong for cups)."""
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
    return Body(name, parts, mass_kg, friction=friction), mesh
