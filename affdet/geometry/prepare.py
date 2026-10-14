"""Bring an Objaverse GLB into the labeller's object frame.

Steps: load (optionally one half of a two-object model, see
scripts/split_pairs.py), rotate the chosen up axis to +z (glTF default: +Y),
level it onto its base (scanned models are often a few degrees off), yaw so
the footprint's minimum-area rectangle lines up with x/y (kept within ±45° of
the artist's front so it changes little), scale uniformly to the target
height, and put the base centre at the origin.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

# glTF (+Y up, front +Z) -> object frame (+Z up, front -Y)
GLTF_TO_ZUP = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])

_AXES = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}


def up_rotation(up_axis: str) -> np.ndarray:
    """Rotation taking the glTF axis `up_axis` to +z of the object frame."""
    v = np.asarray(_AXES[up_axis], float)
    v = GLTF_TO_ZUP[:3, :3] @ v
    z = np.array([0, 0, 1.0])
    if np.allclose(v, z):
        R = np.eye(4)
    elif np.allclose(v, -z):
        R = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
    else:
        R = trimesh.geometry.align_vectors(v, z)
    return R @ GLTF_TO_ZUP


def load_scene(path: str, half: int | None = None) -> trimesh.Scene:
    """Load a GLB as a scene of world-frame geometries (materials kept)."""
    if half is None:
        scene = trimesh.load(path, force="scene")
        geoms = scene.dump()
    else:
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "split_pairs", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "split_pairs.py")
        sp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sp)
        _, pieces, _, _ = sp.split(path)
        geoms = pieces[half]
    out = trimesh.Scene()
    for i, g in enumerate(geoms):
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            out.add_geometry(g, geom_name=f"part_{i}")
    return out


def level(mesh: trimesh.Trimesh, max_deg: float = 25.0, min_area_frac: float = 0.02) -> np.ndarray | None:
    """Rotation putting the largest flat hull facet within max_deg of "down" exactly down.

    None when there is no such facet (round or pointed base) or it is already level.
    """
    hull = mesh.convex_hull
    down = np.array([0, 0, -1.0])
    groups = [list(f) for f in hull.facets] + [[i] for i in range(len(hull.faces))]
    best, best_area = None, min_area_frac * hull.area
    for g in groups:
        area = hull.area_faces[g].sum()
        n = (hull.face_normals[g] * hull.area_faces[g, None]).sum(0)
        n /= np.linalg.norm(n)
        if n @ down >= math.cos(math.radians(max_deg)) and area > best_area:
            best, best_area = n, area
    if best is None or best @ down > math.cos(math.radians(0.5)):
        return None
    return trimesh.geometry.align_vectors(best, down)


def yaw_align(mesh: trimesh.Trimesh, min_gain: float = 0.03) -> float:
    """Yaw (rad) aligning the footprint's min-area rectangle with x/y, or 0 if it gains < min_gain area."""
    xy = mesh.convex_hull.vertices[:, :2]
    T, size = trimesh.bounds.oriented_bounds_2D(xy)
    aabb = np.prod(xy.max(0) - xy.min(0))
    if aabb <= 0 or np.prod(size) > (1 - min_gain) * aabb:
        return 0.0
    ang = -math.atan2(T[1, 0], T[0, 0])
    return (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4


def prepare(path: str, target_height: float, up_axis: str = "+y", half: int | None = None):
    """Return (scene, mesh) in the object frame at metric scale (metres)."""
    scene = load_scene(path, half)
    scene.apply_transform(up_rotation(up_axis))
    mesh = scene.to_mesh()
    L = level(mesh)
    if L is not None:
        scene.apply_transform(L)
        mesh = scene.to_mesh()
    yaw = yaw_align(mesh)
    if yaw:
        scene.apply_transform(trimesh.transformations.rotation_matrix(-yaw, [0, 0, 1]))
        mesh = scene.to_mesh()
    lo, hi = mesh.bounds
    s = target_height / (hi[2] - lo[2])
    T = np.diag([s, s, s, 1.0])
    T[:3, 3] = -s * np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    scene.apply_transform(T)
    return scene, scene.to_mesh()
