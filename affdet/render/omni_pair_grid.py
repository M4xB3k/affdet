"""Pure-Python helpers for omni_pair_render.py — no bpy dependency, so these
are unit-testable without Blender installed. Camera-grid math and instance-id
parsing live here; omni_pair_render.py (bpy-dependent) imports from this module.
"""
from __future__ import annotations

import math

import numpy as np


def parse_grid(spec: str) -> tuple[int, int]:
    """'2x2' -> (2, 2); '3x2' -> (3, 2); '1x1' -> (1, 1) (single baseline angle)."""
    rows, cols = spec.lower().split("x")
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(f"Grid dimensions must be >= 1: {spec!r}")
    return rows, cols


def split_instance(inst: str) -> tuple[str, str]:
    """'durian_030' -> ('durian', 'durian_030'). Handles multi-word categories
    (e.g. 'red_wine_glass_007') by splitting on the last underscore-digit run."""
    tail = inst.rsplit("_", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return tail[0], inst
    raise ValueError(f"Cannot split instance id: {inst!r}")


def resolve_mesh_cat(cat: str) -> str:
    """peeler instances physically live under knife/ on disk (see
    resolved_instances.py docstring) — same remap omni_sampler.py applies
    at scene-generation time, needed here too since we import raw meshes."""
    return "knife" if cat == "peeler" else cat


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, -forward], axis=0)
    trans = -rot @ eye
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


CAMERA_ELEVATION_DEG = 42.0  # fixed elevation when rows == 1
# Safety margin so the bounding sphere doesn't touch the frame edge exactly
# (a sphere is a conservative/looser bound than the actual mesh silhouette
# at most viewing angles, but grazing angles + the AABB-diagonal radius used
# here can still run close to the true edge without a margin).
FRAME_MARGIN = 1.25


def camera_distance_for_fov(radius: float, fov_deg: float, margin: float = FRAME_MARGIN) -> float:
    """Distance at which a sphere of the given radius exactly fills the
    vertical FOV, scaled by a safety margin so nothing touches the frame
    edge. Basic pinhole geometry: half_fov subtends the sphere's radius at
    the returned distance (sin, not tan, since we bound by radius-to-camera
    distance rather than a flat plane)."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    half_fov = math.radians(fov_deg) / 2.0
    dist = radius / math.sin(half_fov)
    return dist * margin


def grid_camera_poses(
    rows: int, cols: int, radius: float, fov_deg: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.08),
) -> list[np.ndarray]:
    """Evenly-spaced, DETERMINISTIC camera poses (not random — pairs must be
    directly comparable to each other and across re-runs), all looking at
    `center` from a distance derived from the real scene's bounding-sphere
    `radius` and the camera's `fov_deg` (see camera_distance_for_fov), so the
    full scene fits in frame regardless of object height/shape. Azimuth spans
    a fixed arc in front of the pair (not full 360, since one side is a
    featureless table); elevation varies across rows from a low 3/4 view to a
    steep top-down view. Row-major order: row 0 (lowest elevation) first,
    each row's columns left-to-right by azimuth."""
    dist = camera_distance_for_fov(radius, fov_deg)
    center_v = np.array(center, dtype=np.float64)
    az_span = math.radians(100.0)  # -50..+50 deg around the front-facing axis
    az_values = (
        [0.0] if cols == 1
        else [-az_span / 2 + az_span * c / (cols - 1) for c in range(cols)]
    )
    el_lo, el_hi = math.radians(28.0), math.radians(75.0)
    el_values = (
        [math.radians(CAMERA_ELEVATION_DEG)] if rows == 1
        else [el_lo + (el_hi - el_lo) * r / (rows - 1) for r in range(rows)]
    )

    poses = []
    for el in el_values:
        for az in az_values:
            offset = np.array([
                dist * math.cos(el) * math.cos(az),
                -dist * math.cos(el) * math.sin(az),
                dist * math.sin(el),
            ], dtype=np.float64)
            eye = center_v + offset
            poses.append(look_at(eye, center_v))
    return poses
