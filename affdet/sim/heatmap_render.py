"""Render per-pixel contact heatmaps from simulated contacts (prototype).

Contacts are turned into per-vertex heat on the object's mesh (affdet.sim.mujoco_support.heat_on_mesh),
then rendered from a robot-like camera. Two label images per object:
- visible: heat only where the contact surface is visible to the camera;
- amodal: heat wherever the contact surface projects, also through the object
  (the underside of a shaker, which the camera never sees).
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pyrender
import trimesh

from affdet.geometry.render_check import _look_at


def _camera_pose(mesh, az_deg=-60, el_deg=40, scale=1.1):
    lo, hi = mesh.bounds
    c = (lo + hi) / 2
    r = np.linalg.norm(hi - lo) / 2 / math.sin(math.radians(15)) * scale
    a, e = math.radians(az_deg), math.radians(el_deg)
    eye = c + r * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return _look_at(eye, c)


def render_heat(mesh: trimesh.Trimesh, heat: np.ndarray, size=512, az=-60, el=40, thresh=0.3):
    """Return (rgb with visible heat, visible heat map, amodal heat map) as arrays."""
    pose = _camera_pose(mesh, az, el)
    cam = pyrender.PerspectiveCamera(yfov=math.radians(30))
    r = pyrender.OffscreenRenderer(size, size)

    def render(m, colors):
        m = m.copy()
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=colors)
        sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.45] * 3)
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        sc.add(cam, pose=pose)
        sc.add(pyrender.DirectionalLight(intensity=2.5), pose=pose)
        return r.render(sc)

    grey = np.array([200, 200, 200, 255], np.uint8)
    hot = np.array([224, 49, 49, 255], np.uint8)
    cols = np.where((heat > thresh)[:, None], hot, grey)
    rgb, depth = render(mesh, cols)
    # heat-only geometry (faces whose vertices are all hot), rendered alone: amodal projection
    face_hot = (heat[mesh.faces] > thresh).all(1)
    if face_hot.any():
        hm = mesh.submesh([np.nonzero(face_hot)[0]], append=True)
        hm = trimesh.Trimesh(np.vstack([hm.vertices, hm.vertices]),
                             np.vstack([hm.faces, hm.faces[:, ::-1] + len(hm.vertices)]),
                             process=False)  # double-sided: undersides face away from the camera
        _, hdepth = render(hm, np.tile(hot, (len(hm.vertices), 1)))
        amodal = hdepth > 0
        visible = amodal & (np.abs(hdepth - depth) < 0.002)
    else:
        amodal = visible = np.zeros((size, size), bool)
    r.delete()
    return rgb, visible, amodal
