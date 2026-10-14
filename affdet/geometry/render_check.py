"""Offscreen renders of a prepared object for checking uprighting, scale and labels.

Uses pyrender (set PYOPENGL_PLATFORM=osmesa or egl when there is no display).
Each check image shows the object from the front, the right, three-quarter
above and three-quarter below, on a 1 cm grid floor, with the labels as text.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw

VIEWS = {"front": (-90, 5), "right": (0, 5), "3/4 above": (-60, 35), "3/4 below": (-120, -35)}


def _look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, eye
    return M


def render_views(scene: trimesh.Scene, size: int = 320) -> dict[str, Image.Image]:
    mesh = scene.to_mesh()
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    radius = np.linalg.norm(hi - lo) / 2
    ps = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35] * 3)
    for name, g in scene.geometry.items():
        try:
            pm = pyrender.Mesh.from_trimesh(g, smooth=False)
        except Exception:  # broken material: fall back to plain grey
            g = g.copy()
            g.visual = trimesh.visual.ColorVisuals(g, face_colors=[180, 180, 180, 255])
            pm = pyrender.Mesh.from_trimesh(g, smooth=False)
        for node in scene.graph.geometry_nodes.get(name, []):
            ps.add(pm, pose=scene.graph.get(node)[0])
    # floor grid: 1 cm squares
    span = max(0.05, 1.2 * max(hi[0] - lo[0], hi[1] - lo[1]))
    lines = []
    for t in np.arange(-span, span + 1e-9, 0.01):
        lines += [[t, -span, 0], [t, span, 0], [-span, t, 0], [span, t, 0]]
    grid = pyrender.Primitive(positions=np.array(lines, np.float32), mode=pyrender.GLTF.LINES,
                              color_0=np.tile([[0.75, 0.75, 0.75, 1.0]], (len(lines), 1)))
    ps.add(pyrender.Mesh([grid]))
    cam = pyrender.PerspectiveCamera(yfov=math.radians(30))
    light = pyrender.DirectionalLight(intensity=3.0)
    r = pyrender.OffscreenRenderer(size, size)
    out = {}
    for view, (az, el) in VIEWS.items():
        a, e = math.radians(az), math.radians(el)
        d = radius / math.sin(math.radians(15)) * 1.1
        eye = centre + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        pose = _look_at(eye, centre)
        cn = ps.add(cam, pose=pose)
        ln = ps.add(light, pose=pose)
        col, _ = r.render(ps)
        ps.remove_node(cn)
        ps.remove_node(ln)
        out[view] = Image.fromarray(col)
    r.delete()
    return out


def check_sheet(scene: trimesh.Scene, text: list[str], size: int = 320) -> Image.Image:
    views = render_views(scene, size)
    W = size * len(views)
    sheet = Image.new("RGB", (W, size + 14 * (len(text) + 1)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(views.items()):
        sheet.paste(im, (i * size, 0))
        d.text((i * size + 4, 4), name, fill="black")
    for k, line in enumerate(text):
        d.text((6, size + 6 + 14 * k), line, fill="black")
    return sheet
