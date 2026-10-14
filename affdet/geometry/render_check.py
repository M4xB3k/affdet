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


def _opaque_enough(g: trimesh.Trimesh, min_alpha: int) -> trimesh.Trimesh:
    """Raise near-transparent material alpha so glass stays visible without refraction."""
    mat = getattr(g.visual, "material", None)
    f = getattr(mat, "baseColorFactor", None)
    if f is not None and f[3] < min_alpha:
        g = g.copy()
        g.visual.material.baseColorFactor = np.array([*f[:3], min_alpha], np.uint8)
    return g


def render_views(scene: trimesh.Scene, size: int = 320, views: dict | None = None,
                 min_alpha: int = 0, return_cameras: bool = False):
    """Render `views` ({name: (azimuth, elevation) in degrees}); azimuth -90 looks at the front (-y)."""
    views = views or VIEWS
    mesh = scene.to_mesh()
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    radius = np.linalg.norm(hi - lo) / 2
    ps = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35] * 3)
    for name, g in scene.geometry.items():
        if min_alpha:
            g = _opaque_enough(g, min_alpha)
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
    out, cams = {}, {}
    for view, (az, el) in views.items():
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
        cams[view] = pose
    r.delete()
    return (out, cams) if return_cameras else out


def project(points: np.ndarray, pose: np.ndarray, size: int, yfov_deg: float = 30.0) -> np.ndarray:
    """Pixel coordinates of world points for a camera from render_views."""
    f = size / 2 / math.tan(math.radians(yfov_deg) / 2)
    pc = (np.linalg.inv(pose) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    return np.c_[size / 2 + f * pc[:, 0] / -pc[:, 2], size / 2 - f * pc[:, 1] / -pc[:, 2]]


def face_labelled_views(scene: trimesh.Scene, views: dict, size: int = 384) -> dict[str, Image.Image]:
    """Renders with the six box-face names written next to each face the camera sees."""
    from affdet.geometry.face_labels import FACES
    ims, cams = render_views(scene, size, views, min_alpha=110, return_cameras=True)
    lo, hi = scene.to_mesh().bounds
    c, half = (lo + hi) / 2, (hi - lo) / 2
    for view, im in ims.items():
        d = ImageDraw.Draw(im)
        eye = cams[view][:3, 3]
        for name, n in FACES.items():
            n = np.asarray(n, float)
            centre = c + n * half
            if n @ (eye - centre) <= 0:  # face turned away from this camera
                continue
            tip = c + n * (half + 0.25 * np.linalg.norm(hi - lo))
            (x0, y0), (x1, y1) = project(np.stack([centre, tip]), cams[view], size)
            tw = 7 * len(name)
            x1, y1 = np.clip(x1, tw / 2 + 3, size - tw / 2 - 3), np.clip(y1, 20, size - 10)
            d.line([(x0, y0), (x1, y1)], fill=(220, 30, 30), width=2)
            d.rectangle([x1 - tw / 2 - 2, y1 - 8, x1 + tw / 2 + 2, y1 + 8], fill=(255, 255, 255), outline=(220, 30, 30))
            d.text((x1 - tw / 2, y1 - 6), name, fill=(200, 0, 0))
        d.text((4, 4), view, fill="black")
    return ims


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
