"""Render actionability maps onto the object as a robot camera would see them (per-pixel heat).

    python scripts/sim_map_render.py PICKS_JSON GLB_DIR PLACEMENT_DIR GRASP_DIR OUT_DIR

Each map value over (x, y) is copied to the vertices of the (subdivided) mesh by their (x, y) position; only
upward-facing surfaces get heat, since the camera looks from above. Output: one PNG per map.
"""
import glob
import json
import math
import os
import pickle
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from affdet.geometry.prepare import prepare  # noqa: E402
from affdet.geometry.render_check import _look_at  # noqa: E402


def lookup(m, xs, ys, grid):
    ix = np.clip(np.round((xs - m["xs"][0]) / m["step"]).astype(int), 0, len(m["xs"]) - 1)
    iy = np.clip(np.round((ys - m["ys"][0]) / m["step"]).astype(int), 0, len(m["ys"]) - 1)
    return grid[iy, ix]


def render(mesh, colors, size=520, az=-60, el=50):
    sub = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.008, max_iter=10), process=False)
    c = colors(sub)
    sub.visual = trimesh.visual.ColorVisuals(sub, vertex_colors=c)
    lo, hi = sub.bounds
    ctr = (lo + hi) / 2
    r = np.linalg.norm(hi - lo) * 1.15
    a, e = math.radians(az), math.radians(el)
    eye = ctr + r * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    pose = _look_at(eye, ctr)
    sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.6] * 3)
    sc.add(pyrender.Mesh.from_trimesh(sub, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=math.radians(32)), pose=pose)
    sc.add(pyrender.DirectionalLight(intensity=2.0), pose=pose)
    rr = pyrender.OffscreenRenderer(size, size)
    img = rr.render(sc)[0]
    rr.delete()
    return Image.fromarray(img)


def main():
    picks_json, glb_dir, pl_dir, gr_dir, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    picks = json.load(open(picks_json))

    def mesh_of(role, idx):
        r = picks[role][idx]
        return prepare(os.path.join(glb_dir, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100)[1]

    def tile(img, text):
        t = Image.new("RGB", (img.width, img.height + 20), "white")
        t.paste(img, (0, 20))
        ImageDraw.Draw(t).text((4, 4), text, fill="black")
        return t

    for name, role in (("shaker_on_board", "cutting_board"), ("shaker_on_mug", "mug"), ("shaker_on_tray", "tray")):
        f = os.path.join(pl_dir, name + ".pkl")
        if not os.path.exists(f):
            continue
        m = pickle.load(open(f, "rb"))
        mesh = mesh_of(role, 0)
        mx = max(m["margin_mm"].max(), 1.0)

        def colors(sub, m=m, mx=mx):
            up = sub.vertex_normals[:, 2] > 0.5
            v = lookup(m, sub.vertices[:, 0], sub.vertices[:, 1], m["margin_mm"]) / mx
            k = lookup(m, sub.vertices[:, 0], sub.vertices[:, 1], m["kind"])
            rgb = (cm.viridis(v) * 255).astype(np.uint8)
            rgb[k == 0] = [233, 150, 150, 255]  # falls / tips
            rgb[~up] = [200, 200, 200, 255]
            return rgb
        render(mesh, colors, el=78 if role == "mug" else 50).save(os.path.join(out, f"pixels_{name}.png"))
    # grasp maps: the analytic maps (grasp_analytic_<role>_<uid>.pkl in OUT_DIR), coloured by class
    for f in sorted(glob.glob(os.path.join(out, "grasp_analytic_*.pkl"))):
        _, _, role, uid = os.path.basename(f)[:-4].split("_", 3)[0:1] + os.path.basename(f)[:-4].split("_", 3)[1:2] + [
            os.path.basename(f)[:-4].rsplit("_", 1)[0].split("_", 2)[2]] + [os.path.basename(f)[:-4].rsplit("_", 1)[1]]
        idx = [i for i, r in enumerate(picks[role]) if r["object_uid"].startswith(uid)][0]
        m = pickle.load(open(f, "rb"))
        cls = np.where(m["level"].any(2), 2, np.where(m["fits"].any(2), 1, 0))
        mesh = mesh_of(role, idx)

        def colors(sub, m=m, cls=cls):
            k = lookup(m, sub.vertices[:, 0], sub.vertices[:, 1], cls)
            pal = np.array([[70, 70, 70, 255], [232, 176, 75, 255], [90, 200, 120, 255]], np.uint8)
            return pal[k]
        render(mesh, colors, el=60).save(os.path.join(out, f"pixels_grasp_{role}_{uid}.png"))


if __name__ == "__main__":
    main()
