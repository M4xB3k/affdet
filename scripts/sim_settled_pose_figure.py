"""Show where a shaker actually rests in the simulation: settled pose, contact patch (red), centre of mass (green).

    python scripts/sim_settled_pose_figure.py DOCS_CSV GLB_DIR OUT_PNG uid_prefix:face [uid_prefix:face ...]

Each case is one row: the object seen from the side and from below (floor removed), with the vertices within
0.5 mm of the table drawn red and the centre of mass green. Used to adjudicate static-vs-sim disagreements.
"""
import csv
import math
import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import affdet.sim.mujoco_support as ms  # noqa: E402
from affdet.geometry.face_labels import FACES, _resting_directions  # noqa: E402
from affdet.geometry.prepare import prepare  # noqa: E402
from affdet.geometry.render_check import _look_at  # noqa: E402
from scripts.label_faces import halves  # noqa: E402


def settle(body, face, down_dir=None):
    axis, ang = ms.FACE_DOWN[face]
    R = trimesh.transformations.rotation_matrix(ang, axis)[:3, :3]
    if down_dir is not None:
        R = trimesh.geometry.align_vectors(down_dir, [0, 0, -1])[:3, :3]
    q = trimesh.transformations.quaternion_from_matrix(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    m, data, _, _ = ms._simulate([body], [(ms._place(body, R), q)], 3.0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = data.body(body.name).xmat.reshape(3, 3), data.body(body.name).xpos
    return T, data.body(body.name).xipos.copy(), ms.settled_margin(body, data)


def main():
    csv_path, glb_dir, out = sys.argv[1:4]
    rows = {}
    for r in csv.DictReader(open(csv_path)):
        for h in halves(r.get("note")):
            rows[r["object_uid"] if h is None else f"{r['object_uid']}_{h}"] = (r, h)
    S = 300
    rend = pyrender.OffscreenRenderer(S, S)
    tiles = []
    for spec in sys.argv[4:]:
        pre, face = spec.split(":")
        oid = [k for k in rows if k.startswith(pre)][0]
        row, h = rows[oid]
        _, vis = prepare(os.path.join(glb_dir, f"{row['object_uid']}.glb"), float(row["target_height_cm"]) / 100,
                         row.get("up_axis") or "+y", h)
        hull = vis.convex_hull
        body = ms.Body(oid.replace("_", "x"), [hull], float(row["target_mass_kg"] or 0.1))
        r = ms.face_holds(body, face)
        d = None
        if r["candidate_deg"] > 0:
            nrm = np.asarray(FACES[face], float)
            d = _resting_directions(hull, nrm, ms.SETTLE_CANDIDATE_DEG, max_n=4)[:4]
            d = min(d, key=lambda v: abs(math.degrees(math.acos(np.clip(v @ nrm, -1, 1))) - r["candidate_deg"]))
        T, com, margin = settle(body, face, d)
        v = vis.copy()
        v.visual = trimesh.visual.ColorVisuals(v, vertex_colors=[165, 200, 240, 255])
        V = hull.vertices @ T[:3, :3].T + T[:3, 3]
        patch = V[V[:, 2] < V[:, 2].min() + 0.0005]
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.5, 0.5, 0.5])
        sc.add(pyrender.Mesh.from_trimesh(v, smooth=False), pose=T)
        for p in patch[:: max(1, len(patch) // 60)]:
            s = trimesh.creation.icosphere(radius=0.0012)
            s.visual = trimesh.visual.ColorVisuals(s, vertex_colors=[220, 30, 30, 255])
            sc.add(pyrender.Mesh.from_trimesh(s, smooth=False), pose=trimesh.transformations.translation_matrix(p))
        g = trimesh.creation.icosphere(radius=0.002)
        g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=[30, 170, 60, 255])
        sc.add(pyrender.Mesh.from_trimesh(g, smooth=False), pose=trimesh.transformations.translation_matrix(com))
        ext = (V.max(0) - V.min(0)).max()
        c = (V.max(0) + V.min(0)) / 2
        views = []
        for name, eye_dir in (("side", [0.0, -1.0, 0.25]), ("from below", [0.0, -0.35, -1.0])):
            eye = c + 3.0 * ext * np.array(eye_dir) / np.linalg.norm(eye_dir)
            pose = _look_at(eye, c, up=(0, 0, 1) if name == "side" else (0, 1, 0))
            cam = sc.add(pyrender.PerspectiveCamera(yfov=math.radians(35)), pose=pose)
            light = sc.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
            views.append(Image.fromarray(rend.render(sc)[0]))
            sc.remove_node(cam)
            sc.remove_node(light)
        tile = Image.new("RGB", (2 * S, S + 22), "white")
        for i, im in enumerate(views):
            tile.paste(im, (i * S, 22))
        ImageDraw.Draw(tile).text((4, 4), f"{oid[:8]} face={face}  sim holds={r['holds']}  rests {r['candidate_deg']:.0f} deg off the face  "
                                          f"settled margin {1000 * margin:.1f} mm  (red = contact patch, green = COM)", fill="black")
        tiles.append(tile)
    img = Image.new("RGB", (2 * S, len(tiles) * (S + 22)), "white")
    for i, t in enumerate(tiles):
        img.paste(t, (0, i * (S + 22)))
    img.save(out)


if __name__ == "__main__":
    main()
