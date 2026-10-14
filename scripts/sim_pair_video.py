"""Video of the placement simulation behind a pair map: A is released on B at chosen points and left to settle.

    python scripts/sim_pair_video.py PAIR_PKL PICKS_JSON GLB_DIR SHAKER_CSV TOP_SPEC BASE_SPEC OUT_MP4
    (specs as in affdet/sim/registry.py, e.g. shaker:d00ef026 mug:0 bowl:1)

Four clips side by side, picked from the stored map: the most robust placement, the same face at a point just
outside the stable region, the middle of B (the planner's default), and a point far from every stable one. Each clip replays exactly what `stack_test` simulates (release 2 mm above the local surface,
2 s of physics), at half speed; the label says what the map recorded for that placement.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

os.environ["MUJOCO_GL"] = "glfw"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import affdet.sim.mujoco_support as ms  # noqa: E402
from affdet.sim.objects import load_body  # noqa: E402

TOP_RGBA = [0.93, 0.55, 0.15, 1.0]
BASE_RGBA = [0.55, 0.68, 0.88, 0.38]  # see-through, so a shaker inside the mug stays visible


def clips_from_map(m):
    """Pick (label, face, x, y, stable) tuples from a pair map, each one a placement where something visible happens:
    the most robust point (stays), a nearby point where it tips over inside B, the middle of B (the planner's
    default), and another point where it touches B and fails."""
    faces = m["faces"]
    xs, ys = m["xs"], m["ys"]
    marg = m["b_margin_mm"]
    iy, ix = np.unravel_index(np.argmax(marg), marg.shape)
    face = faces[int(m["b_best_face"][iy, ix])]
    pf = m["per_face"][face]
    X, Y = np.meshgrid(xs, ys)
    d_best = np.hypot(X - xs[ix], Y - ys[iy])
    fail, touch = ~pf["stable"], pf["n_contacts"] > 0
    out = [(f"{face} down, most robust point", face, xs[ix], ys[iy], True)]
    tip = fail & touch & (pf["tilt_deg"] >= 25)
    if tip.any():
        k = np.unravel_index(np.argmin(np.where(tip, d_best, np.inf)), tip.shape)
        out.append((f"{face} down, {1000 * d_best[k]:.0f} mm from the best point", face, X[k], Y[k], False))
    cy, cx = np.argmin(abs(ys)), np.argmin(abs(xs))
    out.append((f"{face} down, middle of B (default)", face, xs[cx], ys[cy], bool(pf["stable"][cx * 0 + cy, cx])))
    slide = fail & touch & ~tip
    if slide.any():
        k = np.unravel_index(np.argmax(np.where(slide, pf["drift_m"], -1)), slide.shape)
        out.append((f"{face} down, another failing point", face, X[k], Y[k], False))
    return out[:4]


def make_clip(base, top, face, x, y, size=360, seconds=2.0, fps=60):
    Rb = np.eye(3)
    Rt = trimesh.transformations.rotation_matrix(ms.FACE_DOWN[face][1], ms.FACE_DOWN[face][0])[:3, :3]
    pb = ms._place(base, Rb)
    ext = np.ptp(np.concatenate([p.vertices for p in top.parts]) @ Rt.T, axis=0)[:2]
    hx, hy = 0.5 * ext
    probe = [(x + fx * hx, y + fy * hy) for fx, fy in [(0, 0), (.8, 0), (-.8, 0), (0, .8), (0, -.8), (.6, .6), (-.6, .6), (.6, -.6), (-.6, -.6)]]
    z_rel = ms.surface_height(base, Rb, pb, probe)
    pt = ms._place(top, Rt, xy=(x, y), z_base=z_rel, gap=0.002)
    q = lambda R: trimesh.transformations.quaternion_from_matrix(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    with tempfile.TemporaryDirectory() as d:
        for b in (base, top):
            ms._write_parts(b, d)
        xml = ms.build_xml([base, top], [(pb, q(Rb)), (pt, q(Rt))])
        xml = xml.replace("<asset>", '<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="8" height="8"/>', 1)
        xml = xml.replace('<geom name="table"', '<geom name="table" rgba="0.82 0.82 0.82 1"')
        xml = xml.replace("<option", "<visual><headlight ambient='0.5 0.5 0.5' diffuse='0.6 0.6 0.6'/></visual><option", 1)
        m = mujoco.MjModel.from_xml_string(xml)
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == m.body(top.name).id:
            m.geom_rgba[g] = TOP_RGBA
        elif b == m.body(base.name).id:
            m.geom_rgba[g] = BASE_RGBA
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    rend = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.085]
    cam.distance = 0.46
    cam.azimuth = 125
    cam.elevation = -52
    frames, every = [], int(1 / fps / m.opt.timestep)
    for k in range(int(seconds / m.opt.timestep) + 1):
        if k % every == 0:
            rend.update_scene(data, cam)
            frames.append(Image.fromarray(rend.render().copy()))
        mujoco.mj_step(m, data)
    rend.close()
    return frames


def main():
    import argparse
    import pickle

    from affdet.sim.registry import Registry
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl"); ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv")
    ap.add_argument("top"); ap.add_argument("base"); ap.add_argument("out")
    a = ap.parse_args()
    m = pickle.load(open(a.pkl, "rb"))
    reg = Registry(a.picks, a.glb, a.shaker_csv, os.path.join(os.path.dirname(os.path.abspath(a.pkl)), "coacd"))
    top, _ = reg.get(a.top)
    base, _ = reg.get(a.base)
    out = a.out
    clips = clips_from_map(m)
    allf = []
    for label, face, x, y, _ in clips:
        r = ms.stack_test(base, top, "under", face, offset=(float(x), float(y)))  # the label comes from the same test as the clip
        why = ""
        if r["release_penetration_m"] > 0.003:
            why = f" (overlaps B by {1000 * r['release_penetration_m']:.0f} mm at release)"
        elif not r["stable"]:
            why = f" (ends {r['tilt_deg']:.0f} deg tilted)" if r["tilt_deg"] >= 10 else ""
        label = f"{label.split(':')[0].split(' (')[0]}: {'stays' if r['stable'] else 'fails'}{why}"
        print(label, round(float(x), 3), round(float(y), 3), flush=True)
        allf.append((label, make_clip(base, top, face, x, y), bool(r["stable"])))
    hold = 48  # frames of stillness before the release and after the end, so the start and result can be read
    allf = [(lab, [fr[0]] * hold + fr + [fr[-1]] * hold, ok) for lab, fr, ok in allf]
    n = min(len(f) for _, f, _ in allf)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            g = Image.new("RGB", (720, 760), "white")
            dr = ImageDraw.Draw(g)
            for j, (label, fr, stable) in enumerate(allf):
                px, py = (j % 2) * 360, (j // 2) * 380
                g.paste(fr[i], (px, py + 20))
                dr.text((px + 4, py + 4), label, fill=(20, 110, 40) if stable else (170, 30, 30))
            t = min(max(i - hold, 0), n - 2 * hold - 1) / 60
            phase = "start pose, nothing moving yet" if i < hold else ("result" if i >= n - hold else "simulating")
            dr.text((4, 744), f"{a.top} on {a.base}: t={t:.2f}s ({phase}). 40% speed. Green: stays on B.", fill="black")
            g.save(os.path.join(d, f"{i:04d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", os.path.join(d, "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
    print("wrote", out, f"{n} frames")


if __name__ == "__main__":
    main()
