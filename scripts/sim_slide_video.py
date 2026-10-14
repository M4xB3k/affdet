"""Video of the canSlide test: one object pushed at four heights (and one azimuth), side by side.

    python scripts/sim_slide_video.py PICKS_JSON GLB_DIR SHAKER_CSV SPEC OUT_MP4 [--azimuth 0] [--coacd DIR]
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ["MUJOCO_GL"] = "glfw"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from affdet.sim.registry import Registry  # noqa: E402
from affdet.sim.slide_map import slide_test  # noqa: E402


def clip(body, az, frac, size=340):
    frames, rend = [], [None]

    def rec(m, d):
        if rend[0] is None:
            for g in range(m.ngeom):
                b = m.body(m.geom_bodyid[g]).name
                if b == "push":
                    m.geom_rgba[g] = [0.85, 0.2, 0.2, 1]
                elif b == body.name:
                    m.geom_rgba[g] = [0.45, 0.65, 0.9, 1]
            rend[0] = mujoco.Renderer(m, size, size)
        oid = m.body(body.name).id
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [d.xipos[oid][0] + 0.04 * np.cos(np.radians(az)), d.xipos[oid][1] + 0.04 * np.sin(np.radians(az)), 0.05]
        cam.distance = 0.34
        cam.azimuth = az + 90
        cam.elevation = -22
        rend[0].update_scene(d, cam)
        frames.append(Image.fromarray(rend[0].render().copy()))

    def style(xml):  # a bright scene, as in the pair videos
        xml = xml.replace("<asset>", '<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="8" height="8"/>', 1)
        xml = xml.replace('<geom name="table"', '<geom name="table" rgba="0.82 0.82 0.82 1"')
        return xml.replace("<option", "<visual><headlight ambient='0.5 0.5 0.5' diffuse='0.6 0.6 0.6'/></visual><option", 1)

    res = slide_test(body, az, frac, record=rec, xml_hook=style)
    if rend[0] is not None:
        rend[0].close()
    return res, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("spec"); ap.add_argument("out")
    ap.add_argument("--azimuth", type=float, default=0.0); ap.add_argument("--coacd", default="/tmp/coacd_slide")
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.coacd)
    body, _ = reg.get(a.spec)
    fr = (0.15, 0.4, 0.65, 0.9)
    clips = []
    for f in fr:
        r, frames = clip(body, a.azimuth, f)
        why = "slides" if r["success"] else ("topples" if r["tilt_deg"] >= 10 else ("spins" if r["yaw_deg"] >= 25 else "does not slide"))
        print(f, why, round(r["moved_frac"], 2), round(r["tilt_deg"]), round(r["yaw_deg"]), len(frames), flush=True)
        clips.append((f"push at {int(f * 100)}% of the height: {why}", frames, r["success"]))
    hold = 12
    clips = [(l, [f[0]] * hold + f + [f[-1]] * hold, ok) for l, f, ok in clips]
    n = min(len(f) for _, f, _ in clips)
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            g = Image.new("RGB", (680, 740), "white")
            dr = ImageDraw.Draw(g)
            for j, (label, f, ok) in enumerate(clips):
                px, py = (j % 2) * 340, (j // 2) * 360
                g.paste(f[i], (px, py + 20))
                dr.text((px + 4, py + 4), label, fill=(20, 110, 40) if ok else (170, 30, 30))
            dr.text((4, 724), f"{a.spec}: pusher (red) moves 12 cm toward the object at 5 cm/s", fill="black")
            g.save(os.path.join(d, f"{i:04d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", os.path.join(d, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out], check=True)
    print("wrote", a.out, n, "frames")


if __name__ == "__main__":
    main()
