"""Videos of the grasp simulations behind a stored grasp map, to check that they were done correctly.

    python scripts/sim_grasp_video.py MAP_DIR PICKS_JSON GLB_DIR OUT_DIR --specs frying_pan:0 mug:1 [--coacd-dir D] [--shaker-csv F]

For each spec, replays exactly what `grasp_test` simulated (same settings as stored in the map) for up to four cells
picked from the map: a clean success, a grasp that is held but swings past the level limit, a grasp that slips or is not
lifted, and the cell with the least tilt among the failures. The label shows the outcome stored in the map next to the
outcome of the replay (they must agree: that is the check that the video is the simulation, not an illustration) and
live pad forces. Output: OUT_DIR/grasp_<spec>.mp4, clips side by side, plus OUT_DIR/grasp_<spec>_replay.json.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from affdet.sim.grasp_map import grasp_test  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402

OBJ_RGBA = [0.93, 0.55, 0.15, 1.0]
PAD_RGBA = [0.10, 0.55, 0.95, 1.0]


def pick_cells(m):
    """(label, iy, ix, iyaw) for up to four cells that show different outcomes."""
    ok, held, swing, blocked, on = m["success"], m["held"], m["swing_deg"], m["blocked"], m["on_object"]
    out = []
    if ok.any():  # success with the least tilt, nearest the middle of the successful cells
        idx = np.argwhere(ok)
        sw = swing[ok]
        cy, cx = idx[:, 0].mean(), idx[:, 1].mean()
        d = np.hypot(idx[:, 0] - cy, idx[:, 1] - cx)
        i = int(np.argmin(sw + 0.5 * d))
        out.append(("held and level (success)", *idx[i]))
    sw_mask = held & ~ok & on
    if sw_mask.any():
        idx = np.argwhere(sw_mask)
        i = int(np.argmax(swing[sw_mask]))
        out.append(("held but swings past 30 deg", *idx[i]))
    fail = on & ~held & ~blocked
    if fail.any():
        idx = np.argwhere(fail)
        i = int(np.argmin(m["lifted_m"][fail]))
        out.append(("not held (drops or slips)", *idx[i]))
        i2 = int(np.argmax(m["slip_m"][fail]))
        if i2 != i:
            out.append(("not held, largest slip", *idx[i2]))
    return out[:4]


def render_clip(body, m, label, iy, ix, iyaw, size=400, fps=30, close=False, tmax=None):
    import mujoco
    x, y, yaw = float(m["xs"][ix]), float(m["ys"][iy]), float(m["yaws"][iyaw])
    st = {"rend": None, "frames": [], "k": 0, "every": None, "cam": None, "oid": None, "pads": []}
    ext = float(np.max(m["extents"]))

    def record(mm, data):
        if st["rend"] is None:
            st["rend"] = mujoco.Renderer(mm, size, size)
            st["every"] = max(1, int(1 / fps / mm.opt.timestep))
            st["oid"] = mm.body(body.name).id
            mm.geom_rgba[mm.geom('table').id] = [0.85, 0.85, 0.85, 1.0]
            for g in range(mm.ngeom):
                b = mm.geom_bodyid[g]
                if b == st["oid"]:
                    mm.geom_rgba[g] = OBJ_RGBA
                elif b in (mm.body("fl").id, mm.body("fr").id):
                    mm.geom_rgba[g] = PAD_RGBA
            cam = mujoco.MjvCamera()
            com = data.xipos[st["oid"]]
            lift = float(m['params'].get('lift', 0.3))
            cam.lookat[:] = [com[0], com[1], com[2] + (0.0 if close else 0.5 * lift)]
            cam.distance = max(0.45, 2.2 * ext) + 1.15 * lift if not close else max(0.25, 2.6 * ext)
            cam.azimuth, cam.elevation = 125.0, (-12.0 if not close else -35.0)
            st["cam"] = cam
            st["bl"], st["br"] = mm.body("fl").id, mm.body("fr").id
        if st["k"] % st["every"] == 0 and (tmax is None or data.time <= tmax):
            st["rend"].update_scene(data, st["cam"])
            im = Image.fromarray(st["rend"].render().copy())
            z = float(data.xipos[st["oid"]][2])
            st["frames"].append((im, z, float(data.time)))
        st["k"] += 1

    res = grasp_test(body, x, y, yaw, record=record, params=m["params"])
    if st["rend"] is not None:
        st["rend"].close()
    stored = {"success": bool(m["success"][iy, ix, iyaw]), "held": bool(m["held"][iy, ix, iyaw]),
              "swing_deg": float(m["swing_deg"][iy, ix, iyaw]), "lifted_m": float(m["lifted_m"][iy, ix, iyaw])}
    replay = {"success": bool(res["success"]), "held": bool(res["held"]), "swing_deg": float(res["swing_deg"]),
              "lifted_m": float(res["lifted_m"])}
    agree = stored["success"] == replay["success"] and stored["held"] == replay["held"]
    frames = []
    z0 = st["frames"][0][1] if st["frames"] else 0.0
    for im, z, t in st["frames"]:
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, size, 54], fill=(255, 255, 255))
        d.text((6, 3), label, fill=(0, 0, 0))
        d.text((6, 16), f"cell x={x*100:+.1f} y={y*100:+.1f} cm  yaw={yaw:.0f} deg", fill=(0, 0, 0))
        d.text((6, 29), f"stored: held={stored['held']} swing={stored['swing_deg']:.0f} deg", fill=(0, 0, 0))
        d.text((6, 42), f"replay: held={replay['held']} swing={replay['swing_deg']:.0f} deg  {'MATCH' if agree else 'MISMATCH'}",
               fill=(0, 120, 0) if agree else (200, 0, 0))
        d.rectangle([0, size - 18, size, size], fill=(255, 255, 255))
        d.text((6, size - 15), f"t={t:.1f}s  object lifted {100*(z-z0):.1f} cm", fill=(0, 0, 0))
        frames.append(im)
    return frames, {"label": label, "cell": [x, y, yaw], "stored": stored, "replay": replay, "match": agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps"); ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("out")
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--shaker-csv", default=os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_set.csv"))
    ap.add_argument("--coacd-dir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.coacd_dir or os.path.join(a.out, "coacd"))
    for spec in a.specs:
        name = reg.spec_name(spec)
        m = pickle.load(open(os.path.join(a.maps, f"grasp_{name}.pkl"), "rb"))
        body, _ = reg.get(spec)
        clips, meta = [], []
        for label, iy, ix, iyaw in pick_cells(m):
            fr, info = render_clip(body, m, label, int(iy), int(ix), int(iyaw))
            clips.append(fr); meta.append(info)
            print(spec, label, info["match"], info["stored"], info["replay"], flush=True)
        if not clips:
            continue
        n = max(len(c) for c in clips)
        w = clips[0][0].width
        d = os.path.join(a.out, f"frames_{name}")  # the cluster has no ffmpeg: encode with `ffmpeg -framerate 30 -i frames_X/%05d.png X.mp4`
        os.makedirs(d, exist_ok=True)
        for k in range(n):
            canvas = Image.new("RGB", (w * len(clips), clips[0][0].height), (255, 255, 255))
            for j, c in enumerate(clips):
                canvas.paste(c[min(k, len(c) - 1)], (j * w, 0))
            canvas.save(os.path.join(d, f"{k:05d}.png"))
        if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(d, "%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(a.out, f"grasp_{name}.mp4")], check=True)
        json.dump(meta, open(os.path.join(a.out, f"grasp_{name}_replay.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
