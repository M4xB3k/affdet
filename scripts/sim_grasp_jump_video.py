"""Slow-motion close-ups of chosen grasp cells (closing phase only): python scripts/sim_grasp_jump_video.py MAPS PICKS GLB OUT NAME spec:iy,ix,iyaw[:note] ..."""
import os, pickle, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
from sim_grasp_video import render_clip
from affdet.sim.registry import Registry
maps, picks, glb, out, name, *cells = sys.argv[1:]
reg = Registry(picks, glb, os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_set.csv"), os.environ["COACD"])
clips = []
for c in cells:
    spec, idx, note = c.split(":")[0] + ":" + c.split(":")[1], c.split(":")[2], c.split(":")[3]
    iy, ix, iw = map(int, idx.split(","))
    m = pickle.load(open(os.path.join(maps, f"grasp_{reg.spec_name(spec)}.pkl"), "rb"))
    fr, _ = render_clip(reg.get(spec)[0], m, f"{spec} {note}", iy, ix, iw, fps=120, close=True, tmax=1.0)
    clips.append(fr)
d = os.path.join(out, f"frames_{name}"); os.makedirs(d, exist_ok=True)
for k in range(max(map(len, clips))):
    cv = Image.new("RGB", (400 * len(clips), 400), "white")
    for j, c in enumerate(clips): cv.paste(c[min(k, len(c) - 1)], (400 * j, 0))
    cv.save(os.path.join(d, f"{k:05d}.png"))
