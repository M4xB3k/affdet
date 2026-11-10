"""Full-lift videos of the held cells of one object (lowest tilt first): python scripts/sim_grasp_cells_video.py MAPS PICKS GLB OUT NAME SPEC [--n 4] [--success]"""
import os, pickle, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
from sim_grasp_video import render_clip
from affdet.sim.registry import Registry
maps, picks, glb, out, name, spec = sys.argv[1:7]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4
reg = Registry(picks, glb, os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_set.csv"), os.environ["COACD"])
m = pickle.load(open(os.path.join(maps, f"grasp_{reg.spec_name(spec)}.pkl"), "rb"))
mask = m["success"] if "--success" in sys.argv else m["held"]
idx = np.argwhere(mask); idx = idx[np.argsort(m["swing_deg"][mask])][:n]
print(spec, "cells available", len(np.argwhere(mask)), flush=True)
clips = []
for iy, ix, iw in idx:
    fr, info = render_clip(reg.get(spec)[0], m, f"{spec} " + ("success" if "--success" in sys.argv else "held"), int(iy), int(ix), int(iw))
    print(info["stored"], info["replay"], info["match"], flush=True); clips.append(fr)
d = os.path.join(out, f"frames_{name}"); os.makedirs(d, exist_ok=True)
for k in range(max(map(len, clips))):
    cv = Image.new("RGB", (400 * len(clips), 400), "white")
    for j, c in enumerate(clips): cv.paste(c[min(k, len(c) - 1)], (400 * j, 0))
    cv.save(os.path.join(d, f"{k:05d}.png"))
