import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
from sim_grasp_video import render_clip
from affdet.sim.registry import Registry
maps, picks, glb, coacd, phys, out, name, *items = sys.argv[1:]
reg = Registry(picks, glb, os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_set.csv"), coacd)
d = json.load(open(phys)); clips = []
for it in items:  # spec:stratum
    spec, stratum = it.split(":")[0] + ":" + it.split(":")[1], it.split(":")[2]
    k = reg.spec_name(spec); m = pickle.load(open(os.path.join(maps, f"grasp_{k}.pkl"), "rb"))
    r = [x for x in d[k]["rows"] if x["stratum"] == stratum][0]; iy, ix, iw = r["cell"]
    fr, info = render_clip(reg.get(spec)[0], m, f"{spec} {stratum}", iy, ix, iw, fps=60, close=True, tmax=2.2)
    print(it, r["reason"], info["match"], flush=True); clips.append(fr)
dd = os.path.join(out, f"frames_{name}"); os.makedirs(dd, exist_ok=True)
for k in range(max(map(len, clips))):
    cv = Image.new("RGB", (400 * len(clips), 400), "white")
    for j, c in enumerate(clips): cv.paste(c[min(k, len(c) - 1)], (400 * j, 0))
    cv.save(os.path.join(dd, f"{k:05d}.png"))
