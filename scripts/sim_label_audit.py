"""Audit of the pair maps: re-run randomly chosen placements and show their final frames next to the label.

    python scripts/sim_label_audit.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR PAIR_DIR [PAIR_DIR ...] [--n 36] [--seed 0] [--coacd DIR]

Half of the placements are cells the map calls stable, half are cells it calls failing although the top touched the base (the interesting
failures). Each tile is the settled state with the map's verdict and margin, and the verdict of a fresh run; contact sheets of 6 tiles
are written to OUT_DIR/audit_<k>.png and the table to OUT_DIR/audit.json. A person (or a model) then judges each tile: is the verdict physically right?
"""
import argparse
import glob
import json
import os
import pickle
import random
import re
import sys

os.environ["MUJOCO_GL"] = "glfw"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import affdet.sim.mujoco_support as ms  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402
from scripts.sim_pair_video import make_clip  # noqa: E402


def spec_of(name):
    m = re.match(r"^(shaker)([0-9a-f]{8})$", name)
    if m:
        return f"shaker:{m.group(2)}"
    m = re.match(r"^(.*?)(\d+)$", name)
    return f"{m.group(1)}:{m.group(2)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--n", type=int, default=36); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--coacd", default="/tmp/coacd_audit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "*_on_*.pkl"), recursive=True))
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.coacd)
    picks = []
    for k in range(a.n):
        want_stable = k % 2 == 0
        for _ in range(200):
            f = rng.choice(files)
            m = pickle.load(open(f, "rb"))
            face = rng.choice(list(m["per_face"]))
            pf = m["per_face"][face]
            cond = pf["stable"] if want_stable else (~pf["stable"] & (pf["n_contacts"] > 0))
            js, is_ = np.nonzero(cond)
            if len(js):
                q = rng.randrange(len(js))
                picks.append((os.path.basename(f)[:-4], face, float(pf["xs"][is_[q]]), float(pf["ys"][js[q]]), bool(pf["stable"][js[q], is_[q]]), float(pf["margin_mm"][js[q], is_[q]])))
                break
    tiles, rows = [], []
    for name, face, x, y, st, mg in picks:
        tn, bn = name.split("_on_")
        top, _ = reg.get(spec_of(tn)); base, _ = reg.get(spec_of(bn))
        r = ms.stack_test(base, top, "under", face, offset=(x, y))
        frames, _ = make_clip(base, top, face, x, y, size=300, fps=4, elevation=-45.0, see_through=bool(r["surface_depth_m"] > 0.03))
        img = frames[-1].copy()
        cap = ImageDraw.Draw(img)
        cap.rectangle([0, 0, 300, 28], fill=(255, 255, 255))
        cap.text((3, 2), f"{name} ({face} down)", fill="black")
        cap.text((3, 14), f"map: {'STABLE' if st else 'fails'} {mg:.0f} mm | re-run: {'stays' if r['stable'] else 'fails'} tilt {r['tilt_deg']:.0f}", fill=(20, 110, 40) if st else (170, 30, 30))
        tiles.append(img)
        rows.append({"pair": name, "face": face, "x": x, "y": y, "map_stable": st, "map_margin_mm": mg, "rerun_stable": bool(r["stable"]), "rerun_tilt_deg": float(r["tilt_deg"]), "drift_m": float(r["drift_m"])})
        print(len(rows), name, face, "map", st, "re-run", r["stable"], flush=True)
    for k in range(0, len(tiles), 6):
        sheet = Image.new("RGB", (900, 600), "white")
        for j, t in enumerate(tiles[k:k + 6]):
            sheet.paste(t, ((j % 3) * 300, (j // 3) * 300))
        sheet.save(os.path.join(a.out, f"audit_{k // 6}.png"))
    json.dump(rows, open(os.path.join(a.out, "audit.json"), "w"), indent=1)
    print("map and re-run agree on", sum(r["map_stable"] == r["rerun_stable"] for r in rows), "of", len(rows))


if __name__ == "__main__":
    main()
