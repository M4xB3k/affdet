"""Pair support maps (a heatmap on each object) for A placed on B, from physics simulation.

    python scripts/sim_pair_maps.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR [--base mug board tray] [--steps 0.005 ...]
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from affdet.sim.objects import load_body  # noqa: E402
from affdet.sim.pair_map import pair_map  # noqa: E402

ROLE = {"board": "cutting_board", "mug": "mug", "tray": "tray"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out")
    ap.add_argument("--base", nargs="+", default=["mug"])
    ap.add_argument("--cells", type=int, default=36, help="grid cells along the base's longer side")
    a = ap.parse_args()
    picks = json.load(open(a.picks))
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "coacd")
    srow = [r for r in csv.DictReader(open(a.shaker_csv)) if r["object_uid"].startswith("270b9b45")][0]
    shaker, shaker_mesh = load_body(os.path.join(a.glb, srow["object_uid"] + ".glb"), float(srow["target_height_cm"]) / 100,
                                    float(srow["target_mass_kg"]), "shaker", cache_dir=cache, convex=True)
    for name in a.base:
        r = picks[ROLE[name]][0]
        base, base_mesh = load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                                    float(r["objathor_mass_kg"]), name, cache_dir=cache)
        ext = base_mesh.extents
        step = max(0.005, round(max(ext[:2]) / a.cells, 3))
        t0 = time.time()
        m = pair_map(base, shaker, ext[:2], step=step)
        m["base_extents"] = ext
        pickle.dump(m, open(os.path.join(a.out, f"pair_shaker_on_{name}.pkl"), "wb"))
        print(name, "grid", m["b_any"].shape, "skipped faces", m["skipped"], "B-side supported %.0f%%" % (100 * m["b_any"].mean()),
              {f: round(v, 1) for f, v in m["a_face_margin_mm"].items()}, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
