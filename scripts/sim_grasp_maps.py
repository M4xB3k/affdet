"""Grasp actionability maps from simulated parallel-jaw grasps (see affdet/sim/grasp_map.py).

    python scripts/sim_grasp_maps.py PICKS_JSON GLB_DIR OUT_DIR [--role frying_pan --index 0 1 --step 0.02]
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.grasp_map import grasp_map  # noqa: E402
from affdet.sim.objects import load_body  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb_dir"); ap.add_argument("out")
    ap.add_argument("--role", default="frying_pan")
    ap.add_argument("--index", type=int, nargs="+", default=[0])
    ap.add_argument("--step", type=float, default=0.02)
    a = ap.parse_args()
    picks = json.load(open(a.picks))
    os.makedirs(a.out, exist_ok=True)
    for i in a.index:
        r = picks[a.role][i]
        name = f"{a.role}_{r['object_uid'][:8]}"
        body, mesh = load_body(os.path.join(a.glb_dir, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                               float(r["objathor_mass_kg"]), "obj", cache_dir=os.path.join(a.out, "coacd"))
        m = grasp_map(body, mesh.extents[:2], step=a.step)
        m["extents"] = mesh.extents
        pickle.dump(m, open(os.path.join(a.out, f"{name}.pkl"), "wb"))
        print(name, "grid", m["score"].shape, "any-success cells %.0f%%" % (100 * (m["score"] > 0).mean()),
              "com", np.round(m["com"], 3), flush=True)


if __name__ == "__main__":
    main()
