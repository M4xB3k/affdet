"""Grasp actionability maps (a top-down parallel-jaw gripper, see affdet/sim/grasp_map.py) for object specs.

    python scripts/sim_grasp_maps.py PICKS_JSON GLB_DIR OUT_DIR --specs frying_pan:0 mug:1 shaker:86450daa [--cells 16 | --step 0.02] [--procs 8]
        [--shaker-csv docs/objaverse/salt_shaker_set.csv]

`--cells N` sets the grid step to the object's longer footprint side / N (at least 4 mm); `--step` fixes it. Specs as in affdet/sim/registry.py.
Writes OUT_DIR/grasp_<name>.pkl, `name` as in registry.spec_name.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.grasp_map import grasp_map  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("out")
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--shaker-csv", default=os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_set.csv"))
    ap.add_argument("--cells", type=int, default=0); ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = Registry(a.picks, a.glb, a.shaker_csv, os.path.join(a.out, "coacd"))
    for spec in a.specs:
        f = os.path.join(a.out, f"grasp_{reg.spec_name(spec)}.pkl")
        if os.path.exists(f):
            continue
        body, mesh = reg.get(spec)
        step = max(0.004, round(float(max(mesh.extents[:2])) / a.cells, 3)) if a.cells else a.step
        t0 = time.time()
        m = grasp_map(body, mesh.extents[:2], step=step, procs=a.procs)
        m["extents"] = mesh.extents
        pickle.dump(m, open(f, "wb"))
        print(spec, "grid", m["score"].shape, "step", step, "held cells %.0f%%" % (100 * m["held"].any(2).mean()), "stable cells %.0f%%" % (100 * (m["held"] & (m["swing_deg"] < 30)).any(2).mean()),
              "blocked %.0f%%" % (100 * m["blocked"].all(2).mean()), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
