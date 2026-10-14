"""canSlide maps (push an object across the table from every side and height) for object specs.

    python scripts/sim_slide_maps.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR --specs shaker:86450daa mug:0 bowl:1 [--procs 8]
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from affdet.sim.registry import Registry  # noqa: E402
from affdet.sim.slide_map import slide_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out")
    ap.add_argument("--specs", nargs="+", required=True); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = Registry(a.picks, a.glb, a.shaker_csv, os.path.join(a.out, "coacd"))
    for spec in a.specs:
        f = os.path.join(a.out, f"slide_{reg.spec_name(spec)}.pkl")
        if os.path.exists(f):
            continue
        body, _ = reg.get(spec)
        t0 = time.time()
        m = slide_map(body, procs=a.procs)
        pickle.dump(m, open(f, "wb"))
        ok = m["success"][m["on_object"]].mean() if m["on_object"].any() else 0.0
        print(spec, "slides in %.0f %% of pushes" % (100 * ok), "by height:", [round(float(m["success"][i][m["on_object"][i]].mean()), 2) if m["on_object"][i].any() else None for i in range(len(m["fracs"]))],
              "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
