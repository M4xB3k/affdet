"""Does a larger margin mean a more robust placement? Re-run stable placements with small random position errors and count survivals.

    python scripts/sim_margin_calibration.py PICKS_JSON GLB_DIR SHAKER_CSV PAIR_DIR(S...) [--per-bin 40] [--jitter-mm 3] [--k 4] [--out OUT.json]

Stable cells are sampled from all maps in margin bins (mm: 0-5, 5-10, 10-20, 20-40, 40+); each is re-simulated `k` times with the release
point shifted by a uniform random error of up to `jitter-mm` in x and y (a robot's positioning error). The survival rate per bin is the
empirical meaning of the score.
"""
import argparse
import glob
import json
import os
import pickle
import random
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

import affdet.sim.mujoco_support as ms  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402

BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
_G = {}


def spec_of(name):
    m = re.match(r"^(shaker)([0-9a-f]{8})$", name)
    if m:
        return f"shaker:{m.group(2)}"
    m = re.match(r"^(.*?)(\d+)$", name)
    return f"{m.group(1)}:{m.group(2)}"


def init(args):
    _G["reg"] = Registry(*args)


def one(job):
    name, face, x, y, margin, dx, dy = job
    tn, bn = name.split("_on_")
    top, _ = _G["reg"].get(spec_of(tn)); base, _ = _G["reg"].get(spec_of(bn))
    try:
        return ms.stack_test(base, top, "under", face, offset=(x + dx, y + dy))["stable"]
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--per-bin", type=int, default=40); ap.add_argument("--jitter-mm", type=float, default=3.0); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--coacd", default="/tmp/coacd_cal")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "*_on_*.pkl"), recursive=True))
    pool = {b: [] for b in BINS}
    for f in files:
        m = pickle.load(open(f, "rb"))
        name = os.path.basename(f)[:-4]
        for face, pf in m["per_face"].items():
            js, is_ = np.nonzero(pf["stable"])
            for j, i in zip(js, is_):
                mg = float(pf["margin_mm"][j, i])
                for b in BINS:
                    if b[0] <= mg < b[1]:
                        pool[b].append((name, face, float(pf["xs"][i]), float(pf["ys"][j]), mg))
    jobs, owner = [], []
    for b in BINS:
        rng.shuffle(pool[b])
        for c in pool[b][: a.per_bin]:
            for _ in range(a.k):
                d = a.jitter_mm / 1000
                jobs.append((*c, rng.uniform(-d, d), rng.uniform(-d, d)))
                owner.append(b)
    with Pool(a.procs, initializer=init, initargs=((a.picks, a.glb, a.shaker_csv, a.coacd),)) as p:
        res = p.map(one, jobs, chunksize=4)
    out = {}
    for b in BINS:
        r = [x for x, o in zip(res, owner) if o == b]
        out[f"{b[0]}-{int(b[1]) if b[1] < 1e8 else 'inf'} mm"] = {"cells": len(r) // a.k, "runs": len(r), "survival": round(float(np.mean(r)), 3) if r else None, "available_cells": len(pool[b])}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
