"""Share of grasp tests that pass each stage, per object and map folder (a markdown table).

    python scripts/grasp_stage_table.py MAP_DIR [MAP_DIR ...]

Counted over every (cell, yaw) on the object that is not blocked, so the numbers do not depend on how much of the grid the
object covers. Old maps (no clamp fields) show `held` as lift and `held and swing < 30 deg` as level. The fail column counts
the first stage that failed. The two "old" columns use the definitions of the full32 maps and compare across settle times;
clamp at settle 0.1 is 0 % by construction (the pads still press about 300 N, above the 1.5 x 30 N ceiling). Hold force: mean pad force during the hold, over the tests that held.
"""
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.grasp_stages import FAIL, stages  # noqa: E402

NAMES = {v: k for k, v in FAIL.items()}


def row(f):
    m = pickle.load(open(f, "rb"))
    ok = m["on_object"] & ~m["blocked"]
    n = int(ok.sum())
    s = stages(m) if "clamp" not in m else m
    pct = lambda a: "-" if a is None else f"{100 * a[ok].mean():.0f}"
    fails = ""
    if "fail_stage" in s:
        fs = s["fail_stage"][ok]
        fails = ", ".join(f"{NAMES[c]} {100 * (fs == c).mean():.0f}" for c in (2, 3, 4, 5) if (fs == c).any())
    hf = m["hold_force_n"][m["held"]]
    settle = m["params"].get("settle", 0.1)
    old_ok = m["success"] if "success" in m else m["held"] & (m["swing_deg"] < 30)
    return (f"| {os.path.basename(f)[6:-4]} | {os.path.basename(os.path.dirname(f))} | {settle} | {n} | {pct(m['held'])} | {pct(old_ok)} | "
            f"{pct(s['clamp'])} | {pct(s['lift'])} | {pct(s['level'])} | {fails} | {hf.mean() if hf.size else 0:.1f} |")


print("| object | maps | settle (s) | tests (not blocked) | old held % | old held & swing < 30 % | clamp % | lift % | level % | first failure (% of tests) | hold force (N) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
files = sorted((os.path.basename(f), f) for d in sys.argv[1:] for f in glob.glob(os.path.join(d, "grasp_*.pkl")))
for _, f in files:
    print(row(f))
