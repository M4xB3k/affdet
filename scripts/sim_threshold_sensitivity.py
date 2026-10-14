"""How much do the pair maps change if the stability thresholds change? Re-applies thresholds to the stored raw metrics.

    python scripts/sim_threshold_sensitivity.py PAIR_DIR [PAIR_DIR ...] [--out OUT.json]

For every stored face map the share of cells that are stable is recomputed with one threshold changed at a time (tilt 5/10/15 deg,
slide 1/3/5 cm, overlap at release 1/3/6 mm), and the shift of the "best margin" is reported. No simulation is run.
"""
import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.placement_map import finalize  # noqa: E402

VARIANTS = {"tilt 5": {"max_tilt": 5.0}, "tilt 10 (used)": {}, "tilt 15": {"max_tilt": 15.0},
            "slide 1 cm": {"max_drift": 0.01}, "slide 3 cm (used)": {}, "slide 5 cm": {"max_drift": 0.05},
            "overlap 1 mm": {"max_release_penetration": 0.001}, "overlap 3 mm (used)": {}, "overlap 6 mm": {"max_release_penetration": 0.006}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "*_on_*.pkl"), recursive=True))
    res = {k: {"stable_frac": [], "best_margin": []} for k in VARIANTS}
    for f in files:
        pf = pickle.load(open(f, "rb"))["per_face"]
        for face, m in pf.items():
            if "release_penetration_m" not in m:
                continue
            for k, kw in VARIANTS.items():
                out = finalize({kk: (vv.copy() if hasattr(vv, "copy") else vv) for kk, vv in m.items()}, **kw)
                res[k]["stable_frac"].append(float(out["stable"].mean()))
                res[k]["best_margin"].append(float(out["best_margin_mm"]))
    base = np.array(res["tilt 10 (used)"]["stable_frac"]); basem = np.array(res["tilt 10 (used)"]["best_margin"])
    rows = {}
    for k in VARIANTS:
        sf = np.array(res[k]["stable_frac"]); bm = np.array(res[k]["best_margin"])
        rows[k] = {"mean_stable_frac": round(float(sf.mean()), 4), "mean_change_pts": round(100 * float((sf - base).mean()), 2),
                   "max_abs_change_pts": round(100 * float(np.abs(sf - base).max()), 1), "face_maps_changed_over_5pts": int((np.abs(sf - base) > 0.05).sum()),
                   "mean_best_margin_mm": round(float(bm.mean()), 1), "best_margin_changed": int((np.abs(bm - basem) > 1e-6).sum())}
    print(f"{len(res['tilt 10 (used)']['stable_frac'])} face maps from {len(files)} pairs")
    for k, v in rows.items():
        print("%-20s stable share %.3f (%+.2f pts vs used, worst map %.1f pts, %d maps shift >5 pts); best margin changes in %d maps" % (k, v["mean_stable_frac"], v["mean_change_pts"], v["max_abs_change_pts"], v["face_maps_changed_over_5pts"], v["best_margin_changed"]))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
