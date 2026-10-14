"""Check coarse-to-fine placement sweeps against full sweeps already on disk.

    python scripts/sim_adaptive_check.py PICKS_JSON GLB_DIR SHAKER_CSV PAIR_DIR [--factor 3]
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.objects import load_body  # noqa: E402
from affdet.sim.placement_map import placement_map, placement_map_adaptive  # noqa: E402

CASES = [("shaker270b9b45", "cutting_board0", "under"), ("shakerd00ef026", "mug0", "under"), ("shakerd00ef026", "mug0", "front"),
         ("frying_pan0", "tray0", "under"), ("shakerc442c916", "frying_pan0", "on")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("pair_dir")
    ap.add_argument("--factor", type=int, default=3)
    a = ap.parse_args()
    picks = json.load(open(a.picks))
    srows = {r["object_uid"][:8]: r for r in csv.DictReader(open(a.shaker_csv))}
    cache = os.path.join(a.pair_dir, "coacd")

    def get(name):
        if name.startswith("shaker"):
            r = srows[name[6:]]
            return load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["target_height_cm"]) / 100,
                             float(r["target_mass_kg"]), name, cache_dir=cache, convex=True)[0]
        role = name.rstrip("0123456789"); r = picks[role][int(name[len(role):])]
        return load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                         float(r["objathor_mass_kg"]), f"{role}{name[len(role):]}", cache_dir=cache)[0]

    out = []
    for top, base, face in CASES:
        stored = pickle.load(open(os.path.join(a.pair_dir, f"{top}_on_{base}.pkl"), "rb"))["per_face"][face]
        ext = (np.ptp(stored["xs"]), np.ptp(stored["ys"]))
        tb, bb = get(top), get(base)
        t0 = time.time()
        full = placement_map(bb, tb, ext, step=stored["step"], top_face=face)  # a fresh full sweep, same bodies and names
        t_full = time.time() - t0
        t0 = time.time()
        m = placement_map_adaptive(bb, tb, ext, step=stored["step"], factor=a.factor, top_face=face)
        dt = time.time() - t0
        noise = float((full["stable"] != stored["stable"]).mean())  # run-to-run: same physics, different body names
        assert m["stable"].shape == full["stable"].shape, (m["stable"].shape, full["stable"].shape)
        agree = float((m["stable"] == full["stable"]).mean())
        missed = int((full["stable"] & ~m["stable"]).sum()); extra = int((~full["stable"] & m["stable"]).sum())
        rec = {"case": f"{top} on {base} ({face})", "cells": m["n_cells"], "simulated": m["n_simulated"],
               "saving": round(1 - m["n_simulated"] / m["n_cells"], 2), "cell_agreement": round(agree, 3),
               "stable_cells_full": int(full["stable"].sum()), "missed": missed, "extra": extra,
               "best_margin_full_mm": round(float(full["margin_mm"].max()), 1), "best_margin_adaptive_mm": round(float(m["margin_mm"].max()), 1),
               "seconds_adaptive": round(dt, 1), "seconds_full": round(t_full, 1), "stored_vs_fresh_full_disagreement": round(noise, 3)}
        print(json.dumps(rec), flush=True)
        out.append(rec)
    json.dump(out, open(os.path.join(a.pair_dir, f"adaptive_check_f{a.factor}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
