"""Run pair support maps (A placed on B) for tops x bases; resumable, one JSON line per pair.

    python scripts/sim_pair_batch.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR --tops shaker:86450daa mug:0 bowl:1 --bases mug:0 tray:0
        [--cells 16] [--adaptive 2] [--skip-same]

Specs: `shaker:<first 8 chars of the uid>` or `<role>:<index>` into the picks JSON (affdet/sim/registry.py).
A pair that is already in OUT_DIR/pairs.jsonl is skipped, so an interrupted run can be repeated.
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from affdet.sim.pair_map import face_supports, pair_map  # noqa: E402
from affdet.sim.mujoco_support import FACE_DOWN  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out")
    ap.add_argument("--tops", nargs="+", required=True); ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--cells", type=int, default=16, help="grid cells along the base's longer side")
    ap.add_argument("--adaptive", type=int, default=0, help="coarse-to-fine factor (2 recommended), 0 = full sweeps")
    ap.add_argument("--skip-same", action="store_true", help="skip a spec on itself")
    ap.add_argument("--procs", type=int, default=8, help="worker processes (one per CPU core you have)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = Registry(a.picks, a.glb, a.shaker_csv, os.path.join(a.out, "coacd"))
    log = os.path.join(a.out, "pairs.jsonl")
    done = {json.loads(l)["pair"] for l in open(log)} if os.path.exists(log) else set()
    face_ok = {}
    for top_s in a.tops:
        for base_s in a.bases:
            if a.skip_same and top_s == base_s:
                continue
            name = f"{reg.spec_name(top_s)}_on_{reg.spec_name(base_s)}"
            if name in done:
                continue
            top, _ = reg.get(top_s)
            base, base_mesh = reg.get(base_s)
            if top_s not in face_ok:
                face_ok[top_s] = {f: face_supports(top, f) for f in FACE_DOWN}
            ext = base_mesh.extents
            step = max(0.005, round(max(ext[:2]) / a.cells, 3))
            t0 = time.time()
            try:
                m = pair_map(base, top, ext[:2], step=step, face_ok=face_ok[top_s], adaptive=a.adaptive, procs=a.procs)
            except ValueError as e:
                rec = {"pair": name, "top": top_s, "base": base_s, "error": str(e), "face_ok": face_ok[top_s]}
            else:
                m["base_extents"] = ext
                pickle.dump(m, open(os.path.join(a.out, f"{name}.pkl"), "wb"))
                sims = sum(v.get("n_simulated", v["stable"].size) for v in m["per_face"].values())
                cells = sum(v["stable"].size for v in m["per_face"].values())
                rec = {"pair": name, "top": top_s, "base": base_s, "grid": list(m["b_any"].shape), "step": step, "skipped": m["skipped"],
                       "b_supported_frac": round(float(m["b_any"].mean()), 3),
                       "a_face_margin_mm": {f: round(v, 1) for f, v in m["a_face_margin_mm"].items()},
                       "a_face_stable_frac": {f: round(v, 3) for f, v in m["a_face_stable_frac"].items()},
                       "best_b_margin_mm": round(float(m["b_margin_mm"].max()), 1), "simulations": sims, "cells": cells,
                       "adaptive": a.adaptive, "seconds": round(time.time() - t0, 1)}
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
