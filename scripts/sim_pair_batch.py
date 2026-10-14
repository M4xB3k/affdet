"""Run pair support maps (A placed on B) for a list of object pairs; resumable, one JSON line per pair.

    python scripts/sim_pair_batch.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR [--cells 16] [--shakers 86450daa 270b9b45 ...]
                                     [--role-pairs mug:cutting_board pan:tray]

Tops: the chosen shakers (single convex hull each). Bases: every curated pan, mug, board and tray (CoACD parts).
--role-pairs adds role-on-role pairs (every object of the first role on every object of the second).
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

ROLES = ("frying_pan", "mug", "cutting_board", "tray")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out")
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--shakers", nargs="*", default=["86450daa", "270b9b45", "c442c916", "d00ef026"])
    ap.add_argument("--role-pairs", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="redo pairs already done, reusing per-face sweeps that still apply")
    a = ap.parse_args()
    picks = json.load(open(a.picks))
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "coacd")
    srows = {r["object_uid"][:8]: r for r in csv.DictReader(open(a.shaker_csv))}
    bodies = {}

    def get(key):
        if key not in bodies:
            kind, ident = key
            if kind == "shaker":
                r = srows[ident]
                bodies[key] = load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["target_height_cm"]) / 100,
                                        float(r["target_mass_kg"]), f"shaker_{ident}", cache_dir=cache, convex=True)
            else:
                r = picks[kind][int(ident)]
                bodies[key] = load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                                        float(r["objathor_mass_kg"]), f"{kind}{ident}", cache_dir=cache)
        return bodies[key]

    pairs = [(("shaker", s), (role, str(i))) for s in a.shakers for role in ROLES for i in range(len(picks[role]))]
    for rp in a.role_pairs:
        ta, tb = rp.split(":")
        pairs += [((ta, str(i)), (tb, str(j))) for i in range(len(picks[ta])) for j in range(len(picks[tb]))]
    if a.limit:
        pairs = pairs[:a.limit]
    log = os.path.join(a.out, "pairs.jsonl")
    old = {}
    if os.path.exists(log):
        old = {json.loads(l)["pair"]: json.loads(l) for l in open(log)}
    done = set() if a.refresh else set(old)
    if a.refresh and os.path.exists(log):
        os.replace(log, log + ".prev")
    face_ok = {}
    for top_key, base_key in pairs:
        name = f"{top_key[0]}{top_key[1]}_on_{base_key[0]}{base_key[1]}"
        if name in done:
            continue
        top, _ = get(top_key)
        base, base_mesh = get(base_key)
        ext = base_mesh.extents
        step = max(0.005, round(max(ext[:2]) / a.cells, 3))
        t0 = time.time()
        if top_key not in face_ok:
            from affdet.sim.mujoco_support import FACE_DOWN
            from affdet.sim.pair_map import face_supports
            face_ok[top_key] = {f: face_supports(top, f) for f in FACE_DOWN}
        pk = os.path.join(a.out, f"{name}.pkl")
        reuse = pickle.load(open(pk, "rb"))["per_face"] if a.refresh and os.path.exists(pk) else None
        try:
            m = pair_map(base, top, ext[:2], step=step, face_ok=face_ok[top_key], reuse=reuse)
        except ValueError as e:
            rec = {"pair": name, "error": str(e)}
        else:
            m["base_extents"] = ext
            pickle.dump(m, open(os.path.join(a.out, f"{name}.pkl"), "wb"))
            rec = {"pair": name, "grid": list(m["b_any"].shape), "step": step, "skipped": m["skipped"],
                   "b_supported_frac": round(float(m["b_any"].mean()), 3),
                   "a_face_margin_mm": {f: round(v, 1) for f, v in m["a_face_margin_mm"].items()},
                   "a_face_stable_frac": {f: round(v, 3) for f, v in m["a_face_stable_frac"].items()},
                   "best_b_margin_mm": round(float(m["b_margin_mm"].max()), 1), "seconds": round(time.time() - t0, 1)}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
