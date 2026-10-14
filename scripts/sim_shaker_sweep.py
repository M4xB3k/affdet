"""Sweep the simulated tilt over the labelled shakers and store raw metrics.

    python scripts/sim_shaker_sweep.py DOCS_CSV GLB_DIR OUT_JSON [--tilts 3 5 8 12]

For each object, face and tilt, store the raw `face_holds` metrics (worst settle angle, centre-of-mass motion in
the last second, per-direction values). Thresholds are applied afterwards by scripts/sim_shaker_report.py.
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from affdet.geometry.prepare import prepare  # noqa: E402
from affdet.sim.mujoco_support import Body, face_holds  # noqa: E402
from scripts.label_faces import halves  # noqa: E402

FACES = ["on", "under", "front", "back", "left", "right"]


def job(a):
    row, h, glb_dir, tilts = a
    uid = row["object_uid"]
    oid = uid if h is None else f"{uid}_{h}"
    _, m = prepare(os.path.join(glb_dir, f"{uid}.glb"), float(row["target_height_cm"]) / 100, row.get("up_axis") or "+y", h)
    body = Body(oid, [m.convex_hull], float(row["target_mass_kg"] or 0.1))
    return oid, {str(t): {f: face_holds(body, f, tilt_deg=t) for f in FACES} for t in tilts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("glb_dir")
    ap.add_argument("out")
    ap.add_argument("--tilts", type=float, nargs="+", default=[3, 5, 8, 12])
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(r, h, a.glb_dir, a.tilts) for r in csv.DictReader(open(a.csv)) for h in halves(r.get("note"))]
    with Pool(a.procs) as p:
        res = dict(p.map(job, jobs))
    json.dump(res, open(a.out, "w"))
    print(len(res), "objects")


if __name__ == "__main__":
    main()
