"""Compare static and simulated face labels on the shaker set, across tilt angles and thresholds.

    python scripts/sim_shaker_report.py DOCS_CSV GLB_DIR SWEEP_JSON OUT_JSON

Static labels are recomputed with min_tip_deg = the simulated tilt, so both tests ask the same question
("does the face survive a T degree disturbance?"). Writes per-(object, face, tilt) static and sim verdicts.
"""
import argparse
import csv
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from affdet.geometry.face_labels import Params, label_faces  # noqa: E402
from affdet.geometry.prepare import prepare  # noqa: E402
from scripts.label_faces import halves  # noqa: E402


def sim_holds(m, settle_deg=2.0, drift_m=0.005):
    """Re-apply thresholds to the stored raw metrics (ref drop must have settled: worst_off_deg < 180)."""
    if m["worst_off_deg"] >= 180.0 or not m.get("per_dir"):
        return False
    return all(off < settle_deg and mot < drift_m for off, mot in m["per_dir"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("glb_dir"); ap.add_argument("sweep"); ap.add_argument("out")
    a = ap.parse_args()
    sweep = json.load(open(a.sweep))
    tilts = sorted({float(t) for v in sweep.values() for t in v})
    out = {}
    for row in csv.DictReader(open(a.csv)):
        for h in halves(row.get("note")):
            uid = row["object_uid"]; oid = uid if h is None else f"{uid}_{h}"
            _, mesh = prepare(os.path.join(a.glb_dir, f"{uid}.glb"), float(row["target_height_cm"]) / 100,
                              row.get("up_axis") or "+y", h)
            mass = float(row["target_mass_kg"] or 0) or None
            ent = {"extents_cm": None, "faces": {}}
            for T in tilts:
                lab = label_faces(mesh, dataclasses.replace(Params(), min_tip_deg=T), mass=mass, uid=oid)
                ent["extents_cm"] = [round(100 * e, 1) for e in lab.extents]
                for f in lab.faces:
                    d = ent["faces"].setdefault(f.face, {})
                    d[str(int(T))] = {"static": bool(f.supportable), "tip_deg": f.tip_deg, "margin_mm": 1000 * f.com_margin,
                                      "review": bool(f.review), "sim": sweep[oid][str(int(T)) if str(int(T)) in sweep[oid] else str(T)][f.face]}
            out[oid] = ent
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
