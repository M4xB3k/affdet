"""Export every simulated map as per-vertex heatmap labels (docs/sim_labels.md) plus an index.

    python scripts/export_labels.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR --pairs DIR [DIR ...] [--grasp DIR ...] [--slide DIR ...]

Pair directories hold `<top>_on_<base>.pkl` from sim_pair_batch.py (names as in affdet/sim/registry.py: `shaker<uid8>`, `<role><index>`),
grasp directories `<role>_<uid8>.pkl` from sim_grasp_maps.py, slide directories `slide_<name>.pkl` from sim_slide_maps.py.
Writes OUT_DIR/pair/<pair>.A.npz and .B.npz, OUT_DIR/grasp/<object>.npz, OUT_DIR/slide/<object>.npz and OUT_DIR/index.json.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.geometry.prepare import prepare  # noqa: E402
from affdet.sim import label_export as le  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402


def spec_of(name, reg):
    """`shaker86450daa` -> `shaker:86450daa`; `frying_pan0` -> `frying_pan:0`."""
    m = re.match(r"^(shaker)([0-9a-f]{8})$", name)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    m = re.match(r"^(.*?)(\d+)$", name)
    return f"{m.group(1)}:{m.group(2)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out")
    ap.add_argument("--pairs", nargs="*", default=[]); ap.add_argument("--grasp", nargs="*", default=[]); ap.add_argument("--slide", nargs="*", default=[])
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, "/tmp")
    meshes = {}

    def mesh(spec):
        if spec not in meshes:
            glb, h = reg.glb(spec)
            meshes[spec] = prepare(glb, h, "+y")[1]
        return meshes[spec]

    for d in ("pair", "grasp", "slide"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    index = {"schema_version": le.SCHEMA_VERSION, "cap_mm": le.CAP_MM, "pair": [], "grasp": [], "slide": []}
    for d in a.pairs:
        for f in sorted(glob.glob(os.path.join(d, "*_on_*.pkl"))):
            name = os.path.basename(f)[:-4]
            top_n, base_n = name.split("_on_")
            ts, bs = spec_of(top_n, reg), spec_of(base_n, reg)
            m = pickle.load(open(f, "rb"))
            lab = le.pair_labels(m, mesh(ts), mesh(bs))
            meta = {"pair": name, "top": ts, "base": bs, "step": m["step"], "faces": m["faces"], "skipped_faces": m["skipped"]}
            le.save(os.path.join(a.out, "pair", f"{name}.A.npz"), lab["A"], {**meta, "side": "A (put down)"})
            le.save(os.path.join(a.out, "pair", f"{name}.B.npz"), lab["B"], {**meta, "side": "B (supporting)"})
            index["pair"].append({"pair": name, "top": ts, "base": bs, "supported_fraction_of_B": round(float(m["b_any"].mean()), 3),
                                  "best_margin_mm": round(float(m["b_margin_mm"].max()), 1), "faces_that_work": [k for k, v in m["a_face_margin_mm"].items() if v > 0]})
    for d in a.grasp:
        for f in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            role, uid = os.path.basename(f)[:-4].rsplit("_", 1)
            idx = [i for i, r in enumerate(reg.picks[role]) if r["object_uid"].startswith(uid)][0]
            spec = f"{role}:{idx}"
            g = pickle.load(open(f, "rb"))
            lab = le.grasp_labels(g, mesh(spec))
            le.save(os.path.join(a.out, "grasp", f"{role}{idx}.npz"), lab, {"object": spec, "step": g["step"], "yaws": g["yaws"]})
            index["grasp"].append({"object": spec, "held_cells": int(g["held"].any(2).sum()), "stable_cells": int((g["held"] & (g["swing_deg"] < 30)).any(2).sum()), "cells": int(g["held"].any(2).size)})
    for d in a.slide:
        for f in sorted(glob.glob(os.path.join(d, "slide_*.pkl"))):
            name = os.path.basename(f)[len("slide_"):-4]
            spec = spec_of(name, reg)
            s = pickle.load(open(f, "rb"))
            lab = le.slide_labels(s, mesh(spec))
            le.save(os.path.join(a.out, "slide", f"{name}.npz"), lab, {"object": spec, "azimuths": s["azimuths"], "fracs": s["fracs"]})
            ok = s["success"][s["on_object"]]
            index["slide"].append({"object": spec, "slides_fraction": round(float(ok.mean()) if ok.size else 0.0, 3)})
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print({k: len(v) for k, v in index.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
