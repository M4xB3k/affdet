"""Independent check of the pair maps on flat bases: a purely geometric prediction of "stays", compared cell by cell with the simulator.

    python scripts/sim_static_vs_sim_placement.py PICKS_JSON GLB_DIR SHAKER_CSV PAIR_DIR(S...) [--base-role cutting_board] [--margin-mm 5] [--out OUT.json]

Static rule for a top put down on a flat base (the base's upward surface is its horizontal outline, a thin slab lying on the table): the
top rests on the part of its face-down patch that lies over the base; it stays if its centre of mass projects at least `margin` mm inside that
overlap polygon. Nothing here uses physics. Run on boards (and other flat slabs) only: a cup or tray has a cavity the rule does not model.
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
import trimesh  # noqa: E402
from shapely import affinity  # noqa: E402
from shapely.geometry import MultiPoint, Point  # noqa: E402

from affdet.geometry.prepare import prepare  # noqa: E402
import affdet.sim.mujoco_support as ms  # noqa: E402
from affdet.sim.objects import centre_of_mass  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402


def spec_of(name):
    m = re.match(r"^(shaker)([0-9a-f]{8})$", name)
    if m:
        return f"shaker:{m.group(2)}"
    m = re.match(r"^(.*?)(\d+)$", name)
    return f"{m.group(1)}:{m.group(2)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base-role", default="cutting_board"); ap.add_argument("--margin-mm", type=float, default=5.0); ap.add_argument("--out")
    ap.add_argument("--coacd", default="/tmp/coacd_static"); ap.add_argument("--sim-tilt", type=float, default=10.0, help="re-apply this tilt limit to the stored raw metrics")
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.coacd)
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "**", f"*_on_{a.base_role}*.pkl"), recursive=True))
    rows = []
    agree = both_yes = both_no = sim_only = static_only = 0
    by_top = {}
    flat = [0, 0, 0, 0]  # faces with a real footprint (>= 4 cm^2): both stable, both fail, sim only, static only
    thin = [0, 0, 0, 0]
    for f in files:
        name = os.path.basename(f)[:-4]
        tn, bn = name.split("_on_")
        ts, bs = spec_of(tn), spec_of(bn)
        top, _ = reg.get(ts)
        bmesh = prepare(*reg.glb(bs))[1]
        v = bmesh.vertices
        outline = MultiPoint(v[v[:, 2] > v[:, 2].max() - 0.0015][:, :2].tolist()).convex_hull  # the base's top face
        com_local = centre_of_mass(top.parts, top.part_mass)
        allv = np.concatenate([p.vertices for p in top.parts])
        m = pickle.load(open(f, "rb"))
        for face, pf in m["per_face"].items():
            R = trimesh.transformations.rotation_matrix(ms.FACE_DOWN[face][1], ms.FACE_DOWN[face][0])[:3, :3]
            rv = allv @ R.T
            patch = rv[rv[:, 2] < rv[:, 2].min() + 0.001][:, :2]
            patch_poly = MultiPoint(patch.tolist()).convex_hull
            com_xy = (R @ com_local)[:2]
            bucket = flat if patch_poly.area >= 4e-4 else thin
            n_ok = 0
            from affdet.sim.placement_map import finalize
            sim_grid = finalize({k: (v.copy() if hasattr(v, 'copy') else v) for k, v in pf.items()}, max_tilt=a.sim_tilt)['stable']
            for j, y in enumerate(pf["ys"]):
                for i, x in enumerate(pf["xs"]):
                    over = affinity.translate(patch_poly, x, y).intersection(outline)
                    static = (not over.is_empty) and over.area > 0 and over.buffer(-a.margin_mm / 1000).contains(Point(com_xy[0] + x, com_xy[1] + y))
                    sim = bool(sim_grid[j, i])
                    bucket[0 if static and sim else 1 if (not static and not sim) else 2 if sim else 3] += 1
                    d = by_top.setdefault(ts + " " + face, [0, 0, 0, 0]); d[(0 if static and sim else 1 if (not static and not sim) else 2 if sim else 3)] += 1
                    agree += static == sim
                    both_yes += static and sim; both_no += (not static) and (not sim)
                    sim_only += sim and not static; static_only += static and not sim
                    n_ok += 1
            rows.append({"pair": name, "face": face, "cells": n_ok})
    total = both_yes + both_no + sim_only + static_only
    res = {"cells": int(total), "agree": int(agree), "agreement": round(agree / total, 4), "both_stable": int(both_yes), "both_fail": int(both_no),
           "sim_only": int(sim_only), "static_only": int(static_only), "pairs": len(files), "face_maps": len(rows), "margin_mm": a.margin_mm}
    res["by_top_face"] = {k: {"both_stable": v[0], "both_fail": v[1], "sim_only": v[2], "static_only": v[3]} for k, v in by_top.items()}
    worst = sorted(by_top.items(), key=lambda kv: -kv[1][2])[:12]
    for nm, b in (("faces with a real footprint (>= 4 cm2)", flat), ("faces whose footprint is a line or ring", thin)):
        t = sum(b)
        res[nm] = {"cells": t, "agreement": round((b[0] + b[1]) / max(t, 1), 4), "both_stable": b[0], "both_fail": b[1], "sim_only": b[2], "static_only": b[3]}
    print(json.dumps({k: v for k, v in res.items() if k != "by_top_face"}, indent=1))
    print("largest sim-only (top face: both-stable / both-fail / sim-only / static-only):")
    for k, v in worst:
        print("  %-26s %6d %6d %6d %6d" % (k, *v))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
