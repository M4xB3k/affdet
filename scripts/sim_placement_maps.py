"""Placement actionability maps for object pairs (B put down on A) from physics simulation.

    python scripts/sim_placement_maps.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.objects import load_body  # noqa: E402
from affdet.sim.placement_map import placement_map  # noqa: E402


def main():
    picks = json.load(open(sys.argv[1]))
    glb, shaker_csv, out = sys.argv[2:5]
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "coacd")

    def role_body(role, i, name):
        r = picks[role][i]
        return load_body(os.path.join(glb, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                         float(r["objathor_mass_kg"]), name, cache_dir=cache)

    import csv
    srow = [r for r in csv.DictReader(open(shaker_csv)) if r["object_uid"].startswith("270b9b45")][0]
    shaker, shaker_mesh = load_body(os.path.join(glb, srow["object_uid"] + ".glb"), float(srow["target_height_cm"]) / 100,
                                    float(srow["target_mass_kg"]), "shaker", cache_dir=cache, convex=True)
    board, board_mesh = role_body("cutting_board", 0, "board")
    mug, mug_mesh = role_body("mug", 0, "mug")
    tray, tray_mesh = role_body("tray", 0, "tray")
    pairs = {"shaker_on_board": (board, board_mesh, shaker), "shaker_on_mug": (mug, mug_mesh, shaker),
             "shaker_on_tray": (tray, tray_mesh, shaker),}
    for name, (base, base_mesh, top) in pairs.items():
        ext = base_mesh.extents
        step = max(0.005, round(max(ext[:2]) / 36, 3))
        m = placement_map(base, top, ext[:2], step=step)
        m["base_extents"] = ext
        pickle.dump(m, open(os.path.join(out, f"{name}.pkl"), "wb"))
        print(name, "grid", m["stable"].shape, "stable %.0f%%" % (100 * m["stable"].mean()),
              "centre stable", m["centre_stable"], "best", np.round(m["best_xy"], 3), "margin mm %.0f" % m["best_margin_mm"], flush=True)


if __name__ == "__main__":
    main()
