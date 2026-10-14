"""Compare the simulated grasp map with the analytic torque model on the same grid.

    python scripts/sim_grasp_compare.py SIM_PKL GLB OBJECT_HEIGHT_M MASS_KG CACHE_DIR [--swing 30]

Prints, per cell (best yaw): sim held (lifted, no slip), sim stable (held and swing < limit), analytic fits and
analytic level, and the 2x2 tables that matter: stable vs level, held vs fits.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.grasp_analytic import grasp_analytic  # noqa: E402
from affdet.sim.objects import load_body  # noqa: E402


def compare(sim, body, swing_deg=30.0):
    a = grasp_analytic(body.parts, body.mass, sim["xs"], sim["ys"], sim["yaws"], part_mass=body.part_mass)
    held = sim["held"].any(2)
    stable = (sim["held"] & (sim["swing_deg"] < swing_deg)).any(2)
    fits, level = a["fits"].any(2), a["level"].any(2)
    t = lambda p, q: {"both": int((p & q).sum()), "sim_only": int((p & ~q).sum()), "analytic_only": int((~p & q).sum()),
                      "neither": int((~p & ~q).sum())}
    return {"cells": int(held.size), "held_vs_fits": t(held, fits), "stable_vs_level": t(stable, level),
            "held_cells": int(held.sum()), "stable_cells": int(stable.sum()), "fit_cells": int(fits.sum()),
            "level_cells": int(level.sum()), "com": [round(float(c), 3) for c in a["com"]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sim"); ap.add_argument("glb"); ap.add_argument("height", type=float); ap.add_argument("mass", type=float)
    ap.add_argument("cache"); ap.add_argument("--swing", type=float, default=30.0)
    ap.add_argument("--mass-model", default="auto")
    g = ap.parse_args()
    body, _ = load_body(g.glb, g.height, g.mass, "obj", cache_dir=g.cache, mass_model=g.mass_model)
    print(json.dumps(compare(pickle.load(open(g.sim, "rb")), body, g.swing)))
