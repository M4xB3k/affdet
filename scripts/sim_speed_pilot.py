"""Pilot: can a placement be decided with less simulation? Samples cells of real pair sweeps (the grid and faces of
`sim_pair_batch.py --cells 32`), runs each speed variant and the full run on the same cells, and compares the verdicts.

    python scripts/sim_speed_pilot.py PICKS GLB SHAKER_CSV CACHE OUT_JSON --pairs shaker:d00ef026,mug:0 ... [--n 120] [--procs 16]

A pair is "TOP,BASE". Prints per variant: cells, agreement with the full run, false-stable / false-fail counts, wall time ratio.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from affdet.sim.mujoco_support import FACE_DOWN, placement_stable, stack_test  # noqa: E402
from affdet.sim.pair_map import face_supports  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402

VARIANTS = {
    "full": dict(seconds=2.0, fast=None),
    "table": dict(seconds=2.0, fast={"on_table": True}),
    "abort": dict(seconds=2.0, fast={"abort_drift_m": 0.06, "abort_tilt_deg": 30.0}),
    "abort_tight": dict(seconds=2.0, fast={"abort_drift_m": 0.045, "abort_tilt_deg": 20.0}),
    "table+abort": dict(seconds=2.0, fast={"on_table": True, "abort_drift_m": 0.06, "abort_tilt_deg": 30.0}),
    "table+abort, 1.5 s": dict(seconds=1.5, fast={"on_table": True, "abort_drift_m": 0.06, "abort_tilt_deg": 30.0}),
    "table+abort, 1.25 s": dict(seconds=1.25, fast={"on_table": True, "abort_drift_m": 0.06, "abort_tilt_deg": 30.0}),
}
G = {}


def _init(base, top):
    G["base"], G["top"] = base, top


def _run(a):
    face, xy, v = a
    t = time.time()
    r = stack_test(G["base"], G["top"], "under", face, offset=xy, **VARIANTS[v])
    ok = bool(placement_stable(r["n_contacts"], r["tilt_deg"], r["drift_m"], r["excess_drop_m"], r["w_end"], r["v_end"], r["release_penetration_m"]))
    return ok, time.time() - t, r["aborted_at"], r["on_table"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("cache"); ap.add_argument("out")
    ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--procs", type=int, default=16); ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.cache)
    rng = np.random.default_rng(0)
    res = []
    for pr in a.pairs:
        top_s, base_s = pr.split(",")
        top, _ = reg.get(top_s); base, bm = reg.get(base_s)
        faces = [f for f in FACE_DOWN if face_supports(top, f)]
        ext = bm.extents; step = max(0.005, round(max(ext[:2]) / a.cells, 3))
        xs = np.arange(-ext[0] / 2, ext[0] / 2 + 1e-9, step); ys = np.arange(-ext[1] / 2, ext[1] / 2 + 1e-9, step)
        cells = [(f, (float(x), float(y))) for f in faces for y in ys for x in xs]
        pick = [cells[i] for i in rng.choice(len(cells), min(a.n, len(cells)), replace=False)]
        out = {"pair": pr, "faces": faces, "grid_cells_per_face": len(xs) * len(ys), "n": len(pick), "variants": {}}
        ref = None
        for v in a.variants:
            t0 = time.time()
            with Pool(a.procs, initializer=_init, initargs=(base, top)) as p:
                rr = p.map(_run, [(f, xy, v) for f, xy in pick], chunksize=2)
            wall = time.time() - t0
            ok = np.array([r[0] for r in rr]); cpu = sum(r[1] for r in rr)
            if v == "full":
                ref, ref_cpu = ok, cpu
            out["variants"][v] = {"stable_ref": int(ref.sum()), "agree": int((ok == ref).sum()), "false_stable": int((ok & ~ref).sum()),
                                  "false_fail": int((~ok & ref).sum()), "cpu_s": round(cpu, 1), "cpu_vs_full": round(cpu / ref_cpu, 3),
                                  "wall_s": round(wall, 1), "aborted": int(sum(r[2] is not None for r in rr)), "on_table": int(sum(r[3] for r in rr))}
            print(pr, v, out["variants"][v], flush=True)
        res.append(out)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
