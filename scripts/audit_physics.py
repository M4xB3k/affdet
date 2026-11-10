"""Physics audit of the unblocked cells of plates and boards: replay a stratified sample and classify why the grasp failed.
    python audit_physics.py MAPS PICKS GLB COACD CSV OUT.json SPEC...     (needs the maps' stored outcomes to replay exactly)"""
import json, os, pickle, sys
from multiprocessing import Pool
import numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import affdet.sim.mujoco_support as ms
from affdet.sim.registry import Registry
from affdet.sim.grasp_map import grasp_test
A = {}
def init(a): A["reg"] = Registry(*a)
def work(a):
    maps, spec, name, n = a
    m = pickle.load(open(os.path.join(maps, f"grasp_{name}.pkl"), "rb")); body, mesh = A["reg"].get(spec)
    unb = m["on_object"] & ~m["blocked"]
    rng = np.random.default_rng(1)
    out = {"spec": spec, "unblocked": int(unb.sum()), "held": int((unb & m["held"]).sum()), "success": int(m["success"].sum()),
           "height_cm": float(mesh.extents[2] * 100), "width_cm": float(mesh.extents[:2].max() * 100), "mass": float(body.mass), "rows": []}
    # dish depth: highest point minus the height at the centre
    cz = mesh.bounds[1][2]
    h = mesh.ray.intersects_location(np.array([[0, 0, 2.0]]), np.array([[0, 0, -1.0]]), multiple_hits=True)[0]
    out["depth_cm"] = float((cz - h[:, 2].max()) * 100) if len(h) else 0.0
    for lab, mask in (("not_held", unb & ~m["held"]), ("held_notlevel", m["held"] & ~m["success"]), ("success", m["success"])):
        idx = np.argwhere(mask)
        for i in (rng.choice(len(idx), min(n, len(idx)), replace=False) if len(idx) else []):
            iy, ix, iw = idx[i]
            r = grasp_test(body, float(m["xs"][ix]), float(m["ys"][iy]), float(m["yaws"][iw]), params=m["params"])
            reason = ("held level" if r["success"] else "held, swings %.0f deg" % r["swing_deg"]) if r["held"] else (
                "not lifted (<5 cm)" if r["lifted_m"] < 0.05 else ("slipped in the grip (%.1f cm)" % (100 * r["slip_m"]) if r["slip_m"] >= 0.05 else "dropped on the table"))
            out["rows"].append({"stratum": lab, "cell": [int(iy), int(ix), int(iw)], "stored_held": bool(m["held"][iy, ix, iw]), "replay_held": bool(r["held"]),
                                "swing": float(r["swing_deg"]), "lifted_m": float(r["lifted_m"]), "slip_m": float(r["slip_m"]), "hold_force_n": float(r.get("hold_force_n", 0)), "reason": reason})
    return name, out
if __name__ == "__main__":
    maps, picks, glb, coacd, csv_, outp, *specs = sys.argv[1:]
    a = (picks, glb, csv_, coacd); reg = Registry(*a)
    with Pool(16, initializer=init, initargs=(a,)) as p:
        res = dict(p.imap_unordered(work, [(maps, s, reg.spec_name(s), 5) for s in specs]))
    json.dump(res, open(outp, "w"), indent=1); print("done", len(res))
