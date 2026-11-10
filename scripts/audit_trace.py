import json, os, pickle, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from affdet.sim.registry import Registry
from affdet.sim.grasp_map import grasp_test
A = {}
def init(a): A["reg"] = Registry(*a)
def work(a):
    maps, spec, name, rows = a
    m = pickle.load(open(os.path.join(maps, f"grasp_{name}.pkl"), "rb")); body, _ = A["reg"].get(spec); out = []
    for r in rows:
        z = []
        def rec(mm, d): z.append(float(d.xipos[mm.body(body.name).id][2]))
        iy, ix, iw = r["cell"]
        res = grasp_test(body, float(m["xs"][ix]), float(m["ys"][iy]), float(m["yaws"][iw]), record=rec, params=m["params"])
        z = np.array(z); out.append({"stratum": r["stratum"], "max_lift_cm": float(100 * (z.max() - z[0])), "final_lift_cm": float(100 * (z[-1] - z[0])), "held": bool(res["held"]), "slip_cm": float(100 * res["slip_m"])})
    return name, out
if __name__ == "__main__":
    maps, picks, glb, coacd, csv_, phys, outp = sys.argv[1:8]
    a = (picks, glb, csv_, coacd); reg = Registry(*a); d = json.load(open(phys))
    jobs = [(maps, v["spec"], k, [r for r in v["rows"] if r["stratum"] == "not_held"]) for k, v in d.items()]
    with Pool(16, initializer=init, initargs=(a,)) as p: res = dict(p.imap_unordered(work, [j for j in jobs if j[3]]))
    json.dump(res, open(outp, "w")); print("done")
