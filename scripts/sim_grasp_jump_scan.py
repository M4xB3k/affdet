"""How often does the object jump during the closing phase of a grasp (before the lift starts)?
    python scripts/sim_grasp_jump_scan.py MAP_DIR PICKS GLB OUT.json --specs ... [--n 30]
For a random sample of on-object cells per spec, replays grasp_test and logs the object's centre of mass: the displacement
and the peak speed while the pads are still closing (pad height unchanged), and the largest one-step jump."""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from affdet.sim.grasp_map import grasp_test
from affdet.sim.registry import Registry


def scan(body, m, n, rng):
    on = np.argwhere(m["on_object"] & ~m["blocked"])
    if len(on) == 0:
        return []
    sel = on[rng.choice(len(on), min(n, len(on)), replace=False)]
    out = []
    for iy, ix, iw in sel:
        log = []

        def rec(mm, d, log=log, body=body):
            oid = mm.body(body.name).id
            il = mm.body("tfl").mocapid[0]
            log.append((d.time, d.xipos[oid].copy(), float(d.mocap_pos[il][2])))
        r = grasp_test(body, float(m["xs"][ix]), float(m["ys"][iy]), float(m["yaws"][iw]), record=rec, params=m["params"])
        t = np.array([l[0] for l in log]); c = np.array([l[1] for l in log]); z = np.array([l[2] for l in log])
        pre = z <= z[0] + 1e-4   # pads not yet lifting
        dt = np.diff(t, prepend=t[0] - 1e-3)
        v = np.linalg.norm(np.diff(c, axis=0, prepend=c[:1]), axis=1) / np.maximum(dt, 1e-9)
        out.append({"cell": [int(iy), int(ix), int(iw)], "held": bool(r["held"]), "success": bool(r["success"]),
                    "pre_disp_cm": float(100 * np.linalg.norm(c[pre] - c[0], axis=1).max()),
                    "pre_up_cm": float(100 * (c[pre][:, 2] - c[0][2]).max()),
                    "pre_peak_speed": float(v[pre].max()), "pre_time_s": float(t[pre][-1] - t[0])})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("maps"); ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("out")
    ap.add_argument("--specs", nargs="+", required=True); ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--coacd-dir"); ap.add_argument("--shaker-csv", default=os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_set.csv"))
    a = ap.parse_args()
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.coacd_dir)
    rng = np.random.default_rng(0); res = {}
    for spec in a.specs:
        m = pickle.load(open(os.path.join(a.maps, f"grasp_{reg.spec_name(spec)}.pkl"), "rb"))
        res[spec] = scan(reg.get(spec)[0], m, a.n, rng)
        r = res[spec]; d = np.array([x["pre_disp_cm"] for x in r])
        print(spec, len(r), "median pre-lift displacement %.2f cm, >1cm: %.0f%%, >3cm: %.0f%%" % (np.median(d), 100 * (d > 1).mean(), 100 * (d > 3).mean()), flush=True)
        json.dump(res, open(a.out, "w"))
