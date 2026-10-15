"""Compare a pair-map run made with the speed options (adaptive + grow + abort + reuse) with a full run of the same pairs.

    python scripts/sim_fast_vs_full.py FAST_DIR REF_DIR OUT_JSON [--stored STORED_DIR]

Per pair and in total: simulations and wall time, cell agreement per face, stable cells lost / invented, error of the best margin
(per face and on B), mean |difference| of the B-side margin map. With --stored, the reference run is also compared with the stored
labels (it should be identical: the simulation is deterministic).
"""
import argparse, glob, json, os, pickle
import numpy as np


def load(d):
    recs = {}
    for l in open(os.path.join(d, "pairs.jsonl")):
        r = json.loads(l)
        if "error" not in r:
            recs[r["pair"]] = r
    return recs


def cmp_maps(a, b):
    out = dict(cells=0, wrong=0, lost=0, invented=0, stable=0, best_err=[])
    for f in set(a["per_face"]) & set(b["per_face"]):
        sa, sb = np.asarray(a["per_face"][f]["stable"], bool), np.asarray(b["per_face"][f]["stable"], bool)
        out["cells"] += sa.size; out["wrong"] += int((sa != sb).sum()); out["stable"] += int(sb.sum())
        out["lost"] += int((sb & ~sa).sum()); out["invented"] += int((sa & ~sb).sum())
        out["best_err"].append(float(b["per_face"][f]["best_margin_mm"] - a["per_face"][f]["best_margin_mm"]))
    out["b_margin_mean_abs_diff_mm"] = float(np.abs(np.asarray(a["b_margin_mm"]) - np.asarray(b["b_margin_mm"])).mean())
    out["b_any_agree"] = float((np.asarray(a["b_any"]) == np.asarray(b["b_any"])).mean())
    out["b_best_margin_err_mm"] = float(np.max(b["b_margin_mm"]) - np.max(a["b_margin_mm"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fast"); ap.add_argument("ref"); ap.add_argument("out"); ap.add_argument("--stored")
    a = ap.parse_args()
    fr, rr = load(a.fast), load(a.ref)
    rows, tot = [], dict(cells=0, wrong=0, lost=0, invented=0, stable=0, sims_fast=0, sims_ref=0, s_fast=0.0, s_ref=0.0)
    for name in sorted(set(fr) & set(rr)):
        mf = pickle.load(open(os.path.join(a.fast, name + ".pkl"), "rb")); mr = pickle.load(open(os.path.join(a.ref, name + ".pkl"), "rb"))
        c = cmp_maps(mf, mr)
        row = dict(pair=name, sims_fast=fr[name]["simulations"], sims_ref=rr[name]["simulations"], s_fast=fr[name]["seconds"], s_ref=rr[name]["seconds"], **{k: v for k, v in c.items() if k != "best_err"},
                   best_margin_err_mm_max_abs=max(map(abs, c["best_err"]), default=0.0))
        if a.stored:
            sp = glob.glob(os.path.join(a.stored, "**", name + ".pkl"), recursive=True)
            if sp:
                ms = pickle.load(open(sp[0], "rb")); row["ref_vs_stored_cells_differing"] = cmp_maps(mr, ms)["wrong"]
        rows.append(row)
        for k in ("cells", "wrong", "lost", "invented", "stable"):
            tot[k] += c[k]
        tot["sims_fast"] += row["sims_fast"]; tot["sims_ref"] += row["sims_ref"]; tot["s_fast"] += row["s_fast"]; tot["s_ref"] += row["s_ref"]
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    tot["sims_ratio"] = tot["sims_fast"] / tot["sims_ref"]; tot["time_ratio"] = tot["s_fast"] / tot["s_ref"]; tot["cell_agreement"] = 1 - tot["wrong"] / tot["cells"]
    print(json.dumps(tot, indent=1))
    json.dump({"total": tot, "pairs": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
