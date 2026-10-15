"""Offline pilot: how symmetric are the stored pair maps, and how much could symmetry save? For every per-face map the verdict at (x, y)
is compared with the verdict at the mirrored / rotated point (x -> -x, y -> -y, both), and the saving of "simulate half, mirror the
rest, check on a small random sample of the other half" is emulated.

    python scripts/sim_symmetry_pilot.py MAP_DIR OUT_JSON [--check 0.05]
"""
import argparse, collections, glob, json, os, pickle
import numpy as np

T = {"mirror_x": (-1, 1), "mirror_y": (1, -1), "rot180": (-1, -1)}


def partner(xs, ys, sx, sy):
    """Index of the grid point nearest the image of every cell under (x, y) -> (sx x, sy y), and whether it lies on the grid (within half a step)."""
    step = abs(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    ii = np.abs(xs[None, :] * sx - xs[:, None]).argmin(1)  # for each x: nearest index of sx*x
    jj = np.abs(ys[None, :] * sy - ys[:, None]).argmin(1)
    okx = np.abs(xs * sx - xs[ii]) <= 0.5 * step + 1e-9
    oky = np.abs(ys * sy - ys[jj]) <= 0.5 * step + 1e-9
    return ii, jj, okx, oky


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("maps"); ap.add_argument("out"); ap.add_argument("--check", type=float, default=0.05)
    ap.add_argument("--pass_agree", type=float, default=0.98)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for fn in sorted(glob.glob(os.path.join(a.maps, "**", "*.pkl"), recursive=True)):
        m = pickle.load(open(fn, "rb"))
        if not isinstance(m, dict) or "per_face" not in m:
            continue
        name = os.path.basename(fn)[:-4]
        top, base = name.split("_on_")
        trole = top.rstrip("0123456789abcdef").rstrip("_") if not top.startswith("shaker") else "shaker"
        brole = base.rstrip("0123456789")
        for f, v in m["per_face"].items():
            S = np.asarray(v["stable"], bool)
            if S.sum() < 3:
                continue
            xs, ys = np.asarray(v["xs"], float), np.asarray(v["ys"], float)
            for tn, (sx, sy) in T.items():
                ii, jj, okx, oky = partner(xs, ys, sx, sy)
                P = S[np.ix_(jj, ii)]                       # verdict at the partner point
                valid = oky[:, None] & okx[None, :]
                if valid.sum() < 16:
                    continue
                agree = float((P == S)[valid].mean())
                # half-domain emulation: keep the half x<=0 / y<=0 (or the half-plane for rot180), mirror it to the rest
                X, Y = np.meshgrid(xs, ys)
                half = (X * sx + Y * sy * 0 <= 1e-9) if tn == "mirror_x" else ((Y <= 1e-9) if tn == "mirror_y" else (X + 1e-9 * Y <= 1e-9))
                other = ~half & valid
                est_err = float((P != S)[other].mean()) if other.any() else 0.0
                n_other = int(other.sum())
                k = max(8, int(a.check * n_other))
                idx = np.flatnonzero(other.ravel())
                chk = rng.choice(idx, min(k, len(idx)), replace=False)
                passed = bool((P.ravel()[chk] == S.ravel()[chk]).mean() >= a.pass_agree)
                sim_frac = (half.sum() + (len(chk) if True else 0)) / S.size
                rows.append(dict(pair=name, top=trole, base=brole, face=f, tf=tn, agree=agree, err_if_mirrored=est_err, passed=passed,
                                 sim_frac_if_passed=float(sim_frac), n=int(S.size), stable_frac=float(S.mean())))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["tf"])].append(r)
    summary = {}
    for tn, rs in by.items():
        ps = [r for r in rs if r["passed"]]
        summary[tn] = {"maps": len(rs), "passed_check": len(ps), "mean_agree_all": round(float(np.mean([r["agree"] for r in rs])), 4),
                       "mean_err_passed": round(float(np.mean([r["err_if_mirrored"] for r in ps])), 4) if ps else None,
                       "mean_sim_frac_passed": round(float(np.mean([r["sim_frac_if_passed"] for r in ps])), 3) if ps else None}
    byb = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        byb[r["base"]][r["tf"]].append(r["agree"])
    per_base = {b: {tn: round(float(np.mean(v)), 3) for tn, v in d.items()} for b, d in byb.items()}
    json.dump({"summary": summary, "per_base_mean_agreement": per_base, "rows": rows}, open(a.out, "w"))
    print(json.dumps(summary, indent=1)); print(json.dumps(per_base, indent=1))


if __name__ == "__main__":
    main()
