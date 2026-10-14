"""Re-apply the release-collision rule (B may not overlap A by more than 3 mm when let go) to stored placement maps.

    python scripts/sim_recheck_release.py PICKS_JSON GLB_DIR SHAKER_CSV PAIR_DIR [--placement-dir DIR]

Rewrites every pair map in PAIR_DIR (and pairs.jsonl) and the part-1 shaker-on-board/mug/tray maps in --placement-dir.
Only cells that were stable are re-checked, with a zero-step simulation, so this takes minutes, not hours.
"""
import argparse
import csv
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.sim.objects import load_body  # noqa: E402
from affdet.sim.pair_map import aggregate, recheck_release  # noqa: E402
from affdet.sim.placement_map import finalize  # noqa: E402

ROLES = ("frying_pan", "mug", "cutting_board", "tray")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("pair_dir")
    ap.add_argument("--placement-dir")
    a = ap.parse_args()
    picks = json.load(open(a.picks))
    srows = {r["object_uid"][:8]: r for r in csv.DictReader(open(a.shaker_csv))}
    cache = os.path.join(a.pair_dir, "coacd")
    bodies = {}

    def get(kind, ident):
        key = (kind, ident)
        if key not in bodies:
            if kind == "shaker":
                r = srows[ident]
                bodies[key] = load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["target_height_cm"]) / 100,
                                        float(r["target_mass_kg"]), f"shaker_{ident}", cache_dir=cache, convex=True)[0]
            else:
                r = picks[kind][int(ident)]
                bodies[key] = load_body(os.path.join(a.glb, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                                        float(r["objathor_mass_kg"]), f"{kind}{ident}", cache_dir=cache)[0]
        return bodies[key]

    def split(name):
        for k in ("shaker",) + ROLES:
            if name.startswith(k):
                rest = name[len(k):]
                ident = rest.split("_on_")[0] if k == "shaker" else rest
                return k, ident
        raise ValueError(name)

    log = os.path.join(a.pair_dir, "pairs.jsonl")
    recs = []
    changed = 0
    for l in open(log):
        rec = json.loads(l)
        if "error" in rec:
            recs.append(rec); continue
        name = rec["pair"]
        top_s, base_s = name.split("_on_")
        tk, ti = split(top_s) if top_s.startswith("shaker") else (top_s.rstrip("0123456789"), top_s[len(top_s.rstrip("0123456789")):])
        bk = base_s.rstrip("0123456789"); bi = base_s[len(bk):]
        if top_s.startswith("shaker"):
            tk, ti = "shaker", top_s[len("shaker"):]
        top, base = get(tk, ti), get(bk, bi)
        pk = os.path.join(a.pair_dir, name + ".pkl")
        m = pickle.load(open(pk, "rb"))
        before = int(sum(v["stable"].sum() for v in m["per_face"].values()))
        per_face = {f: recheck_release(base, top, v, f) for f, v in m["per_face"].items()}
        after = int(sum(v["stable"].sum() for v in per_face.values()))
        changed += before != after
        ext = m.get("base_extents")
        m2 = aggregate(per_face, m["faces"], m["skipped"], m["step"])
        m2["base_extents"] = ext
        pickle.dump(m2, open(pk, "wb"))
        rec.update(b_supported_frac=round(float(m2["b_any"].mean()), 3),
                   a_face_margin_mm={f: round(v, 1) for f, v in m2["a_face_margin_mm"].items()},
                   a_face_stable_frac={f: round(v, 3) for f, v in m2["a_face_stable_frac"].items()},
                   best_b_margin_mm=round(float(m2["b_margin_mm"].max()), 1), stable_cells_before=before, stable_cells_after=after)
        recs.append(rec)
        print(name, before, "->", after, flush=True)
    with open(log, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("pairs changed:", changed, "of", len(recs))
    if a.placement_dir:
        role = {"shaker_on_board": ("cutting_board", "0"), "shaker_on_mug": ("mug", "0"), "shaker_on_tray": ("tray", "0")}
        shaker = [r for r in srows.values() if r["object_uid"].startswith("270b9b45")][0]
        top = get("shaker", shaker["object_uid"][:8])
        for name, (k, i) in role.items():
            f = os.path.join(a.placement_dir, name + ".pkl")
            if not os.path.exists(f):
                continue
            m = pickle.load(open(f, "rb"))
            before = int(m["stable"].sum())
            m2 = recheck_release(get(k, i), top, m, "under")
            pickle.dump(m2, open(f, "wb"))
            print(name, "stable cells", before, "->", int(m2["stable"].sum()), "best margin mm %.0f" % m2["best_margin_mm"],
                  "centre stable", m2["centre_stable"])


if __name__ == "__main__":
    main()
