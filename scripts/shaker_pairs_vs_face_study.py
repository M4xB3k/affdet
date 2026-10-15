"""All curated shakers as tops: which faces get supported by some base, against the shaker face study (static and simulated verdicts).

    python scripts/shaker_pairs_vs_face_study.py JSONL [JSONL ...] [--out OUT.json]
"""
import argparse
import collections
import csv
import json
import os

FACES = ("under", "on", "front", "back", "left", "right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+"); ap.add_argument("--out")
    ap.add_argument("--study", default=os.path.join(os.path.dirname(__file__), "..", "docs", "objaverse", "salt_shaker_sim_vs_static.csv"))
    a = ap.parse_args()
    study = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.study)):
        study[r["object_id"][:8]][r["face"]] = (r["sim_verdict"] == "1", r["static_supportable"] == "1")
    pairs = collections.defaultdict(list)
    for f in a.jsonl:
        for l in open(f):
            r = json.loads(l)
            if r["top"].startswith("shaker:"):
                pairs[r["top"].split(":")[1]].append(r)
    tot = agree_sim = agree_static = n = 0
    rows, mism = [], []
    for sid, rs in sorted(pairs.items()):
        ok = [r for r in rs if "error" not in r]
        works = {f: any(r["a_face_margin_mm"][f] > 0 for r in ok) for f in FACES}
        if sid not in study:
            continue
        for f in FACES:
            if f not in study[sid]:
                continue
            sim, static = study[sid][f]
            tot += 1; agree_sim += works[f] == sim; agree_static += works[f] == static
            if works[f] != sim:
                mism.append((sid, f, "pair maps say " + ("works" if works[f] else "never works") + ", study sim says " + ("holds" if sim else "does not hold")))
        rows.append({"shaker": sid, "bases": len(rs), "no_face_holds": len(rs) - len(ok), "faces_that_work": [f for f in FACES if works[f]]})
    res = {"shakers": len(rows), "faces": tot, "agree_with_study_sim": agree_sim, "agree_with_static_labeller": agree_static,
           "pair_maps": sum(len(v) for v in pairs.values()), "mismatches_vs_sim": mism}
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches_vs_sim"}, indent=1))
    for m in mism:
        print("  ", *m)
    if a.out:
        json.dump({**res, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
