import json, sys, random, re, statistics as st
d = json.load(open(sys.argv[1]))
WRONG = re.compile(r"satellite|cocktail|protein|shaker bottle|pepper mill|grinder|\bset\b|stack|scene|knife block|plank|slab|with food|document|plant|mounting|bottle|sculpture|decorative object", re.I)
STEP = (5, 5, 0.5)  # annotation granularity: cm, cm, kg
out = {}
for role, v in d.items():
    rng = v["range"]
    allr = v["ok"] + v["bad"]
    kept, items = [], []
    for r in allr:
        kind = None
        if WRONG.search(r["cat"] or ""):
            kind = "wrong_object"
        elif r["why"]:
            vals = (r["h"], r["d"], r["m"])
            off = []
            for (lo, hi), x, s in zip(rng, vals, STEP):
                if x is None: off.append(99); continue
                if x < lo: off.append((lo - x) / max(s, 0.25 * lo))
                elif x > hi: off.append((x - hi) / max(s, 0.25 * hi))
            kind = "borderline" if max(off) <= 1 else "far_off"
        r["kind"] = kind
        if kind != "wrong_object": kept.append(r)
        if kind: items.append(r)
    ok = [r for r in kept if not r["kind"]]
    random.seed(0)
    typical = sorted(random.sample([r for r in ok if (r["q"] or 0) >= 8] or ok, 6), key=lambda r: r["d"] or 0)
    med = lambda k: st.median([r[k] for r in kept if r[k] is not None])
    out[role] = dict(range=rng, n_usable=len(allr), n_kept=len(kept), n_ok=len(ok),
                     median=[med("h"), med("d"), med("m")],
                     values=[[r["h"], r["d"], r["m"]] for r in kept],
                     typical=typical, outliers=items)
    from collections import Counter
    print(f"{role:14s}", len(allr), Counter(r["kind"] for r in items))
json.dump(out, open(sys.argv[2], "w"))
