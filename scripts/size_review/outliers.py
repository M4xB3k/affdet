import csv, sys, re, json
rows = list(csv.DictReader(open(sys.argv[1])))
pat = {"cutting_board": r"cutting|chopping|board", "frying_pan": r"pan|skillet|wok",
       "mug": r"mug|cup", "plate_bowl": r"plate|bowl|dish",
       "salt_shaker": r"salt|pepper|shaker", "tray": r"tray|platter"}
# plausible real-world ranges: (height cm, max_dim cm, mass kg); height = resting upright/flat
R = {"salt_shaker": ((5, 15), (5, 15), (0.03, 0.4)),
     "mug": ((6, 15), (7, 17), (0.1, 0.7)),
     "cutting_board": ((0.8, 6), (18, 60), (0.2, 3.5)),
     "frying_pan": ((3, 15), (25, 60), (0.4, 3.5)),
     "tray": ((1, 10), (20, 65), (0.1, 3.0)),
     "plate_bowl": ((1, 15), (10, 35), (0.1, 2.0))}
f = lambda x: float(x) if x not in ("", None) else None
out = {}
for role, p in pat.items():
    good = [r for r in rows if r["role"] == role and (f(r["quality"]) or 0) >= 7
            and re.search(p, (r["objathor_category"] or "") + " " + (r["synset"] or ""), re.I)]
    (h0, h1), (d0, d1), (m0, m1) = R[role]
    ok, bad = [], []
    for r in good:
        h, d, m = f(r["height_cm"]), f(r["max_dim_cm"]), f(r["mass_kg"])
        width = None
        if h is not None and d is not None and d < h:
            # GPT-4o often reports the width here for tall objects; longest side is then the height
            width, d = d, h
        why = []
        if h is None or not h0 <= h <= h1: why.append(f"height {h:g} cm" if h is not None else "no height")
        if d is None or not d0 <= d <= d1: why.append(f"max dim {d:g} cm" if d is not None else "no max dim")
        if m is None or not m0 <= m <= m1: why.append(f"mass {m:g} kg" if m is not None else "no mass")
        rec = dict(uid=r["object_uid"], h=h, d=d, m=m, mat=r["materials"], q=f(r["quality"]), width=width,
                   cat=r["objathor_category"], lic=r["license"], desc=r["description"],
                   bb=[f(r["bbox_x_m"]), f(r["bbox_y_m"]), f(r["bbox_z_m"])], why=why)
        (bad if why else ok).append(rec)
    out[role] = dict(range=R[role], ok=ok, bad=bad)
    print(f"{role:14s} usable {len(good):5d}  in range {len(ok):5d}  outliers {len(bad):4d}")
json.dump(out, open(sys.argv[2], "w"))
