import csv, gzip, json, sys, statistics as st
from collections import Counter, defaultdict
ann_path, cand_path, out_path = sys.argv[1:4]
cands = list(csv.DictReader(open(cand_path)))
want = {r["object_uid"] for r in cands}
d = json.load(gzip.open(ann_path))
rows = []
for r in cands:
    a = d.get(r["object_uid"])
    if a is None:
        continue
    bb = (a.get("thor_metadata") or {}).get("assetMetadata", {}).get("boundingBox")
    dims = [round(bb["max"][k] - bb["min"][k], 4) for k in "xyz"] if bb else [None] * 3
    lic = (a.get("license_info") or {}).get("license")
    rows.append(dict(role=r["role"], object_uid=r["object_uid"], dm_type=r["type"],
                     dm_type_prob=r["type_prob"], objathor_category=a.get("category"),
                     synset=a.get("synset"), height_cm=a.get("height"),
                     max_dim_cm=a.get("max_dimension"), mass_kg=a.get("mass"),
                     materials="|".join(a.get("materials") or []), quality=a.get("quality"),
                     receptacle=a.get("receptacle"), front_view=a.get("frontView"),
                     bbox_x_m=dims[0], bbox_y_m=dims[1], bbox_z_m=dims[2], license=lic,
                     description=a.get("description")))
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
by = defaultdict(list)
for r in rows: by[r["role"]].append(r)
tot = Counter(r["role"] for r in cands)
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 2) if xs else None
print(f"matched {len(rows)} / {len(cands)}")
for role in tot:
    rs = by[role]
    q7 = [r for r in rs if (r["quality"] or 0) >= 7]
    print(f"\n{role}: {len(rs)}/{tot[role]} annotated, quality>=7: {len(q7)}")
    print("  median height_cm", med(r["height_cm"] for r in rs), "max_dim_cm", med(r["max_dim_cm"] for r in rs), "mass_kg", med(r["mass_kg"] for r in rs))
    print("  licences", Counter(r["license"] for r in rs).most_common(5))
    print("  materials", Counter(m for r in rs for m in r["materials"].split("|") if m).most_common(6))
    print("  objathor categories", Counter(r["objathor_category"] for r in rs).most_common(6))
