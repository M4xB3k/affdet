"""Per-face UTAMP-2 labels for a curated Objaverse set (e.g. salt_shaker_set.csv).

For each row: load the GLB (one half of a two-object model when the note says
"use half ..."), turn the `up_axis` column's glTF axis to +z (default +y),
scale to `target_height_cm`, label the faces (affdet/geometry/face_labels.py)
and render a check sheet.

    PYOPENGL_PLATFORM=osmesa python scripts/label_faces.py \
        docs/objaverse/salt_shaker_set.csv data/objaverse/glbs data/face_labels/salt_shaker \
        --summary docs/objaverse/salt_shaker_face_labels.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import traceback
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from affdet.geometry.face_labels import FACES, Params, label_faces  # noqa: E402
from affdet.geometry.prepare import prepare  # noqa: E402


def halves(note: str):
    m = re.search(r"use half ([\d and,]+)", note or "")
    return [int(h) for h in re.findall(r"\d", m.group(1))] if m else [None]


def summary_row(oid, row, lab):
    f = {x.face: x for x in lab.faces}
    g = [x for x in lab.grasps if x.graspable]
    ex = [round(100 * e, 1) for e in lab.extents]
    out = {"object_id": oid, "tier": row["tier"], "w_cm": ex[0], "d_cm": ex[1], "h_cm": ex[2],
           "mass_kg": lab.mass}
    for name in FACES:
        out[f"supportable_{name}"] = int(f[name].supportable)
    for name in FACES:
        out[f"tip_deg_{name}"] = round(f[name].tip_deg, 1)
    out["review"] = " ".join(f"{x.face}({x.smooth_deg:.0f}deg)" for x in lab.faces if x.review)
    out["supportive_on"] = int(f["on"].supportive)
    out["top_flat_width_cm"] = round(100 * f["on"].support_min_width, 1)
    out["n_grasps"] = len(g)
    out["grasps"] = " ".join(f"{x.palm}:{x.finger1}-{x.finger2}" for x in g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("glb_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--summary")
    ap.add_argument("--only", nargs="*", help="object ids (uid or uid_half) to run")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "check"), exist_ok=True)
    p = Params()
    rows_out = []
    for row in csv.DictReader(open(a.csv)):
        for h in halves(row.get("note")):
            uid = row["object_uid"]
            oid = uid if h is None else f"{uid}_{h}"
            if a.only and oid not in a.only and uid not in a.only:
                continue
            try:
                scene, mesh = prepare(os.path.join(a.glb_dir, f"{uid}.glb"),
                                      float(row["target_height_cm"]) / 100,
                                      row.get("up_axis") or "+y", h)
                lab = label_faces(mesh, p, mass=float(row["target_mass_kg"] or 0) or None, uid=oid)
            except Exception:
                print(f"{oid}: FAILED\n{traceback.format_exc()}")
                continue
            d = lab.to_dict()
            d["predicates"] = lab.predicates("obj")
            d["params"] = vars(p)
            json.dump(d, open(os.path.join(a.out_dir, f"{oid}.json"), "w"), indent=1)
            s = summary_row(oid, row, lab)
            rows_out.append(s)
            sup = "".join(k[0].upper() if s[f"supportable_{k}"] else "." for k in FACES)
            print(f"{oid} {row['tier']:<11} {s['w_cm']:>5}x{s['d_cm']:>5}x{s['h_cm']:>5} cm  "
                  f"supportable[OUFBLR]={sup}  top-flat {s['top_flat_width_cm']} cm  grasps {s['n_grasps']}"
                  + (f"  REVIEW smooth-shaded: {s['review']}" if s["review"] else ""))
            if not a.no_render:
                from affdet.geometry.render_check import check_sheet
                text = [f"{oid}  {row['tier']}  {s['w_cm']} x {s['d_cm']} x {s['h_cm']} cm  "
                        f"(grid 1 cm)  {row.get('note', '')[:80]}",
                        "supportable: " + ", ".join(f"{x.face} ({x.tip_deg:.0f} deg{', REVIEW: shaded round' if x.review else ''})" for x in lab.faces if x.supportable),
                        "not supportable: " + ", ".join(f"{x.face} ({x.tip_deg:.0f} deg)" for x in lab.faces if not x.supportable),
                        f"supportive on: {bool(s['supportive_on'])} (flat top {s['top_flat_width_cm']} cm)   "
                        f"grasps: {s['grasps']}"]
                check_sheet(scene, text).save(os.path.join(a.out_dir, "check", f"{oid}.jpg"), quality=85)
    if a.summary and rows_out:
        with open(a.summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows_out)
    print(f"{len(rows_out)} objects labelled")


if __name__ == "__main__":
    main()
