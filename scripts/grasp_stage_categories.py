"""Clamp / lift / level per category over a full set of stage maps, next to the old maps of the same objects.

    python scripts/grasp_stage_categories.py NEW_MAP_DIR OLD_MAP_DIR

Per object: share of its grasp tests (cell, yaw) that are on the object and not blocked. Per category: the mean over its
objects, so every object counts once. Columns:
  old held, old level   the old maps (0.1 s wait): held, and held with the whole rotation under 30 deg
  held, level (swing)   the same two definitions on the new maps (0.5 s wait): what the longer wait alone changes
  clamp, lift, level    the nested stages of the new maps (level = tilt of the up axis from rest under 30 deg)
  lifted w/o clamp      held although the clamp stage failed (wedged or carried on an edge)
"""
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

new_dir, old_dir = sys.argv[1], sys.argv[2]
per = defaultdict(list)
for f in sorted(glob.glob(os.path.join(new_dir, "grasp_*.pkl"))):
    m = pickle.load(open(f, "rb"))
    ok = m["on_object"] & ~m["blocked"]
    if not ok.any():
        continue
    o = pickle.load(open(os.path.join(old_dir, os.path.basename(f)), "rb"))
    oko = o["on_object"] & ~o["blocked"]
    sh = lambda a, k=ok: 100 * a[k].mean()
    per[m["spec"].split(":")[0]].append([
        ok.sum(), 100 * m["blocked"][m["on_object"]].mean(),
        sh(o["held"], oko) if oko.any() else np.nan, sh(o["held"] & (o["swing_deg"] < 30), oko) if oko.any() else np.nan,
        sh(m["held"]), sh(m["held"] & (m["swing_deg"] < 30)),
        sh(m["clamp"]), sh(m["lift"]), sh(m["level"]), sh(m["held"] & ~m["clamp"]), sh(m["held"] & (m["tilt_max_deg"] < 30))])

head = ["category", "objects", "tests", "blocked %", "old held", "old level", "held", "level (swing)", "clamp", "lift", "level", "lifted w/o clamp", "held & tilt < 30"]
print("| " + " | ".join(head) + " |"); print("|" + "---|" * len(head))
allrows = []
for c in sorted(per):
    r = np.array(per[c], float); allrows.append(r)
    print(f"| {c} | {len(r)} | {int(r[:, 0].sum())} | " + " | ".join(f"{x:.0f}" for x in np.nanmean(r[:, 1:], 0)) + " |")
r = np.concatenate(allrows)
print(f"| **all objects** | {len(r)} | {int(r[:, 0].sum())} | " + " | ".join(f"{x:.0f}" for x in np.nanmean(r[:, 1:], 0)) + " |")
