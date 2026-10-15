"""Contact-region labels (which part of A touches which part of B at the most robust placement) for pair maps.

    python scripts/sim_contact_regions.py PICKS_JSON GLB_DIR SHAKER_CSV OUT_DIR --pairs PAIR_DIR [PAIR_DIR ...] [--only name1 name2 ...] [--coacd DIR]
Writes OUT_DIR/<pair>.A.npz and .B.npz (arrays `contact`, `gap_mm`; meshes alongside as <pair>.A.mesh.npz).
"""
import argparse
import glob
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from affdet.geometry.prepare import prepare  # noqa: E402
from affdet.sim import label_export as le  # noqa: E402
from affdet.sim.registry import Registry  # noqa: E402


def spec_of(name):
    m = re.match(r"^(shaker)([0-9a-f]{8})$", name)
    if m:
        return f"shaker:{m.group(2)}"
    m = re.match(r"^(.*?)(\d+)$", name)
    return f"{m.group(1)}:{m.group(2)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("glb"); ap.add_argument("shaker_csv"); ap.add_argument("out")
    ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--only", nargs="*", default=[]); ap.add_argument("--coacd", default="/tmp/coacd_contact")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = Registry(a.picks, a.glb, a.shaker_csv, a.coacd)
    for d in a.pairs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*_on_*.pkl"), recursive=True)):
            name = os.path.basename(f)[:-4]
            if a.only and name not in a.only:
                continue
            tn, bn = name.split("_on_")
            ts, bs = spec_of(tn), spec_of(bn)
            m = pickle.load(open(f, "rb"))
            top, _ = reg.get(ts); base, _ = reg.get(bs)
            tm = prepare(*reg.glb(ts))[1]; bm = prepare(*reg.glb(bs))[1]
            lab, info = le.contact_labels(m, top, base, tm, bm)
            if lab is None:
                continue
            for side in ("A", "B"):
                le.save(os.path.join(a.out, f"{name}.{side}.npz"), {k: (v.astype(np.float16) if v.dtype.kind == "f" else v) for k, v in lab[side].items() if k not in ("vertices", "faces")},
                        {"pair": name, "side": side, **info})
                np.savez_compressed(os.path.join(a.out, f"{name}.{side}.mesh.npz"), vertices=lab[side]["vertices"].astype(np.float32), faces=lab[side]["faces"].astype(np.int32))
            print(name, info, "contact vertices A/B:", int(lab["A"]["contact"].sum()), int(lab["B"]["contact"].sum()), flush=True)


if __name__ == "__main__":
    main()
