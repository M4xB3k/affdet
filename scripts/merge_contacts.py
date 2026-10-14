"""Add the contact-region channels (scripts/sim_contact_regions.py) to the exported pair label files.

    python scripts/merge_contacts.py LABELS_DIR CONTACT_DIR [CONTACT_DIR ...]

For every `<pair>.<A|B>.npz` found in a contact directory, `contact` (int8) and `contact_gap_mm` (float16) are added to
LABELS_DIR/pair/<pair>.<A|B>.npz, and `meta.contact_placement` records the placement they describe. The vertex counts must match
(both exports use the same subdivision); a mismatch is reported and skipped. The per-pair mesh files of the contact run are not needed.
"""
import glob
import json
import os
import sys

import numpy as np


def main():
    labels, dirs = sys.argv[1], sys.argv[2:]
    done = skipped = missing = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.[AB].npz"))):
            if f.endswith(".mesh.npz"):
                continue
            target = os.path.join(labels, "pair", os.path.basename(f))
            if not os.path.exists(target):
                missing += 1
                continue
            c, t = np.load(f), np.load(target)
            cm, tm = json.loads(str(c["meta"])), json.loads(str(t["meta"]))
            if len(c["contact"]) != len(t["valid"]):
                print("vertex count mismatch, skipped:", os.path.basename(f), len(c["contact"]), len(t["valid"]))
                skipped += 1
                continue
            arrays = {k: t[k] for k in t.files if k != "meta"}
            arrays["contact"] = c["contact"].astype(np.int8)
            arrays["contact_gap_mm"] = c["gap_mm"].astype(np.float16)
            tm["contact_placement"] = {k: cm[k] for k in ("face", "x", "y", "stable", "tilt_deg", "n_contact_points") if k in cm}
            np.savez_compressed(target, meta=json.dumps(tm), **arrays)
            done += 1
    print(f"merged {done}, skipped {skipped}, no label file for {missing}")


if __name__ == "__main__":
    main()
