"""Summarise a pair-map matrix: one cell per (top, base class) with the share of B that can carry A and the best margin.

    python scripts/pair_matrix_report.py JSONL [JSONL ...] [--markdown OUT.md]
"""
import argparse
import collections
import json


def cls(spec):
    return "shaker" if spec.startswith("shaker") else spec.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+"); ap.add_argument("--markdown")
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.jsonl for l in open(f)]
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    tops = sorted({r["top"] for r in rows}); bases = sorted({cls(r["base"]) for r in rows})
    cell = collections.defaultdict(list)
    for r in ok:
        cell[(r["top"], cls(r["base"]))].append(r)
    lines = [f"{len(ok)} pair maps, {len(err)} pairs where no face of the top holds on a flat table, {sum(r['simulations'] for r in ok)} simulations "
             f"({sum(r['seconds'] for r in ok) / 3600:.1f} job-hours)\n",
             "| top \\ base class | " + " | ".join(bases) + " |", "|---|" + "---|" * len(bases)]
    for t in tops:
        row = []
        for b in bases:
            v = cell.get((t, b))
            row.append("-" if not v else "%.0f %% (%.0f mm)" % (100 * sum(r["b_supported_frac"] for r in v) / len(v), sum(r["best_b_margin_mm"] for r in v) / len(v)))
        lines.append(f"| {t} | " + " | ".join(row) + " |")
    lines.append("\nCells: share of the base's grid points that can carry the top with some face (mean over the pairs in the cell), "
                 "and the mean best margin in mm. Empty tops (no face rests) are listed below.\n")
    for r in err:
        lines.append(f"- {r['pair']}: {r['error']}")
    text = "\n".join(lines)
    print(text)
    if a.markdown:
        open(a.markdown, "w").write(text + "\n")


if __name__ == "__main__":
    main()
