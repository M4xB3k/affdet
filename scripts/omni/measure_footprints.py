"""Measure calibrated placement footprints per category.

For each curated category, import its representative instance, apply the shared
scale calibration (affdet.render.omni_scale) exactly as the renderer does, and
record its worst-case horizontal footprint. Writes
scripts/omni/calibrated_footprints.json = {cat: [long_m, short_m]} which the
sampler reads for non-overlap spacing.

Why worst-case horizontal: the sampler places objects on the table BEFORE the
physics drop, but the drop tips elongated tools flat. A knife measured standing
upright would report a tiny footprint, then land lying across a much larger
area. So the footprint uses the two LARGEST calibrated bbox dimensions (the two
that can end up in the XY plane once an object rests flat) rather than the
post-orientation XY box.

Run on the cluster (has meshes + Blender):
    ~/blender-5.1/blender --background --python scripts/omni/measure_footprints.py -- \
        --curation scripts/omni/curation.json \
        --omni-root /scratch/maksym.bek/omni/raw_scans \
        --overrides scripts/omni/scale_overrides.json \
        --out scripts/omni/calibrated_footprints.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bpy

from affdet.render.omni_scale import (
    apply_calibrated_scale, orient_upright, bake_import_rotation, load_overrides,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--curation", required=True)
    p.add_argument("--omni-root", required=True)
    p.add_argument("--overrides", default="scripts/omni/scale_overrides.json")
    p.add_argument("--out", default="scripts/omni/calibrated_footprints.json")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args, _ = p.parse_known_args(argv)
    return args


def _representative(cat: str, entry: dict, omni_root: Path) -> str | None:
    """Pick the instance id to measure: the curation 'representative' if its
    mesh exists, else the first available whole/piece/any instance."""
    def mesh_ok(inst: str) -> bool:
        return (omni_root / cat / "extracted" / inst / "Scan" / "Scan.obj").exists()

    rep = entry.get("representative")
    if rep is not None:
        rep_inst = f"{cat}_{rep}"
        if mesh_ok(rep_inst):
            return rep_inst

    pl = entry.get("placement", {})
    drops = set(entry.get("drop", []))
    for form in ("whole", "piece", "opened"):
        for i in pl.get(form, []):
            if i not in drops and mesh_ok(f"{cat}_{i}"):
                return f"{cat}_{i}"
    # no placement split: scan the extracted dir
    ext = omni_root / cat / "extracted"
    if ext.exists():
        for d in sorted(ext.iterdir()):
            if d.name.split("_")[-1] not in drops and mesh_ok(d.name):
                return d.name
    return None


def _import(mesh_path: Path, name: str) -> bpy.types.Object:
    before = {o.name for o in bpy.data.objects}
    bpy.ops.wm.obj_import(filepath=str(mesh_path))
    new = [bpy.data.objects[n] for n in ({o.name for o in bpy.data.objects} - before)
           if bpy.data.objects[n].type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    return obj


def main() -> None:
    args = _parse_args()
    omni_root = Path(args.omni_root)
    overrides = load_overrides(args.overrides)
    with open(args.curation) as f:
        curation = json.load(f)

    footprints: dict[str, list[float]] = {}
    for cat, entry in curation.items():
        if cat.startswith("_") or not isinstance(entry, dict):
            continue
        inst = _representative(cat, entry, omni_root)
        if inst is None:
            print(f"[footprint] {cat}: no mesh, skipped")
            continue
        mesh_path = omni_root / cat / "extracted" / inst / "Scan" / "Scan.obj"

        # Fresh scene per object to keep measurements isolated + memory low.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        obj = _import(mesh_path, inst)
        bake_import_rotation(obj)
        apply_calibrated_scale(obj, cat, inst, overrides)
        orient_upright(obj, cat)

        bpy.context.view_layer.update()
        bb = obj.bound_box
        dims = sorted(
            (max(v[i] for v in bb) - min(v[i] for v in bb) for i in range(3)),
            reverse=True,
        )
        # Worst-case horizontal footprint: two largest calibrated dimensions.
        long_m, short_m = float(dims[0]), float(dims[1])
        footprints[cat] = [round(long_m, 4), round(short_m, 4)]
        print(f"[footprint] {cat} ({inst}): {footprints[cat]}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(footprints, indent=2, sort_keys=True))
    print(f"[footprint] wrote {len(footprints)} categories to {out_path}")


if __name__ == "__main__":
    main()
