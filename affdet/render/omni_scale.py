"""Shared OmniObject3D scale + orientation calibration.

Single source of truth for how a raw OmniObject3D mesh is scaled and oriented
before it enters a scene. Both the diagnostic calibration tool
(`scripts/omni/raw_scale_check.py`) and the real scene renderer
(`affdet/render/omni_blender.py`) import from here so their behaviour cannot
drift.

Scale model (deliberately NOT normalize-to-target): the raw scan is scaled by a
single global factor to bring scanner units into a visible metric range, then
multiplied by a per-category factor and/or a per-instance override. This
preserves the real per-instance size variation between scans (e.g. a large peach
scan stays larger than a small one) — unlike forcing every instance to a fixed
longest-axis target.

    final_scale = GLOBAL_SCALE * multiplier
    multiplier  = overrides[inst]                        # per-instance wins
               or PEELER_SCALE if inst is a peeler
               or CATEGORY_SCALE.get(cat, 1.0)           # else category default

A per-instance override REPLACES the category/peeler default (it does not stack)
— outlier calibration, e.g. peach_052 at 0.64 means 0.64 total.
"""
from __future__ import annotations

import json
from pathlib import Path

import bpy

# Base multiplier applied to every object before category/instance overrides —
# brings raw scanner units into a camera-visible metric range.
GLOBAL_SCALE = 0.0015

# Per-category multiplier on top of GLOBAL_SCALE, calibrated by eye against a
# fixed apple+bowl reference. Categories not listed use 1.0.
CATEGORY_SCALE: dict[str, float] = {
    "apple": 0.93,
    "dish": 1.2,
    "cup": 0.64,
    "kettle": 0.74,
    "pan": 1.08,
    "red_wine_glass": 0.7,
    "teapot": 0.6,
    "thermos": 0.5,
    "vase": 0.7,
    "bread": 1.6,
    "broccoli": 0.75,
    "onion": 0.8,
    "peach": 0.8,  # peach_052/053 are large-scan outliers, pinned to 0.64 in overrides
}

# Per curation.json, these knife_* instances are actually "peeler" (recategorised
# from knife, but the mesh still lives under the knife/ folder). The scale
# calibrated on knife_011 (0.935) applies to all of them by default so a new
# peeler instance doesn't have to be re-tuned from scratch.
PEELER_INSTANCES = {"knife_011", "knife_014", "knife_018", "knife_021"}
PEELER_SCALE = 0.935

# Categories whose objects should be auto-rotated to stand on their longest
# axis (elongated handheld tools). Containers (dish, cup, kettle, bowl,
# bottle, ...) are already upright in their native post-import orientation
# and are excluded so this rotation doesn't knock them onto their side.
STAND_ON_LONG_AXIS = {"knife", "fork", "scissor", "banana", "bread"}


def load_overrides(path: str | Path) -> dict[str, float]:
    """Read the per-instance {instance_id: multiplier} override JSON."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_rotation_overrides(path: str | Path) -> dict[str, list[float]]:
    """Read per-instance rotation corrections {instance_id: [x_deg,y_deg,z_deg]}.

    For scans that were captured in a tipped/rotated pose (e.g. some teapots
    lie on their side), so their native post-import orientation is NOT upright
    and the category orientation logic doesn't cover them. The correction is a
    fixed Euler rotation (degrees) applied right after import, baked in, so the
    object then starts upright like a normal scan."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def apply_rotation_override(obj: bpy.types.Object, inst: str,
                            rot_overrides: dict[str, list[float]]) -> bool:
    """If `inst` has a rotation correction, apply it (Euler XYZ degrees) and
    bake it in. Returns True if a correction was applied. Call right after
    import/bake so it composes cleanly with a (0,0,0) starting rotation."""
    rot = rot_overrides.get(inst)
    if not rot:
        return False
    import math
    from mathutils import Euler
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    eul = Euler([math.radians(a) for a in rot], "XYZ")
    obj.matrix_world = eul.to_matrix().to_4x4() @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.context.view_layer.update()
    return True


def calibrated_multiplier(cat: str, inst: str, overrides: dict[str, float]) -> float:
    """Resolve the multiplier for one instance.

    A per-instance override REPLACES (not stacks with) the category/peeler
    default. Peeler instances get PEELER_SCALE; everything else falls back to
    the category default or 1.0.
    """
    if inst in overrides:
        return overrides[inst]
    if inst in PEELER_INSTANCES:
        return PEELER_SCALE
    return CATEGORY_SCALE.get(cat, 1.0)


def bake_import_rotation(obj: bpy.types.Object) -> None:
    """Apply the OBJ importer's baked coordinate-conversion rotation (e.g.
    Z-up -> Y-up) so downstream rotation logic starts from a clean (0,0,0)
    rotation instead of composing with an unknown one. Call immediately after
    import + join."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(rotation=True)


def apply_calibrated_scale(
    obj: bpy.types.Object, cat: str, inst: str, overrides: dict[str, float]
) -> float:
    """Scale `obj` by GLOBAL_SCALE * calibrated_multiplier and bake it in.
    Returns the multiplier used (for logging)."""
    mult = calibrated_multiplier(cat, inst, overrides)
    s = GLOBAL_SCALE * mult
    obj.scale = (s, s, s)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    return mult


def orient_upright(obj: bpy.types.Object, cat: str) -> None:
    """For STAND_ON_LONG_AXIS categories, rotate so the LONGEST local axis
    points straight up (standing on its handle) and the SHORTEST local axis
    runs front-to-back, so the broad face (mid axis) turns toward the camera
    on world-X instead of showing a thin edge-on silhouette. Meant for
    elongated handheld tools (knives, forks, ...) — NOT containers, which are
    already upright in their native post-import orientation and would be
    knocked onto their side by this rotation. No-op for other categories."""
    if cat not in STAND_ON_LONG_AXIS:
        return
    from mathutils import Matrix, Vector

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()

    bb0 = obj.bound_box
    dims0 = [max(v[i] for v in bb0) - min(v[i] for v in bb0) for i in range(3)]
    long_axis = max(range(3), key=lambda i: dims0[i])
    short_axis = min(range(3), key=lambda i: dims0[i])
    mid_axis = ({0, 1, 2} - {long_axis, short_axis}).pop()
    axis_map = {0: mid_axis, 1: short_axis, 2: long_axis}
    basis = [Vector((0, 0, 0)) for _ in range(3)]
    for world_axis, local_axis in axis_map.items():
        v = Vector((0, 0, 0))
        v[local_axis] = 1.0
        basis[world_axis] = v
    rot_mat = Matrix((tuple(basis[0]), tuple(basis[1]), tuple(basis[2]))).transposed().to_4x4()
    obj.matrix_world = rot_mat @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


# Threshold on container_first_hit_z_frac: above this the container is NOT in an
# opening-up (upright) pose — it landed upside-down OR on its side, and either
# way can't contain / be poured into. Verified 8/8 upright-vs-flipped on
# dish/bowl/cup/red_wine_glass with a clean gap (worst upright 0.68, best 0.84).
NOT_UPRIGHT_THRESH = 0.75


def container_first_hit_z_frac(obj: bpy.types.Object) -> float | None:
    """Cast rays straight DOWN through the container's centre and return the
    mean height (as a fraction of the object's height) of the first surface
    each ray hits. Low => a ray drops through the OPENING and hits the inner
    base (opening on top, upright). High => the ray immediately hits the closed
    underside (opening on bottom, upside-down). Returns None if no ray hits."""
    from mathutils import Vector

    bpy.context.view_layer.update()
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not verts:
        return None
    zmin = min(p.z for p in verts)
    zmax = max(p.z for p in verts)
    xmin, xmax = min(p.x for p in verts), max(p.x for p in verts)
    ymin, ymax = min(p.y for p in verts), max(p.y for p in verts)
    h = (zmax - zmin) or 1e-9
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    rx, ry = (xmax - xmin) / 2, (ymax - ymin) / 2

    mw = obj.matrix_world
    mwi = mw.inverted()
    firsts: list[float] = []
    for ox, oy in [(0, 0), (0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25)]:
        origin = Vector((cx + ox * rx, cy + oy * ry, zmax + 0.1))
        lo = mwi @ origin
        ld = (mwi.to_3x3() @ Vector((0, 0, -1))).normalized()
        hits, cur = [], lo.copy()
        for _ in range(8):  # allow multiple surface crossings
            ok, loc, nrm, idx = obj.ray_cast(cur, ld)
            if not ok:
                break
            hits.append((mw @ loc).z)
            cur = loc + ld * 1e-4
        if hits:
            firsts.append(max(hits))  # highest (first-from-top) surface
    if not firsts:
        return None
    return (sum(firsts) / len(firsts) - zmin) / h


# container_first_hit_z_frac casts rays near the object's XY bbox centre; for a
# container whose true opening isn't centred there (off-centre handle, lidded/
# spouted design, decorated lid, etc.), every sample can land on the rim/wall
# instead of the mouth in EITHER orientation, misjudging an upright instance as
# flipped regardless of its real pose. Tried widening/re-weighting the ray-cast
# grid (MIN and mean statistics) and validated both against 60 real cup/
# red_wine_glass instances (native vs. manually-flipped ground truth,
# 2026-07-12): red_wine_glass separates reliably (~95%), but cup does not
# (~64-83% depending on statistic) -- cups are too shape-diverse (mugs, lidded
# travel cups, cans, spouts) for one geometric rule. Falling back to manual
# visual review per instance instead of further tuning the heuristic.
#
# This set was built by rendering every currently-flagged cup pair through the
# real pipeline and having a human look at the actual image (not the detector's
# opinion) — confirmed genuinely upright in every render checked. Per-instance
# detector exception, not a physics/rendering fix -- analogous to PAN_DROP, but
# here the instance is GOOD and the check is wrong, not the instance being bad.
UPRIGHT_DETECTOR_OVERRIDE: set[str] = {
    "cup_002", "cup_003", "cup_010", "cup_011", "cup_012", "cup_015", "cup_018",
    "cup_019", "cup_020", "cup_021", "cup_022", "cup_023", "cup_025", "cup_026",
    "cup_027", "cup_028", "cup_029", "cup_030", "cup_031", "cup_032", "cup_033",
    "cup_034", "cup_035", "cup_036", "cup_037", "cup_038", "cup_039",
}


def is_container_not_upright(obj: bpy.types.Object, thresh: float = NOT_UPRIGHT_THRESH) -> bool:
    """True if the container is NOT opening-up — landed upside-down or on its
    side — so it can't contain / be poured into. Uses container_first_hit_z_frac;
    returns False when undetermined (no ray hit). Instances in
    UPRIGHT_DETECTOR_OVERRIDE always return False (verified always-upright,
    ray-cast is unreliable for their geometry -- see that set's docstring)."""
    if obj.name in UPRIGHT_DETECTOR_OVERRIDE:
        return False
    frac = container_first_hit_z_frac(obj)
    return frac is not None and frac > thresh


def orient_container_opening_up(obj: bpy.types.Object) -> None:
    """Pre-drop bias: if the container's opening currently faces DOWN, flip it
    180deg about X so it starts opening-up. Combined with a short drop + high
    friction, containers then usually settle upright (positive canContain/
    canPourInto fill reliably) while physics can still occasionally topple one
    for natural negatives. No-op if the opening already faces up or is
    undetermined."""
    frac = container_first_hit_z_frac(obj)
    if frac is None or frac <= NOT_UPRIGHT_THRESH:
        return
    from mathutils import Matrix
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.matrix_world = Matrix.Rotation(3.14159265358979, 4, "X") @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.context.view_layer.update()
