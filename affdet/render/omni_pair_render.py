"""Paired-scene renderer for VLM relabeling — fixes tile-based scale blindness.

Motivation (tools/docs/2026-07-11-session-findings-cansupport-detector.md, §5):
the VLM labeler currently scores each (actor, recipient) pair from two
INDEPENDENTLY auto-zoomed tile images (scripts/omni/omni_render_sheet.py),
which destroys real relative scale before scoring ever happens. This script
composes actor + recipient TOGETHER in one real scene (true scale, real
camera, real physics settling — reusing the same Blender machinery as the
training-data renderer, affdet/render/omni_blender.py) and renders it from a
configurable grid of camera angles, so a pair can be re-labeled from an image
that actually shows how big each object is relative to the other.

Scope: this ONLY produces new images for the labeling step. It does not
touch data/scenes/ (the training-data scene corpus) or dashboard/tiles/ (the
single-object tile set) and has no opinion on how the VLM prompt changes.

Called as:
    blender --background --python affdet/render/omni_pair_render.py -- \\
        --pairs data/labels/pair_pilot/pairs.json \\
        --omni-root /path/to/omni/raw_scans \\
        --out data/labels/pair_pilot/renders \\
        --grid 2x2

Input --pairs file: a JSON list of {"actor": "<cat>_<id>", "recipient": "<cat>_<id>"}.
Output, per pair, under <out>/<actor>__<recipient>/:
    angle_00.png ... angle_NN.png   — one render per camera angle
    grid.png                        — contact-sheet of all angles, for quick QA
    manifest.json                   — instance ids, classes, angle metadata,
                                       grid shape (so downstream code/labeler
                                       code knows exactly what it's looking at)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bpy
from mathutils import Vector

from affdet.render.blender import (
    _reset_scene, _setup_render, _setup_world, _setup_table,
    _ensure_camera, _set_camera_pose, _snap_to_surface,
    _setup_compositor_index_pass, _read_index_exr,
)
from affdet.render.omni_blender import _import_omni_mesh, _run_physics_drop
from affdet.render.omni_sampler import _footprint, TABLE_HEIGHT_M, _CONTAINERS
from affdet.render.omni_scale import (
    load_overrides, load_rotation_overrides, is_container_not_upright,
)
from affdet.render.omni_pair_grid import (
    parse_grid, split_instance, resolve_mesh_cat, grid_camera_poses,
)
from affdet.render.mask_export import extract_bboxes_from_pass
from affdet.label.resolved_instances import load_dropped_set

RESOLUTION = (480, 360)
CAMERA_FOV_DEG = 40.0


def _pair_rng(seed: int, actor_inst: str, recipient_inst: str) -> np.random.RandomState:
    """Deterministic per-pair RNG seeded from (seed, actor, recipient) instead
    of shared run-order state. A pair's rendered pose must depend only on its
    own identity, never on how many pairs preceded it in a given invocation --
    the old shared-stream approach silently invalidated re-renders of any
    subset of pairs (order/batch-membership dependent)."""
    key = f"{seed}:{actor_inst}__{recipient_inst}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    pair_seed = int.from_bytes(digest[:4], "big")
    return np.random.RandomState(pair_seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="JSON list of {actor, recipient}")
    parser.add_argument("--resolved", required=True,
                         help="Path to resolved_instances_v2.json (the single source of "
                              "truth for which instances are dropped -- never rendered, "
                              "even if pairs.json contains one; see load_dropped_set)")
    parser.add_argument("--omni-root", required=True, help="Path to raw_scans root dir")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--tex-dir", default="data/table_textures")
    parser.add_argument("--grid", default="2x2",
                         help="Camera grid ROWSxCOLS, e.g. 1x1, 2x2, 3x2. "
                              "Rows vary elevation, cols vary azimuth.")
    parser.add_argument("--overrides", default="scripts/omni/scale_overrides.json")
    parser.add_argument("--rotation-overrides",
                         default="scripts/omni/rotation_overrides.json")
    parser.add_argument("--seed", type=int, default=0)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args, _ = parser.parse_known_args(argv)
    return args


def _mesh_path(omni_root: Path, cat: str, inst_id: str) -> Path:
    return omni_root / cat / "extracted" / inst_id / "Scan" / "Scan.obj"


def _place_pair(
    omni_root: Path, actor_inst: str, recipient_inst: str,
    overrides: dict, rot_overrides: dict, rng: np.random.RandomState,
) -> tuple[list, list[str]]:
    """Import + place actor and recipient side by side on the table, spaced
    by their real calibrated footprints (same spacing logic as the coverage
    sampler). Returns (placed_bpy_objects, obj_classes); camera framing is
    computed separately from their REAL mesh bounds after physics settles
    (see _scene_bounds), since footprint here is only horizontal placement
    spacing, not an accurate camera-framing bound (no height, no post-physics
    rotation/settling). Pre-drop rotation is a pure yaw on top of the already-
    upright import pose, so X/Y stay at (0, 0) -- see _snap_container_upright,
    which relies on that to restore upright without re-simulating."""
    placed = []
    classes = []
    total_half_w = 0.0
    specs = []
    for inst in (actor_inst, recipient_inst):
        cat, _ = split_instance(inst)
        mesh_cat = resolve_mesh_cat(cat)
        # peeler instances physically live under knife/extracted/knife_NNN/ on
        # disk (recategorized knife, same scan) -- the FILENAME needs the same
        # peeler_NNN -> knife_NNN remap resolve_mesh_cat applies to the
        # category dir, or the lookup 404s even though the mesh exists (same
        # fix omni_sampler.py already applies for the training-scene renderer).
        mesh_inst = inst.replace("peeler_", "knife_", 1) if cat == "peeler" else inst
        mesh = _mesh_path(omni_root, mesh_cat, mesh_inst)
        if not mesh.exists():
            raise FileNotFoundError(f"Mesh not found for {inst}: {mesh}")
        fw, fd = _footprint(cat)
        specs.append((inst, cat, mesh_cat, mesh, fw, fd))
        total_half_w += fw / 2.0

    gap = 0.05
    x = -(total_half_w + gap / 2.0) + specs[0][4] / 2.0
    for i, (inst, cat, mesh_cat, mesh, fw, fd) in enumerate(specs):
        obj = _import_omni_mesh(mesh, inst, mesh_cat, overrides, rot_overrides)
        obj.location = [x, 0.0, TABLE_HEIGHT_M + 0.15]
        obj.rotation_euler = [0.0, 0.0, float(rng.uniform(0.0, 2.0 * math.pi))]
        obj.pass_index = i + 1
        placed.append(obj)
        classes.append(cat)
        if i == 0:
            x += fw / 2.0 + gap + specs[1][4] / 2.0

    return placed, classes


def _flipped_containers(placed: list, classes: list[str]) -> list:
    """Objects among `placed` that are containers and settled NOT opening-up
    (upside-down or on their side) — unusable for a canContain/canSupport
    labeling image. Same detector the training-scene renderer uses to decide
    pos->neg downgrades (affdet/render/omni_scale.is_container_not_upright)."""
    return [
        obj for obj, cat in zip(placed, classes)
        if cat in _CONTAINERS and is_container_not_upright(obj)
    ]


def _snap_container_upright(obj: bpy.types.Object) -> None:
    """Force a flipped container back to its canonical upright pose in place.

    _place_pair only ever gives an object a pure-yaw pre-drop rotation
    (rotation_euler = [0, 0, random_yaw]) on top of the as-scanned pose baked
    in by _import_omni_mesh (bake_import_rotation + orient_upright; no
    pre-drop container flip anymore -- see that function's docstring) -- so
    "upright" for X/Y is always (0, 0). Physics can rotate the object off
    that during the drop; when it settles flipped, zeroing X/Y (keeping
    whatever yaw it settled at) restores the native scan orientation without
    re-simulating. Then re-rest it on the table since rotating in place can
    lift it off or sink it into the surface."""
    rot = obj.rotation_euler
    obj.rotation_euler = (0.0, 0.0, rot.z)
    bpy.context.view_layer.update()
    _snap_to_surface(obj, surface_z=0.0)


def _scene_bounds(objects: list) -> tuple[Vector, float]:
    """True world-space bounding sphere (center, radius) of the given objects'
    ACTUAL mesh geometry (post-import, post-physics-settle) — not an estimate
    from footprint tables, so tall/oddly-shaped objects (durian, bowls, etc.)
    can't get clipped regardless of their horizontal footprint."""
    bpy.context.view_layer.update()
    coords = [
        obj.matrix_world @ Vector(c)
        for obj in objects
        for c in obj.bound_box
    ]
    minv = Vector((min(c[i] for c in coords) for i in range(3)))
    maxv = Vector((max(c[i] for c in coords) for i in range(3)))
    center = (minv + maxv) / 2.0
    radius = (maxv - minv).length / 2.0
    return center, max(radius, 0.05)


def _assemble_grid(image_paths: list[Path], rows: int, cols: int, out_path: Path) -> None:
    """Compose a contact sheet from already-rendered angle PNGs using Blender's
    own image API (bpy.data.images) instead of PIL — Blender's bundled Python
    interpreter has no PIL, only the numpy that ships with bpy. Same technique
    _save_mask_image (blender.py) uses to write PNGs without PIL."""
    loaded = [bpy.data.images.load(str(p)) for p in image_paths]
    w, h = loaded[0].size
    sheet = np.zeros((h * rows, w * cols, 4), dtype=np.float32)
    for idx, img in enumerate(loaded):
        r, c = idx // cols, idx % cols
        px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w, :] = px
        bpy.data.images.remove(img)

    out_img = bpy.data.images.new("grid_tmp", width=w * cols, height=h * rows)
    out_img.pixels = sheet.flatten()
    out_img.filepath_raw = str(out_path)
    out_img.file_format = "PNG"
    out_img.save()
    bpy.data.images.remove(out_img)


def render_pair(
    actor_inst: str, recipient_inst: str, omni_root: Path, tex_dir: Path,
    out_dir: Path, rows: int, cols: int, rng: np.random.RandomState,
    overrides: dict, rot_overrides: dict,
) -> None:
    pair_dir = out_dir / f"{actor_inst}__{recipient_inst}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    _reset_scene()
    scene = bpy.context.scene

    class _Cfg:
        n_samples = 96
        resolution = RESOLUTION
        camera_fov_deg = CAMERA_FOV_DEG

    _setup_render(scene, _Cfg())
    _setup_world()
    _setup_table(tex_dir, rng)
    camera = _ensure_camera(scene)
    camera.data.angle = math.radians(CAMERA_FOV_DEG)

    # A flipped container (upside-down/on-its-side dish, bowl, ...) ruins the
    # labeling image. Physics occasionally tips one on impact (cups especially
    # -- narrow base, off-center handle); rather than gamble on a re-roll
    # settling differently, force it back to its canonical upright pose in
    # place once settling is done (see _snap_container_upright).
    table_top = bpy.data.objects.get("TableTop")
    placed, classes = _place_pair(
        omni_root, actor_inst, recipient_inst, overrides, rot_overrides, rng)
    if table_top is not None:
        _run_physics_drop(placed, table_top)
    else:
        for obj in placed:
            _snap_to_surface(obj, surface_z=0.0)

    still_flipped = _flipped_containers(placed, classes)
    for obj in still_flipped:
        _snap_container_upright(obj)
    still_flipped = _flipped_containers(placed, classes)

    if still_flipped:
        print(f"[pair_render] WARNING: {pair_dir.name}: container still "
              f"flipped after upright-snap correction, rendering anyway "
              f"(flagged in manifest)")

    # Real post-settle mesh bounds, not a footprint-table estimate — captures
    # height and any physics-driven rotation/settling, so tall/oddly-shaped
    # objects (durian, bowls, etc.) can't get clipped at the frame edge.
    center, radius = _scene_bounds(placed)
    poses = grid_camera_poses(rows, cols, radius, CAMERA_FOV_DEG, center=tuple(center))

    width, height = RESOLUTION
    inst_by_pass_idx = {i + 1: inst for i, inst in enumerate((actor_inst, recipient_inst))}

    angle_paths = []
    angle_bboxes: list[dict[str, list[float]]] = []
    for i, pose in enumerate(poses):
        _set_camera_pose(camera, pose)
        out_path = pair_dir / f"angle_{i:02d}.png"
        scene.render.filepath = str(out_path)

        exr_path = _setup_compositor_index_pass(scene, pair_dir, f"angle_{i:02d}")
        bpy.ops.render.render(write_still=True)
        angle_paths.append(out_path)

        index_pass = _read_index_exr(exr_path, width, height)
        exr_path.unlink(missing_ok=True)
        raw_bboxes = extract_bboxes_from_pass(index_pass)
        angle_bboxes.append({
            inst_by_pass_idx[idx]: list(map(float, box))
            for idx, box in raw_bboxes.items()
            if idx in inst_by_pass_idx
        })

        print(f"[pair_render] {pair_dir.name} angle {i+1}/{len(poses)} -> {out_path.name}")

    _assemble_grid(angle_paths, rows, cols, pair_dir / "grid.png")

    manifest = {
        "actor": actor_inst,
        "recipient": recipient_inst,
        "grid": {"rows": rows, "cols": cols},
        "n_angles": len(poses),
        "resolution": list(RESOLUTION),
        "angle_files": [p.name for p in angle_paths],
        "grid_file": "grid.png",
        # Per-angle {obj_id: [x1,y1,x2,y2]} pixel bbox, same convention as
        # affdet.schemas.SceneObject.bbox. One dict per angle_files entry, same
        # order. An object absent from a given angle's dict means it had zero
        # visible pixels in that angle's object-index pass (fully occluded or
        # out of frame) -- not an error.
        "bboxes": angle_bboxes,
        # True if a container in this pair is still upside-down/on-its-side
        # even after the upright-snap correction (should be rare -- only
        # when the snap itself can't find a safe resting pose) -- the image
        # is unusable for canContain/canSupport judgment and should be
        # excluded/re-rendered, not silently trusted.
        "container_flipped": bool(still_flipped),
    }
    (pair_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = _parse_args()
    rows, cols = parse_grid(args.grid)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    omni_root = Path(args.omni_root)
    tex_dir = Path(args.tex_dir).resolve()
    overrides = load_overrides(args.overrides)
    rot_overrides = load_rotation_overrides(args.rotation_overrides)
    dropped = load_dropped_set(args.resolved)

    with open(args.pairs) as f:
        pairs = json.load(f)

    print(f"[pair_render] {len(pairs)} pairs, grid={rows}x{cols} ({rows*cols} angles/pair), "
          f"{len(dropped)} dropped instances loaded from {args.resolved}")
    done = 0
    n_dropped_skipped = 0
    for spec in pairs:
        actor_inst, recipient_inst = spec["actor"], spec["recipient"]
        pair_dir = out_dir / f"{actor_inst}__{recipient_inst}"
        # Hard guard: never render a dropped instance, regardless of what
        # pairs.json contains -- resolved_instances_v2.json is the single
        # source of truth for usability (see load_dropped_set docstring for
        # why this can't just be re-derived from curation.json ad hoc).
        if actor_inst in dropped or recipient_inst in dropped:
            which = actor_inst if actor_inst in dropped else recipient_inst
            print(f"[pair_render] skip (dropped instance {which}): "
                  f"{actor_inst}__{recipient_inst}", file=sys.stderr)
            n_dropped_skipped += 1
            continue
        if (pair_dir / "manifest.json").exists():
            print(f"[pair_render] skip (already rendered): {pair_dir.name}")
            continue
        try:
            rng = _pair_rng(args.seed, actor_inst, recipient_inst)
            render_pair(
                actor_inst, recipient_inst, omni_root, tex_dir, out_dir,
                rows, cols, rng, overrides, rot_overrides,
            )
            done += 1
        except Exception as e:
            print(f"[pair_render] WARNING: skipped {actor_inst}__{recipient_inst}: {e}",
                  file=sys.stderr)

    print(f"[pair_render] done: {done}/{len(pairs)} pairs rendered "
          f"({n_dropped_skipped} skipped for dropped instances) -> {out_dir}")


if __name__ == "__main__":
    main()
