"""Headless Blender renderer for OmniObject3D scenes.

Called as:
    blender --background --python affdet/render/omni_blender.py -- \
        --labels data/labels/omni_vlm_raw.jsonl \
        --omni-root /path/to/omni/raw_scans \
        --out data/scenes/omni \
        --n-scenes 500 \
        --seed 0

Each scene contains one actor + one food from the label JSONL.
The pre-computed VLM score is written into the scene JSON alongside
the instance IDs so the extract/model stages can look up labels directly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bpy
from mathutils import Matrix

from affdet.render.blender import (
    _reset_scene, _setup_render, _setup_world, _setup_table,
    _ensure_camera, _snap_to_surface, _set_camera_pose,
    _save_mask_image, _setup_compositor_index_pass, _read_index_exr,
    _TABLE_TEXTURE_SETS,
)
from affdet.render.mask_export import extract_bboxes_from_pass
from affdet.render.omni_sampler import OmniSceneSampler, OmniSceneConfig
from affdet.render.omni_scale import (
    apply_calibrated_scale, orient_upright, bake_import_rotation, load_overrides,
    load_rotation_overrides, apply_rotation_override,
    is_container_not_upright,
)
from affdet.render.omni_sampler import _CONTAINERS
from affdet.schemas import Scene, SceneObject, Camera

# An object must own at least this fraction of the frame to count as "seen" by
# the model. Below it (occluded or too tiny) a pair involving it is unlearnable.
# ~0.05% of a 960x720 frame ≈ 350px (~19x19 blob).
MIN_VISIBLE_FRAC = 0.0005
# How many times to re-place + re-render a scene whose target objects are
# occluded before giving up and just not counting that pair.
MAX_OCCLUSION_RETRIES = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curation", required=True, help="Path to curation.json")
    parser.add_argument("--omni-root", required=True, help="Path to raw_scans root dir")
    parser.add_argument("--out", required=True, help="Output directory for scenes")
    parser.add_argument("--tex-dir", default="data/table_textures",
                        help="Table texture directory")
    parser.add_argument("--n-scenes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overrides", default="scripts/omni/scale_overrides.json",
                        help="Per-instance scale override JSON (calibration)")
    parser.add_argument("--rotation-overrides",
                        default="scripts/omni/rotation_overrides.json",
                        help="Per-instance rotation-correction JSON (tipped scans)")
    parser.add_argument("--labels", default="data/omni/labels/omni_vlm_raw.jsonl",
                        help="VLM label JSONL for coverage-aware sampling")
    parser.add_argument("--no-coverage", action="store_true",
                        help="Use plain round-robin templates instead of coverage-aware sampling")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args, _ = parser.parse_known_args(argv)
    return args


def _import_omni_mesh(
    mesh_path: Path, obj_name: str, obj_class: str, overrides: dict[str, float],
    rot_overrides: dict[str, list[float]] | None = None,
) -> bpy.types.Object:
    """Import an OmniObject3D OBJ and apply the shared scale calibration.

    Scale/orientation comes from affdet.render.omni_scale (the same calibration
    the raw_scale_check diagnostic uses), so rendered sizes match what was tuned
    by eye — including per-instance size variation. NOT normalize-to-target."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    names_before = {o.name for o in bpy.data.objects}
    result = bpy.ops.wm.obj_import(filepath=str(mesh_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"wm.obj_import failed for {mesh_path}: {result}")

    new_names = {o.name for o in bpy.data.objects} - names_before
    mesh_objs = [bpy.data.objects[n] for n in new_names
                 if bpy.data.objects[n].type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object
    obj.name = obj_name

    # Shared calibration: bake the OBJ importer's coord-conversion rotation,
    # apply GLOBAL_SCALE × per-category/per-instance multiplier (preserves real
    # per-instance size variation), then stand elongated tools upright.
    bake_import_rotation(obj)
    # Per-instance rotation correction for tipped/rotated scans (e.g. teapots
    # scanned lying on their side) — applied BEFORE scale/orient so the object
    # starts upright, then the normal pipeline proceeds.
    apply_rotation_override(obj, obj_name, rot_overrides or {})
    apply_calibrated_scale(obj, obj_class, obj_name, overrides)
    orient_upright(obj, obj_class)
    # orient_container_opening_up used to run here to "fix" containers that
    # looked flipped, but its detector (container_first_hit_z_frac) samples
    # only 5 points near the XY bbox centre -- for a container whose true
    # opening is off-centre (a handle skewing the bbox, or a lidded/spouted
    # design with no wide-open top) every sample can land on the rim/wall
    # instead, misjudging an ALREADY-upright container as flipped and
    # flipping it into a genuinely bad pose before physics even runs.
    # Verified 2026-07-12: every OmniObject3D container scan checked
    # (cup_009/010/015/025, red_wine_glass_004) imports correctly oriented
    # natively -- this "fix" was the thing breaking them, not fixing them.
    # Containers now import as-scanned and get placed/dropped like anything
    # else; is_container_not_upright still runs post-drop to flag/downgrade
    # a container that physics itself topples.

    _apply_omni_texture(obj, mesh_path.parent)
    return obj


def _lowest_world_z(obj: bpy.types.Object) -> float:
    """World-space z of the object's lowest vertex."""
    return min((obj.matrix_world @ v.co).z for v in obj.data.vertices)


def _snap_to_table(obj: bpy.types.Object, surface_z: float = 0.0) -> None:
    """Shift `obj` vertically so its lowest vertex rests exactly on surface_z.
    Corrects both floating (settled short of the table) and interpenetration
    (sunk into it) after the sim."""
    obj.location.z += surface_z - _lowest_world_z(obj)
    bpy.context.view_layer.update()


def _run_physics_drop(
    objects: list[bpy.types.Object],
    table_top: bpy.types.Object,
    n_frames: int = 140,
) -> None:
    """Drop objects onto the table using Blender rigid body simulation.

    Objects are first lowered to just above the table (small drop) so they
    settle in place rather than bouncing/rolling across it, then the sim runs,
    then every object is snapped so its lowest vertex rests exactly on the
    table (fixes residual float/interpenetration). Frames are stepped manually
    (more reliable than ptcache.bake_all in headless mode).
    """
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.frame_set(1)

    # Pre-drop: lower each object so its bottom sits ~2cm above the table.
    # A short fall means minimal bounce, so objects settle near their sampled
    # XY instead of rolling off the edge (fixes vanished pineapple) and
    # elongated items (corn, cucumber) come to rest flat quickly.
    bpy.context.view_layer.update()
    for obj in objects:
        obj.location.z += 0.02 - _lowest_world_z(obj)
    bpy.context.view_layer.update()

    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    rbw.enabled = True
    rbw.substeps_per_frame = 10
    rbw.solver_iterations = 20

    # Table top — passive collider
    bpy.context.view_layer.objects.active = table_top
    table_top.select_set(True)
    bpy.ops.rigidbody.object_add()
    table_top.rigid_body.type = "PASSIVE"
    table_top.rigid_body.collision_shape = "BOX"
    table_top.select_set(False)

    # Objects — active rigid bodies. High friction + damping so they settle
    # (and stop rolling) fast; low restitution so they don't bounce.
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = "ACTIVE"
        obj.rigid_body.collision_shape = "CONVEX_HULL"
        obj.rigid_body.mass = 0.1
        obj.rigid_body.restitution = 0.0
        obj.rigid_body.friction = 1.0
        obj.rigid_body.linear_damping = 0.9
        obj.rigid_body.angular_damping = 0.9
        obj.select_set(False)

    # Step frames manually — reliable in headless Blender
    for frame in range(1, n_frames + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

    # Apply visual transform to bake settled pose into object location/rotation
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.visual_transform_apply()

    # Remove rigid body world before rendering
    bpy.ops.rigidbody.world_remove()

    # Rest every object exactly on the table (fixes residual float or sink).
    bpy.context.view_layer.update()
    for obj in objects:
        _snap_to_table(obj, surface_z=0.0)


def _apply_omni_texture(obj: bpy.types.Object, scan_dir: Path) -> None:
    tex_path = scan_dir / "Scan.jpg"
    if not tex_path.exists():
        return
    mat = bpy.data.materials.new(name=f"{obj.name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(str(tex_path))
    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def render_omni_scene(
    scene_id: str,
    config: OmniSceneConfig,
    out_dir: Path,
    tex_dir: Path,
    rng: np.random.RandomState,
    overrides: dict[str, float] | None = None,
    realised: list | None = None,
    rot_overrides: dict[str, list[float]] | None = None,
) -> tuple[list, dict]:
    """Render one scene. Returns (realised, visible_by_id):
      - realised: (possibly orientation-corrected) list of (affordance, bucket)
        pairs — a positive canContain/canPourInto/canSupport target is downgraded
        to negative when its container is not upright (one-way; never promoted).
      - visible_by_id: {instance_id: bool} whether each object is visible enough
        (>= MIN_VISIBLE_FRAC of the frame) in the final camera to be detectable.
    """
    overrides = overrides or {}
    rot_overrides = rot_overrides or {}
    realised = list(realised or [])
    _reset_scene()
    scene = bpy.context.scene

    # Build a minimal SceneConfig-like object for _setup_render
    class _Cfg:
        n_samples = config.n_samples
        resolution = config.resolution
        camera_fov_deg = config.camera_fov_deg

    _setup_render(scene, _Cfg())
    _setup_world()
    _setup_table(tex_dir, rng)
    camera = _ensure_camera(scene)

    placed_objs: list[bpy.types.Object] = []
    object_infos: list[dict] = []
    for i, placement in enumerate(config.objects):
        obj = _import_omni_mesh(Path(placement.mesh_path), placement.instance_id, placement.obj_class, overrides, rot_overrides)
        obj.location = placement.position
        obj.rotation_euler = placement.rotation_euler
        obj.pass_index = i + 1
        placed_objs.append(obj)
        object_infos.append({
            "obj_id": placement.instance_id,
            "obj_class": placement.obj_class,
            "pass_index": i + 1,
        })

    # Physics drop — let objects settle naturally on the table
    table_top = bpy.data.objects.get("TableTop")
    if table_top is not None:
        _run_physics_drop(placed_objs, table_top)
    else:
        for obj in placed_objs:
            _snap_to_surface(obj, surface_z=0.0)

    # After settling, check each container's orientation. A container that is
    # not opening-up (upside-down OR on its side) cannot contain / be poured
    # into, so its canContain/canPourInto label must become negative downstream.
    any_container_flipped = False
    for obj, info in zip(placed_objs, object_infos):
        if info["obj_class"] in _CONTAINERS:
            flipped = bool(is_container_not_upright(obj))
            info["not_upright"] = flipped
            any_container_flipped = any_container_flipped or flipped
        else:
            info["not_upright"] = False

    # One-way orientation correction: a positive container affordance
    # (canContain/canPourInto/canSupport) becomes negative when the container is
    # not opening-up. Negatives are never promoted to positive.
    _ORIENT_AFFS = {"canContain", "canPourInto", "canSupport"}
    if any_container_flipped:
        realised = [
            (aff, "neg") if (aff in _ORIENT_AFFS and bucket == "pos") else (aff, bucket)
            for aff, bucket in realised
        ]

    _set_camera_pose(camera, config.camera_pose)
    camera.data.angle = math.radians(config.camera_fov_deg)

    scene_dir = out_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    exr_path = _setup_compositor_index_pass(scene, scene_dir, scene_id)
    png_path = scene_dir / f"{scene_id}.png"
    scene.render.filepath = str(png_path)
    bpy.ops.render.render(write_still=True)

    width, height = int(config.resolution[0]), int(config.resolution[1])
    index_pass = _read_index_exr(exr_path, width, height)
    bboxes = extract_bboxes_from_pass(index_pass)
    exr_path.unlink(missing_ok=True)

    # Visible-pixel count per object from the POST-occlusion index pass: an
    # object hidden behind others simply owns fewer/zero pixels. This is exactly
    # what the model can see, so it drives occlusion handling below.
    total_px = width * height
    mask_paths: dict[str, str] = {}
    for info in object_infos:
        mask = (index_pass == info["pass_index"]).astype(np.uint8)
        visible_px = int(mask.sum())
        info["visible_px"] = visible_px
        info["visible"] = visible_px >= MIN_VISIBLE_FRAC * total_px
        if visible_px == 0:
            continue
        mask_path = scene_dir / f"{info['obj_id']}_mask.png"
        _save_mask_image(mask, mask_path)
        mask_paths[info["obj_id"]] = mask_path.name

    objects: list[SceneObject] = []
    for info in object_infos:
        idx = info["pass_index"]
        bbox = list(map(float, bboxes[idx])) if idx in bboxes else [0.0, 0.0, float(width - 1), float(height - 1)]
        objects.append(SceneObject(
            obj_id=info["obj_id"],
            obj_class=info["obj_class"],
            bbox=bbox,
            mask_path=mask_paths.get(info["obj_id"]),
        ))

    fov_rad = math.radians(config.camera_fov_deg)
    fx = 0.5 * width / math.tan(fov_rad / 2.0)
    intrinsics = [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]]

    scene_obj = Scene(
        scene_id=scene_id,
        image_path=png_path.name,
        image_size=[width, height],
        source="blender",
        objects=objects,
        camera=Camera(pose=config.camera_pose.tolist(), intrinsics=intrinsics),
    )
    scene_data = scene_obj.model_dump()
    scene_data["template"] = config.template_name
    # instance_ids list so emit_labels can join against the label JSONL
    scene_data["instance_ids"] = [p.instance_id for p in config.objects]

    # Attach per-object not-upright flag + visibility (from the final camera).
    not_upright_by_id = {info["obj_id"]: info.get("not_upright", False) for info in object_infos}
    visible_by_id = {info["obj_id"]: info.get("visible", True) for info in object_infos}
    visible_px_by_id = {info["obj_id"]: info.get("visible_px", 0) for info in object_infos}
    for obj_dict in scene_data["objects"]:
        oid = obj_dict["obj_id"]
        obj_dict["not_upright"] = not_upright_by_id.get(oid, False)
        obj_dict["visible"] = visible_by_id.get(oid, True)
        obj_dict["visible_px"] = visible_px_by_id.get(oid, 0)
    scene_data["not_upright_containers"] = sorted(
        oid for oid, flipped in not_upright_by_id.items() if flipped
    )
    # Record the orientation-corrected affordance verdict(s) this scene realises.
    scene_data["realised_affordances"] = [[aff, bucket] for aff, bucket in realised]
    (scene_dir / f"{scene_id}.json").write_text(json.dumps(scene_data, indent=2))
    flipped_note = f" [not-upright->neg: {scene_data['not_upright_containers']}]" if scene_data["not_upright_containers"] else ""
    obj_summary = ", ".join(p.instance_id for p in config.objects)
    print(f"[render] {scene_id} [{config.template_name}]: {obj_summary}{flipped_note}")
    return realised, visible_by_id


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = Path(args.tex_dir).resolve()
    rng = np.random.RandomState(args.seed)
    overrides = load_overrides(args.overrides)
    rot_overrides = load_rotation_overrides(args.rotation_overrides)

    sampler = OmniSceneSampler(
        curation_path=args.curation,
        omni_root=args.omni_root,
        seed=args.seed,
    )

    existing = [
        int(p.name.split("_")[1])
        for p in out_dir.iterdir()
        if p.is_dir() and p.name.startswith("scene_") and p.name.split("_")[1].isdigit()
    ]
    start_idx = max(existing, default=0)

    # Coverage-aware sampling (default): balance per-affordance pos/neg pairs +
    # instance coverage. Falls back to plain round-robin templates with --no-coverage.
    tracker = index = None
    if not args.no_coverage:
        from affdet.render.omni_coverage import AffordancePairIndex, CoverageTracker
        index = AffordancePairIndex.from_jsonl(args.labels)
        tracker = CoverageTracker(index=index, n_scenes=args.n_scenes)

    rendered = 0
    empty_streak = 0
    while rendered < args.n_scenes:
        realised: list = []
        required: set = set()
        if tracker is not None:
            config, realised, required = sampler.sample_covered(tracker, index)
            if config is None:
                # All coverage targets met — top up remainder with round-robin
                # so we still hit the scene budget.
                config = sampler.sample()
        else:
            config = sampler.sample()

        if config is None:
            empty_streak += 1
            if empty_streak > 50:
                print(f"[render] sampler exhausted after {rendered} scenes", file=sys.stderr)
                break
            continue
        empty_streak = 0

        scene_id = f"scene_{start_idx + rendered + 1:04d}"
        try:
            # Render; if a required object (the pair's actor/food) is occluded in
            # the final camera, re-place all objects and re-render up to
            # MAX_OCCLUSION_RETRIES times — a pair the model can't see is
            # unlearnable. render returns orientation-corrected pairs + per-object
            # visibility.
            corrected, visible = render_omni_scene(
                scene_id, config, out_dir, tex_dir, rng, overrides, realised,
                rot_overrides)
            attempts = 0
            while required and not all(visible.get(oid, False) for oid in required) \
                    and attempts < MAX_OCCLUSION_RETRIES:
                attempts += 1
                sampler._assign_positions(config.objects)  # fresh XY, same objects
                corrected, visible = render_omni_scene(
                    scene_id, config, out_dir, tex_dir, rng, overrides, realised,
                rot_overrides)
            target_visible = not required or all(visible.get(oid, False) for oid in required)

            if tracker is not None:
                if target_visible:
                    for aff, bucket in corrected:
                        tracker.record_pair(aff, bucket)
                else:
                    # Still occluded after retries — keep the rendered scene but
                    # don't credit the unlearnable pair toward coverage.
                    print(f"[render] {scene_id}: target occluded after "
                          f"{attempts} retries, pair not counted", file=sys.stderr)
            rendered += 1
        except Exception as e:
            print(f"[render] WARNING: skipped {scene_id}: {e}", file=sys.stderr)

    print(f"[render] done: {rendered} scenes written to {out_dir}")
    if tracker is not None:
        print(tracker.report())


if __name__ == "__main__":
    main()
