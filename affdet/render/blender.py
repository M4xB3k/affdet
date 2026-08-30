"""Headless Blender renderer.

Called as:
    blender --background --python affdet/render/blender.py -- \
        --out data/scenes --n-scenes 100 --seed 0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bpy
from mathutils import Matrix

from affdet.render.scene_sampler import SceneSampler, TABLE_SIZE_M, TABLE_HEIGHT_M
from affdet.render.mask_export import extract_bboxes_from_pass
from affdet.schemas import Scene, SceneObject, Camera, GTAttrs


_DEFAULT_GT_TEMPLATES: dict[str, dict] = {
    "knife": {"members": ["body", "handle", "edge"], "bbox_size": [0.20, 0.03, 0.02],
              "weight_kg": 0.15, "edge_type": "serrated"},
    "spoon": {"members": ["body", "handle"], "bbox_size": [0.18, 0.03, 0.04],
              "weight_kg": 0.05, "edge_type": "none"},
    "spatula": {"members": ["body", "handle"], "bbox_size": [0.20, 0.04, 0.05],
                "weight_kg": 0.08, "edge_type": "none"},
    "mug": {"members": ["body", "handle"], "bbox_size": [0.09, 0.10, 0.09],
            "weight_kg": 0.35, "edge_type": "none"},
    "bowl": {"members": ["body"], "bbox_size": [0.16, 0.08, 0.16],
             "weight_kg": 0.30, "edge_type": "none"},
    "pitcher_base": {"members": ["body", "handle"], "bbox_size": [0.12, 0.20, 0.12],
                     "weight_kg": 0.45, "edge_type": "none"},
    "pitcher_lid": {"members": ["body"], "bbox_size": [0.12, 0.03, 0.12],
                    "weight_kg": 0.05, "edge_type": "none"},
    "sugar_box": {"members": ["body"], "bbox_size": [0.08, 0.12, 0.06],
                  "weight_kg": 0.45, "edge_type": "none"},
    "mustard_bottle": {"members": ["body"], "bbox_size": [0.06, 0.16, 0.06],
                       "weight_kg": 0.35, "edge_type": "none"},
    "plate": {"members": ["body"], "bbox_size": [0.26, 0.03, 0.26],
              "weight_kg": 0.45, "edge_type": "none"},
    "skillet": {"members": ["body", "handle"], "bbox_size": [0.28, 0.06, 0.28],
                "weight_kg": 0.80, "edge_type": "none"},
    "skillet_lid": {"members": ["body"], "bbox_size": [0.28, 0.04, 0.28],
                    "weight_kg": 0.20, "edge_type": "none"},
    "banana": {"members": ["body"], "bbox_size": [0.18, 0.04, 0.04],
               "weight_kg": 0.12, "edge_type": "none"},
    "apple": {"members": ["body"], "bbox_size": [0.08, 0.08, 0.08],
              "weight_kg": 0.20, "edge_type": "none"},
    "lemon": {"members": ["body"], "bbox_size": [0.06, 0.05, 0.06],
              "weight_kg": 0.12, "edge_type": "none"},
    "peach": {"members": ["body"], "bbox_size": [0.08, 0.07, 0.08],
              "weight_kg": 0.18, "edge_type": "none"},
    "pear": {"members": ["body"], "bbox_size": [0.08, 0.09, 0.08],
             "weight_kg": 0.20, "edge_type": "none"},
    "orange": {"members": ["body"], "bbox_size": [0.08, 0.08, 0.08],
               "weight_kg": 0.18, "edge_type": "none"},
    "plum": {"members": ["body"], "bbox_size": [0.06, 0.06, 0.06],
             "weight_kg": 0.10, "edge_type": "none"},
    "strawberry": {"members": ["body"], "bbox_size": [0.05, 0.05, 0.05],
                   "weight_kg": 0.05, "edge_type": "none"},
    "cutting_board": {"members": ["body"], "bbox_size": [0.30, 0.02, 0.20],
                      "weight_kg": 0.50, "edge_type": "none"},
    "salt_shaker": {"members": ["body"], "bbox_size": [0.05, 0.09, 0.05],
                    "weight_kg": 0.18, "edge_type": "none"},
    "pot": {"members": ["body", "handle"], "bbox_size": [0.25, 0.12, 0.25],
            "weight_kg": 1.00, "edge_type": "none"},
    "pot_lid": {"members": ["body"], "bbox_size": [0.25, 0.04, 0.25],
                "weight_kg": 0.25, "edge_type": "none"},
    "tray": {"members": ["body"], "bbox_size": [0.35, 0.03, 0.25],
             "weight_kg": 0.40, "edge_type": "none"},
}


def _default_gt(obj_class: str) -> GTAttrs:
    template = _DEFAULT_GT_TEMPLATES.get(
        obj_class,
        {"members": ["body"], "bbox_size": [0.10, 0.05, 0.05],
         "weight_kg": 0.10, "edge_type": "none"},
    )
    return GTAttrs(**template)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Blender renderer.")
    parser.add_argument("--out", required=True, help="Output directory for scenes")
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mesh-dir", default="data/meshes")
    parser.add_argument("--views-per-config", type=int, default=1)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args, _ = parser.parse_known_args(argv)
    return args


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _enable_gpu(scene: bpy.types.Scene) -> str:
    """Enable Cycles GPU rendering (OPTIX > CUDA), falling back to CPU.

    Headless Blender defaults to CPU; on the cluster this must be set
    explicitly or 100-scene jobs blow the wall clock. Returns the device
    type actually used so the caller can log it.
    """
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue  # backend not compiled into this Blender build
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if not gpus:
            continue
        for d in prefs.devices:
            d.use = d.type == backend  # GPU on, CPU off (no hybrid)
        scene.cycles.device = "GPU"
        return backend
    scene.cycles.device = "CPU"
    return "CPU"


def _setup_render(scene: bpy.types.Scene, config) -> None:
    scene.render.engine = "CYCLES"
    device = _enable_gpu(scene)
    print(f"[render] cycles device: {device}")
    scene.cycles.samples = int(config.n_samples)
    scene.render.resolution_x = int(config.resolution[0])
    scene.render.resolution_y = int(config.resolution[1])
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    if scene.view_layers:
        scene.view_layers[0].use_pass_object_index = True


def _setup_world() -> None:
    world = bpy.data.worlds.new("World") if "World" not in bpy.data.worlds else bpy.data.worlds["World"]
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.5
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # 3-point lighting: key, fill, rim
    def _add_sun(name, energy, angle_deg, elevation_deg):
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.name = name
        sun.data.energy = energy
        sun.rotation_euler = (
            math.radians(90.0 - elevation_deg),
            0.0,
            math.radians(angle_deg),
        )
        return sun

    _add_sun("KeyLight",  2.0, 45.0,  60.0)
    _add_sun("FillLight", 1.2, -135.0, 45.0)
    _add_sun("RimLight",  0.8, 180.0,  30.0)


_TABLE_TEXTURE_SETS = [
    ("plywood_diff_2k.jpg",       "plywood_nor_gl_2k.exr",       "plywood_rough_2k.exr"),
    ("wood_table_001_diff_2k.jpg","wood_table_001_nor_gl_2k.exr","wood_table_001_rough_2k.exr"),
]

_TABLE_W = 1.10   # slightly larger than TABLE_SIZE_M (1.00) for comfortable margin
_TABLE_D = 0.85
_TABLE_THICKNESS = 0.04
_LEG_W = 0.04
_LEG_H = 0.72


def _make_wood_material(name: str, tex_dir: Path, diff: str, nor: str, rough: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    out = nt.nodes["Material Output"]

    def _img_node(path: Path, non_color: bool = False):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(path))
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        return n

    diff_node  = _img_node(tex_dir / diff)
    nor_node   = _img_node(tex_dir / nor,   non_color=True)
    rough_node = _img_node(tex_dir / rough, non_color=True)

    map_node = nt.nodes.new("ShaderNodeMapping")
    uv_node  = nt.nodes.new("ShaderNodeTexCoord")
    # tile the texture ~3× so wood grain looks right at tabletop scale
    map_node.inputs["Scale"].default_value = (3.0, 3.0, 3.0)
    nt.links.new(uv_node.outputs["UV"], map_node.inputs["Vector"])
    for img in (diff_node, nor_node, rough_node):
        nt.links.new(map_node.outputs["Vector"], img.inputs["Vector"])

    nor_map = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nor_node.outputs["Color"], nor_map.inputs["Color"])

    nt.links.new(diff_node.outputs["Color"],  bsdf.inputs["Base Color"])
    nt.links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(nor_map.outputs["Normal"],   bsdf.inputs["Normal"])
    return mat


def _setup_table(tex_dir: Path, rng: np.random.RandomState) -> None:
    """Build a simple table: flat top + 4 legs, all geometry below/at z=0.

    Tabletop top face sits exactly at z=0 so objects placed at z=0 rest on it.
    Legs extend downward; a thin floor plane catches the legs at the bottom.
    """
    diff, nor, rough = _TABLE_TEXTURE_SETS[int(rng.randint(len(_TABLE_TEXTURE_SETS)))]
    mat = _make_wood_material("TableMat", tex_dir, diff, nor, rough)

    # Tabletop: top face at z=0, thickness extends downward
    top_z = -_TABLE_THICKNESS / 2
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, 0.0, top_z),
    )
    top = bpy.context.active_object
    top.name = "TableTop"
    top.scale = (_TABLE_W, _TABLE_D, _TABLE_THICKNESS)
    bpy.ops.object.transform_apply(scale=True)
    top.data.materials.append(mat)

    # 4 legs, positioned at corners
    leg_z = -_TABLE_THICKNESS - _LEG_H / 2
    for (sx, sy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        lx = sx * (_TABLE_W / 2 - _LEG_W / 2)
        ly = sy * (_TABLE_D / 2 - _LEG_W / 2)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(lx, ly, leg_z))
        leg = bpy.context.active_object
        leg.name = f"TableLeg_{sx}_{sy}"
        leg.scale = (_LEG_W, _LEG_W, _LEG_H)
        bpy.ops.object.transform_apply(scale=True)
        leg.data.materials.append(mat)

    # Floor plane at the bottom of the legs
    floor_z = -_TABLE_THICKNESS - _LEG_H
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, floor_z))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor_mat = bpy.data.materials.new(name="FloorMat")
    floor_mat.use_nodes = True
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(floor_mat)


def _ensure_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, -0.5, 0.5))
    camera = bpy.context.active_object
    scene.camera = camera
    return camera


def _import_mesh(mesh_path: Path, obj_name: str,
                 target_size: tuple[float, float, float] | None = None) -> bpy.types.Object:
    """Import an OBJ file using Blender 5.x native importer (bpy.ops.wm.obj_import).

    Joins all sub-meshes (important for multi-part objects like banana) and
    scales the result so its longest axis matches the largest value in target_size.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    names_before = {o.name for o in bpy.data.objects}
    result = bpy.ops.wm.obj_import(filepath=str(mesh_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"wm.obj_import failed for {mesh_path}: {result}")
    new_names = {o.name for o in bpy.data.objects} - names_before
    if not new_names:
        raise RuntimeError(f"No objects added after importing {mesh_path}")

    # Select all imported objects and join them into one — this keeps multi-part
    # meshes (banana peel segments, etc.) intact instead of discarding sub-parts.
    bpy.ops.object.select_all(action="DESELECT")
    mesh_objs = [bpy.data.objects[n] for n in new_names
                 if bpy.data.objects[n].type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object

    if not obj.data or len(obj.data.polygons) == 0:
        raise RuntimeError(f"Imported mesh {mesh_path} has 0 faces")
    obj.name = obj_name

    # YCB meshes are already in real-world metres — no scaling needed.

    _apply_ycb_texture(obj, mesh_path.parent)
    return obj


def _apply_ycb_texture(obj: bpy.types.Object, mesh_dir: Path) -> None:
    """Assign the YCB texture image to the object.

    YCB OBJ files reference an MTL but omit usemtl, so Blender loads no
    material. We manually create a principled BSDF material with the texture.
    """
    tex_path = mesh_dir / "texture_map.png"
    if not tex_path.exists():
        tex_path = mesh_dir / "textured.png"
    if not tex_path.exists():
        # fallback: any *_bcolor.png or *_diffuse.png (e.g. cutting board)
        candidates = sorted(mesh_dir.glob("*_bcolor.png")) + sorted(mesh_dir.glob("*_diffuse.png"))
        if candidates:
            tex_path = candidates[0]
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


def _setup_compositor_index_pass(
    scene: bpy.types.Scene, out_dir: Path, scene_id: str
) -> Path:
    """Wire compositor to write the Object Index pass as an EXR.

    Returns the path where the EXR will be written (Blender writes it at
    render time — the path must be polled after bpy.ops.render.render).
    """
    old = scene.compositing_node_group
    node_group = bpy.data.node_groups.new("Compositor", type="CompositorNodeTree")
    scene.compositing_node_group = node_group
    if old is not None and old.name in bpy.data.node_groups:
        bpy.data.node_groups.remove(old)
    tree = node_group

    rl = tree.nodes.new("CompositorNodeRLayers")

    # File output node — OPEN_EXR_MULTILAYER is the only EXR option in 5.x
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.directory = str(out_dir) + "/"
    fo.file_name = f"{scene_id}_idx"
    fo.format.file_format = "OPEN_EXR_MULTILAYER"
    fo.format.color_depth = "32"
    item = fo.file_output_items.new(name="IndexOB", socket_type="FLOAT")
    item.override_node_format = False
    tree.links.new(rl.outputs["Object Index"], fo.inputs["IndexOB"])

    go = tree.nodes.new("NodeGroupOutput")
    node_group.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    tree.links.new(rl.outputs["Image"], go.inputs["Image"])

    return out_dir / f"{scene_id}_idx.exr"


def _read_index_exr(exr_path: Path, width: int, height: int) -> np.ndarray:
    """Read object index pass from EXR written by Blender compositor.

    Returns int32 array of shape (H, W) where value == obj.pass_index.
    Falls back to zeros if the file is missing or unreadable.
    """
    if not exr_path.exists():
        return np.zeros((height, width), dtype=np.int32)
    import OpenEXR
    import Imath
    exr = OpenEXR.InputFile(str(exr_path))
    try:
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())
        # Blender names the channel "IndexOB.V" for a FLOAT scalar
        ch_name = next((c for c in channels if "Index" in c or "index" in c), channels[0])
        raw = exr.channel(ch_name, Imath.PixelType(Imath.PixelType.FLOAT))
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        return np.rint(arr).astype(np.int32)
    except Exception as e:
        print(f"[blender] WARNING: _read_index_exr failed for {exr_path}: {e}", file=sys.stderr)
        return np.zeros((height, width), dtype=np.int32)
    finally:
        exr.close()


def _set_camera_pose(camera: bpy.types.Object, world_to_camera: np.ndarray) -> None:
    cam_to_world = np.linalg.inv(world_to_camera)
    camera.matrix_world = Matrix(cam_to_world.tolist())


def _save_mask_image(mask: np.ndarray, out_path: Path) -> None:
    height, width = mask.shape
    img = bpy.data.images.new("mask_tmp", width=width, height=height)
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    # Blender pixels are bottom-up; flip vertically before writing
    flipped = np.flipud(mask)
    rgba[:, :, 0:3] = flipped[:, :, None].astype(np.float32)
    rgba[:, :, 3] = 1.0
    img.pixels = rgba.flatten()
    img.filepath_raw = str(out_path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _snap_to_surface(obj: bpy.types.Object, surface_z: float = 0.0) -> None:
    """Translate obj so its lowest vertex sits exactly on surface_z."""
    bpy.context.view_layer.update()
    min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    obj.location.z += surface_z - min_z


def setup_scene(config, tex_dir: Path, rng: np.random.RandomState) -> tuple[bpy.types.Scene, bpy.types.Object, list[dict]]:
    _reset_scene()
    scene = bpy.context.scene
    _setup_render(scene, config)
    _setup_world()
    _setup_table(tex_dir, rng)
    camera = _ensure_camera(scene)
    object_infos: list[dict] = []
    placed_objects: dict[str, bpy.types.Object] = {}
    # Some OBJ meshes are exported standing upright — rotate them flat before
    # applying the sampler's placement rotation (which only spins around z).
    _MESH_PRE_ROTATION: dict[str, tuple[float, float, float]] = {
        "cutting_board": (math.pi / 2, 0.0, 0.0),  # exported standing on edge
    }
    for i, placement in enumerate(config.objects):
        obj_id = f"{placement.obj_class}_{i}"
        obj = _import_mesh(Path(placement.mesh_path), obj_id)
        obj.location = placement.position
        pre = _MESH_PRE_ROTATION.get(placement.obj_class, (0.0, 0.0, 0.0))
        rx = pre[0] + placement.rotation_euler[0]
        ry = pre[1] + placement.rotation_euler[1]
        rz = pre[2] + placement.rotation_euler[2]
        obj.rotation_euler = (rx, ry, rz)
        _snap_to_surface(obj, surface_z=0.0)
        obj.pass_index = i + 1
        placed_objects[placement.obj_class] = obj
        object_infos.append({
            "obj_id": obj_id,
            "obj_class": placement.obj_class,
            "pass_index": i + 1,
        })

    return scene, camera, object_infos


def render_scene(scene_id: str, config, camera_pose: np.ndarray,
                 out_dir: Path, scene: bpy.types.Scene,
                 camera: bpy.types.Object, object_infos: list[dict]) -> None:
    _set_camera_pose(camera, camera_pose)
    camera.data.angle = math.radians(config.camera_fov_deg)

    # Each scene gets its own subfolder: data/scenes/scene_0001/
    scene_dir = out_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    exr_path = _setup_compositor_index_pass(scene, scene_dir, scene_id)

    png_path = scene_dir / f"{scene_id}.png"
    scene.render.filepath = str(png_path)
    bpy.ops.render.render(write_still=True)

    width = int(config.resolution[0])
    height = int(config.resolution[1])
    index_pass = _read_index_exr(exr_path, width, height)
    bboxes = extract_bboxes_from_pass(index_pass)

    # Delete index EXR — masks already extracted, no need to keep the file
    exr_path.unlink(missing_ok=True)

    mask_paths: dict[str, str] = {}
    for info in object_infos:
        obj_id = info["obj_id"]
        obj_idx = info["pass_index"]
        mask = (index_pass == obj_idx).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask_path = scene_dir / f"{obj_id}_mask.png"
        _save_mask_image(mask, mask_path)
        mask_paths[obj_id] = mask_path.name

    objects: list[SceneObject] = []
    for info in object_infos:
        obj_id = info["obj_id"]
        obj_idx = info["pass_index"]
        if obj_idx in bboxes:
            bbox = list(map(float, bboxes[obj_idx]))
        else:
            bbox = [0.0, 0.0, float(width - 1), float(height - 1)]

        objects.append(SceneObject(
            obj_id=obj_id,
            obj_class=info["obj_class"],
            bbox=bbox,
            mask_path=mask_paths.get(obj_id),
            gt_attrs=_default_gt(info["obj_class"]),
        ))

    fov_rad = math.radians(config.camera_fov_deg)
    fx = 0.5 * width / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    scene_obj = Scene(
        scene_id=scene_id,
        image_path=png_path.name,
        image_size=[width, height],
        source="blender",
        objects=objects,
        camera=Camera(
            pose=camera_pose.tolist(),
            intrinsics=intrinsics,
        ),
    )
    (scene_dir / f"{scene_id}.json").write_text(scene_obj.model_dump_json(indent=2))


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = Path(args.mesh_dir).parent / "table_textures"
    rng = np.random.RandomState(args.seed)
    sampler = SceneSampler(
        seed=args.seed,
        mesh_dir=args.mesh_dir,
        allow_missing_assets=False,
    )

    # Find highest existing scene number to avoid overwriting
    existing = [
        int(p.name.split("_")[1])
        for p in out_dir.iterdir()
        if p.is_dir() and p.name.startswith("scene_") and p.name.split("_")[1].isdigit()
    ]
    start_idx = max(existing) if existing else 0

    scene_count = 0
    while scene_count < args.n_scenes:
        config = sampler.sample()
        scene, camera, object_infos = setup_scene(config, tex_dir, rng)
        n_views = max(1, int(args.views_per_config))
        for v in range(n_views):
            if scene_count >= args.n_scenes:
                break
            camera_pose = config.camera_pose if v == 0 else sampler.sample_camera_pose()
            scene_id = f"scene_{start_idx + scene_count + 1:04d}"
            render_scene(scene_id, config, camera_pose, out_dir,
                         scene, camera, object_infos)
            scene_count += 1


if __name__ == "__main__":
    main()
