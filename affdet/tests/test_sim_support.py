"""Calibration of the MuJoCo face test on primitives with known answers (slow: ~1 min)."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("mujoco")

from affdet.sim.mujoco_support import Body, face_holds


def _on_table(m):
    m.apply_translation([0, 0, -m.bounds[0, 2]])
    return m


@pytest.mark.parametrize("name,face,expected", [
    ("box", "front", True),
    ("cyl64", "front", False),   # round side: no restoring torque
    ("hex", "left", True),       # flat facet
    ("domed", "on", False),      # domed cap upside down
    ("tall_cyl", "on", True),
])
def test_face_holds_primitives(name, face, expected):
    shapes = {
        "box": lambda: trimesh.creation.box([0.05, 0.05, 0.10]),
        "cyl64": lambda: trimesh.creation.cylinder(radius=0.025, height=0.09, sections=64),
        "hex": lambda: trimesh.creation.cylinder(radius=0.025, height=0.09, sections=6).apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 6, [0, 0, 1])),
        "domed": lambda: trimesh.util.concatenate([
            trimesh.creation.cylinder(radius=0.025, height=0.07, sections=64),
            trimesh.creation.icosphere(subdivisions=4, radius=0.025).apply_translation([0, 0, 0.035])]),
        "tall_cyl": lambda: trimesh.creation.cylinder(radius=0.0225, height=0.18, sections=64),
    }
    mesh = _on_table(shapes[name]())
    assert face_holds(Body(name, [mesh.convex_hull], 0.2), face)["holds"] == expected


def _mesh(kind):
    shapes = {
        "box": lambda: trimesh.creation.box([0.05, 0.05, 0.10]),
        "cyl64": lambda: trimesh.creation.cylinder(radius=0.025, height=0.09, sections=64),
        "cone": lambda: trimesh.creation.cone(radius=0.03, height=0.08, sections=64),
    }
    return _on_table(shapes[kind]())


@pytest.mark.parametrize("kind,face,expected", [
    *[("box", f, True) for f in ("on", "under", "front", "back", "left", "right")],
    ("cyl64", "on", True), ("cyl64", "under", True), ("cyl64", "left", False),
    ("cone", "under", True), ("cone", "on", False),  # a cone stands on its base, never on its tip
])
def test_face_holds_more_primitives(kind, face, expected):
    assert face_holds(Body(kind, [_mesh(kind).convex_hull], 0.2), face)["holds"] == expected


def test_face_holds_uses_the_static_disturbance():
    """The simulated tilt equals the static labeller's tip-angle threshold, so the two tests ask the same question."""
    from affdet.geometry.face_labels import Params
    from affdet.sim.mujoco_support import DISTURBANCE_DEG
    assert DISTURBANCE_DEG == Params().min_tip_deg


def test_grasp_swing_grows_with_distance_from_the_centre_of_mass():
    """A uniform 20 x 3 x 2 cm bar (1.2 kg) pinched across its width: a grasp at the COM lifts it flat, and the
    further from the COM the pinch is, the more it pivots (0, 16, 38 deg at 0, 4, 9.5 cm)."""
    from affdet.sim.grasp_map import grasp_test
    bar = Body("bar", [_on_table(trimesh.creation.box([0.2, 0.03, 0.02]))], 1.2)
    res = [grasp_test(bar, d, 0.0, 0.0) for d in (0.0, 0.04, 0.095)]
    assert res[0]["success"] and res[0]["swing_deg"] < 3
    assert res[1]["swing_deg"] > res[0]["swing_deg"] + 5
    assert not res[2]["success"]


def test_analytic_grasp_model_matches_the_simulated_bar():
    """The analytic torque model predicts the same cut-off the simulation shows on the uniform bar: level up to
    tau_max / (m g) = 5.1 cm from the COM, not beyond."""
    from affdet.sim.grasp_analytic import grasp_analytic
    bar = _on_table(trimesh.creation.box([0.2, 0.03, 0.02]))
    r = grasp_analytic([bar], 1.2, np.array([0.0, 0.04, 0.06, 0.095]), np.array([0.0]), [0, 90])
    assert abs(r["d_max"] - 0.051) < 0.002
    assert list(r["level"][0, :, 0]) == [True, True, False, False]
    assert not r["fits"][0, :, 1].any()  # closing along the 20 cm length does not fit the 8 cm opening


def _walled_slab():
    """A low wide slab (top 1.3 cm) with a 5 cm tall, 1 cm thick wall on its +x edge: a toy pan rim. Two convex parts."""
    slab = trimesh.creation.box([0.4, 0.2, 0.013]); slab.apply_translation([0, 0, 0.0065])
    wall = trimesh.creation.box([0.01, 0.2, 0.05]); wall.apply_translation([0.195, 0, 0.025])
    return slab, wall


def test_part_masses_follow_the_surface_for_shells_and_the_volume_for_solids():
    from affdet.sim.objects import part_masses
    wall = trimesh.creation.box([0.002, 0.1, 0.1])
    cube = trimesh.creation.box([0.03, 0.03, 0.03]); cube.apply_translation([0.1, 0, 0])
    both = trimesh.util.concatenate([wall, cube])
    shell = part_masses(both, [wall, cube], 1.0, "shell")
    solid = part_masses(both, [wall, cube], 1.0, "solid")
    assert abs(sum(shell) - 1.0) < 1e-9 and abs(sum(solid) - 1.0) < 1e-9
    assert shell[0] > 0.7 > 0.5 > solid[0]  # the thin wall has area but no volume


def test_grasp_pads_start_clear_of_a_floor_next_to_the_wall():
    """Pinching a toy pan's rim: the inner pad lands over the slab, not inside it, so the grasp lifts the object."""
    from affdet.sim.grasp_map import grasp_test
    slab, wall = _walled_slab()
    body = Body("pan", [slab, wall], 0.3)
    r = grasp_test(body, 0.195, 0.0, 90.0)
    assert r["on_object"] and not r.get("blocked") and r["held"]


def test_grasp_is_blocked_when_a_pad_would_start_inside_a_taller_neighbour():
    """A 2 cm ledge 4 cm from a 5 cm wall: the far pad of a pinch on the ledge would land inside the wall."""
    from affdet.sim.grasp_map import grasp_test
    ledge = trimesh.creation.box([0.12, 0.1, 0.02]); ledge.apply_translation([-0.05, 0, 0.01])
    wall = trimesh.creation.box([0.01, 0.1, 0.05]); wall.apply_translation([0.03, 0, 0.025])
    r = grasp_test(Body("ledge", [ledge, wall], 0.3), -0.015, 0.0, 90.0)
    assert r["blocked"] and not r["success"]


def test_analytic_grasp_agrees_that_the_neighbouring_wall_blocks_the_pinch_and_rim_pitches():
    from affdet.sim.grasp_analytic import grasp_analytic
    ledge = trimesh.creation.box([0.12, 0.1, 0.02]); ledge.apply_translation([-0.05, 0, 0.01])
    wall = trimesh.creation.box([0.01, 0.1, 0.05]); wall.apply_translation([0.03, 0, 0.025])
    r = grasp_analytic([ledge, wall], 0.3, np.array([-0.015]), np.array([0.0]), [90])
    assert not r["fits"].any()
    slab, wall = _walled_slab()
    pan = grasp_analytic([slab, wall], 1.2, np.array([0.195]), np.array([0.0]), [90])
    assert pan["fits"].all() and not pan["level"].any()  # radial rim pinch of a 1.2 kg pan: the weight pitches it


def test_patch_heat_marks_only_the_resting_patch_of_each_face():
    from affdet.sim.pair_map import patch_heat
    box = _on_table(trimesh.creation.box([0.04, 0.04, 0.08]))
    h = patch_heat(box.vertices, {"under": 1.0})
    assert (h[box.vertices[:, 2] < 0.001] == 1.0).all() and (h[box.vertices[:, 2] > 0.07] == 0.0).all()
    h2 = patch_heat(box.vertices, {"under": 0.3, "on": 0.9})  # 'on' puts the top face down
    assert h2[box.vertices[:, 2] > 0.07].max() == 0.9 and h2[box.vertices[:, 2] < 0.001].max() == 0.3


def test_pair_map_has_a_map_on_each_side():
    """A tall 4 x 4 x 8 cm box on a slab: the wide faces are the robust supports, standing up is sensitive."""
    from affdet.sim.pair_map import pair_map
    base = Body("slab", [_on_table(trimesh.creation.box([0.2, 0.2, 0.02]))], 1.0)
    top = Body("box", [_on_table(trimesh.creation.box([0.04, 0.04, 0.08]))], 0.2)
    m = pair_map(base, top, (0.12, 0.12), step=0.04, faces=("under", "front"), procs=2)
    assert m["b_any"].shape == (4, 4) and m["b_any"].any()
    assert set(m["a_face_margin_mm"]) == {"under", "front"}
    assert m["a_face_margin_mm"]["under"] > 0 and m["a_face_margin_mm"]["front"] > 0
    assert m["b_best_face"].max() <= 1
