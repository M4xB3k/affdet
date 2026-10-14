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
    """A uniform 20 x 3 x 2 cm bar (1.2 kg) pinched across its width at 30 N: a grasp near the COM lifts it flat, and past
    the torsional limit of the pads (m g d > 2 mu_t F, about 5 to 7 cm) it pivots: 0.4 to 1.3 deg up to 6 cm, 21 deg at
    8 cm, 52 deg at 9.5 cm. The squeeze is regulated to 30 N (pads that overshoot give a rigid grip and no cliff)."""
    from affdet.sim.grasp_map import grasp_test
    bar = Body("bar", [_on_table(trimesh.creation.box([0.2, 0.03, 0.02]))], 1.2)
    res = [grasp_test(bar, d, 0.0, 0.0) for d in (0.0, 0.05, 0.095)]
    assert res[0]["success"] and res[0]["swing_deg"] < 3
    assert res[1]["swing_deg"] < 3  # inside the limit
    assert res[2]["swing_deg"] > 30 and not res[2]["success"]  # past it


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


def test_a_placement_that_overlaps_the_base_at_release_is_not_stable():
    """A 10 cm box laid down across two thin walls 9.2 cm apart: the height probes under its footprint (within 80 % of
    its half length) miss the walls, so it is released at the floor and starts inside them. It must not count as stable
    however the solver pushes it out."""
    from affdet.sim.mujoco_support import stack_test
    floor = trimesh.creation.box([0.2, 0.2, 0.01]); floor.apply_translation([0, 0, 0.005])
    w1 = trimesh.creation.box([0.006, 0.2, 0.1]); w1.apply_translation([-0.046, 0, 0.05])
    w2 = trimesh.creation.box([0.006, 0.2, 0.1]); w2.apply_translation([0.046, 0, 0.05])
    slot = Body("slot", [floor, w1, w2], 1.0)
    bar = Body("bar", [_on_table(trimesh.creation.box([0.10, 0.03, 0.03]))], 0.2)
    r = stack_test(slot, bar, "under", "under", offset=(0.0, 0.0), seconds=0.0)
    assert r["release_penetration_m"] > 0.003 and not r["stable"]
    small = Body("small", [_on_table(trimesh.creation.box([0.03, 0.03, 0.03]))], 0.1)
    assert stack_test(slot, small, "under", "under", offset=(0.0, 0.0), seconds=0.0)["release_penetration_m"] <= 0.003


def _fake_pair():
    from affdet.sim.pair_map import aggregate
    xs = np.arange(-0.1, 0.1001, 0.05); ys = np.arange(-0.1, 0.1001, 0.05)
    stable = np.zeros((5, 5), bool); stable[1:4, 1:4] = True
    margin = np.zeros((5, 5)); margin[2, 2] = 40.0; margin[1:4, 1:4] = np.maximum(margin[1:4, 1:4], 10.0)
    pf = {"under": {"stable": stable, "margin_mm": margin, "best_margin_mm": 40.0, "xs": xs, "ys": ys}}
    return aggregate(pf, ["under", "on"], ["on"], 0.05)


def test_pair_labels_put_a_heatmap_on_each_object_and_round_trip(tmp_path):
    from affdet.sim.label_export import CAP_MM, load, pair_labels, save
    base = _on_table(trimesh.creation.box([0.2, 0.2, 0.02]))
    top = _on_table(trimesh.creation.box([0.04, 0.04, 0.08]))
    lab = pair_labels(_fake_pair(), top, base)
    b, a = lab["B"], lab["A"]
    assert b["score"].max() == 1.0 and b["score"].min() == 0.0 and b["margin_mm"].max() == 40.0
    assert b["valid"].sum() > 0 and (b["score"][~b["valid"]] == 0).all()   # only upward faces carry a label
    assert a["valid"].sum() > 0 and a["face"].max() == 0                   # the 'under' patch only: 'on' cannot rest
    assert np.allclose(a["margin_mm"][a["face"] == 0], 40.0)
    save(str(tmp_path / "b.npz"), b, {"object": "slab"})
    got, meta = load(str(tmp_path / "b.npz"))
    assert meta["cap_mm"] == CAP_MM and meta["schema_version"] == 1 and np.allclose(got["score"], b["score"])


def test_grasp_labels_report_held_and_stable_fractions():
    from affdet.sim.label_export import grasp_labels
    xs = ys = np.array([-0.05, 0.0, 0.05])
    held = np.zeros((3, 3, 2), bool); held[1, 1] = True
    swing = np.full((3, 3, 2), 50.0); swing[1, 1, 0] = 10.0
    g = {"xs": xs, "ys": ys, "step": 0.05, "held": held, "swing_deg": swing}
    lab = grasp_labels(g, _on_table(trimesh.creation.box([0.1, 0.1, 0.02])))
    assert lab["held_frac"].max() == 1.0 and abs(lab["stable_frac"].max() - 0.5) < 1e-9


def test_slide_a_tall_narrow_box_topples_when_pushed_high_and_a_flat_one_slides():
    """Push a 4 x 4 x 12 cm, 0.2 kg box at 15 % and 50 % of its height, and a 12 x 12 x 4 cm, 1 kg box at 90 %."""
    from affdet.sim.slide_map import slide_test
    def box(x, y, z, mass):
        return Body("b", [_on_table(trimesh.creation.box([x, y, z]))], mass)
    tall = box(0.04, 0.04, 0.12, 0.2)
    assert slide_test(tall, 0.0, 0.15)["success"]
    high = slide_test(tall, 0.0, 0.5)
    assert not high["success"] and high["tilt_deg"] > 30
    assert slide_test(box(0.12, 0.12, 0.04, 1.0), 0.0, 0.9)["success"]


def test_slide_labels_paint_the_samples_on_the_side_surface():
    from affdet.sim.label_export import slide_labels
    az = list(range(0, 360, 90)); fr = [0.25, 0.75]
    ok = np.zeros((2, 4), bool); ok[0, :] = True  # pushing low works from every side, high from none
    sm = {"azimuths": az, "fracs": fr, "success": ok, "moved_frac": ok * 1.0, "tilt_deg": np.where(ok, 0.0, 60.0), "yaw_deg": np.zeros((2, 4)),
          "on_object": np.ones((2, 4), bool)}
    box = _on_table(trimesh.creation.box([0.06, 0.06, 0.1]))
    lab = slide_labels(sm, box)
    z = lab["vertices"][:, 2]
    side = lab["valid"]
    assert side.sum() > 0 and (lab["slide_score"][side & (z < 0.04)] == 1.0).all() and (lab["slide_score"][side & (z > 0.07)] == 0.0).all()
