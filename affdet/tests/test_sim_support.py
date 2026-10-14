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
