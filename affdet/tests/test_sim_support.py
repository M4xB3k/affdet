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
