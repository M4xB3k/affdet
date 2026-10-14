import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from affdet.geometry.face_labels import label_faces


def _on_table(m):
    m.apply_translation([0, 0, -m.bounds[0, 2]])
    return m


def _faces(labels):
    return {f.face: f for f in labels.faces}


def _grasps(labels):
    return {(g.palm, g.finger1, g.finger2): g.graspable for g in labels.grasps}


def test_round_shaker_only_stands_on_its_ends():
    lab = label_faces(_on_table(trimesh.creation.cylinder(radius=0.025, height=0.09, sections=64)))
    f = _faces(lab)
    assert f["under"].supportable and f["on"].supportable
    assert not any(f[s].supportable for s in ("left", "right", "front", "back"))


def test_faceted_shaker_also_lies_on_its_side():
    # hexagonal prism with a pair of flats facing left/right
    hexa = trimesh.creation.cylinder(radius=0.025, height=0.09, sections=6)
    hexa.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 6, [0, 0, 1]))
    f = _faces(label_faces(_on_table(hexa)))
    assert f["left"].supportable and f["right"].supportable


def test_domed_cap_is_not_supportable_upside_down():
    body = trimesh.creation.cylinder(radius=0.025, height=0.07, sections=64)
    dome = trimesh.creation.icosphere(subdivisions=4, radius=0.025)
    dome.apply_translation([0, 0, 0.035])
    f = _faces(label_faces(_on_table(trimesh.util.concatenate([body, dome]))))
    assert f["under"].supportable
    assert not f["on"].supportable


def test_supportive_needs_a_wide_flat_top():
    box = _faces(label_faces(_on_table(trimesh.creation.box([0.3, 0.2, 0.02]))))
    assert box["on"].supportive
    thin = _faces(label_faces(_on_table(trimesh.creation.cylinder(radius=0.01, height=0.09))))
    assert not thin["on"].supportive


def test_grasp_width_against_gripper_opening():
    g = _grasps(label_faces(_on_table(trimesh.creation.cylinder(radius=0.025, height=0.09, sections=64))))
    assert g[("on", "left", "right")] and g[("front", "left", "right")]
    assert not g[("front", "under", "on")]  # 9 cm tall: fingers cannot span it
    wide = _grasps(label_faces(_on_table(trimesh.creation.cylinder(radius=0.06, height=0.05, sections=64))))
    assert not wide[("on", "left", "right")]
    assert wide[("front", "under", "on")]


def test_predicates_and_lift():
    lab = label_faces(_on_table(trimesh.creation.box([0.05, 0.05, 0.1])), mass=0.2)
    preds = lab.predicates("salt_shaker")
    assert "(isSupportable salt_shaker under)" in preds
    assert "(canGraspFrom salt_shaker body on left right)" in preds
    assert "(canLift salt_shaker body)" in preds


def test_wide_finely_tessellated_round_body_still_rolls():
    f = _faces(label_faces(_on_table(trimesh.creation.cylinder(radius=0.045, height=0.2, sections=84))))
    assert not any(f[s].supportable for s in ("left", "right", "front", "back"))


def test_nearly_flat_cap_counts_as_flat():
    # octagonal shaker whose top is a 0.4 mm high cone over 6 mm (4 deg facets)
    body = trimesh.creation.cylinder(radius=0.029, height=0.086, sections=8)
    body.apply_translation([0, 0, 0.043])
    ring = [[0.006 * np.cos(a), 0.006 * np.sin(a), z] for z in (0.086, 0.0896) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    cap = trimesh.convex.convex_hull(np.vstack([ring, [[0, 0, 0.09]], [[0.012 * np.cos(a), 0.012 * np.sin(a), 0.086] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]]))
    f = _faces(label_faces(_on_table(trimesh.util.concatenate([body, cap]))))
    assert f["on"].supportable
