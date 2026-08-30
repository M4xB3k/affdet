import math

import numpy as np
import pytest

from affdet.render.omni_pair_grid import (
    parse_grid, split_instance, resolve_mesh_cat, grid_camera_poses, look_at,
    camera_distance_for_fov, FRAME_MARGIN,
)

FOV = 40.0


def test_parse_grid_basic():
    assert parse_grid("2x2") == (2, 2)
    assert parse_grid("3x2") == (3, 2)
    assert parse_grid("1x1") == (1, 1)


def test_parse_grid_case_insensitive():
    assert parse_grid("2X3") == (2, 3)


def test_parse_grid_rejects_zero():
    with pytest.raises(ValueError):
        parse_grid("0x2")


def test_split_instance_simple():
    assert split_instance("durian_030") == ("durian", "durian_030")


def test_split_instance_multiword_category():
    assert split_instance("red_wine_glass_007") == ("red_wine_glass", "red_wine_glass_007")


def test_split_instance_rejects_bare_category():
    with pytest.raises(ValueError):
        split_instance("durian")


def test_resolve_mesh_cat_peeler_remaps_to_knife():
    assert resolve_mesh_cat("peeler") == "knife"


def test_resolve_mesh_cat_passthrough():
    assert resolve_mesh_cat("bowl") == "bowl"
    assert resolve_mesh_cat("dish") == "dish"


def test_camera_distance_for_fov_sphere_fits_exactly_before_margin():
    radius = 0.3
    fov = 40.0
    dist = camera_distance_for_fov(radius, fov, margin=1.0)
    # At this distance, the half-FOV angle exactly subtends the sphere radius:
    # sin(half_fov) == radius / dist.
    half_fov = math.radians(fov) / 2.0
    assert math.isclose(math.sin(half_fov) * dist, radius, rel_tol=1e-9)


def test_camera_distance_for_fov_applies_margin():
    radius, fov = 0.3, 40.0
    no_margin = camera_distance_for_fov(radius, fov, margin=1.0)
    with_margin = camera_distance_for_fov(radius, fov, margin=1.25)
    assert math.isclose(with_margin, no_margin * 1.25, rel_tol=1e-9)


def test_camera_distance_for_fov_default_margin_matches_module_constant():
    radius, fov = 0.3, 40.0
    default = camera_distance_for_fov(radius, fov)
    explicit = camera_distance_for_fov(radius, fov, margin=FRAME_MARGIN)
    assert math.isclose(default, explicit, rel_tol=1e-9)


def test_camera_distance_for_fov_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        camera_distance_for_fov(0.0, FOV)


def test_camera_distance_for_fov_narrower_fov_needs_more_distance():
    radius = 0.3
    wide = camera_distance_for_fov(radius, 60.0)
    narrow = camera_distance_for_fov(radius, 20.0)
    assert narrow > wide


def test_grid_camera_poses_count_matches_rows_cols():
    for rows, cols in [(1, 1), (2, 2), (3, 2), (2, 3), (1, 4)]:
        poses = grid_camera_poses(rows, cols, radius=0.15, fov_deg=FOV)
        assert len(poses) == rows * cols
        for p in poses:
            assert p.shape == (4, 4)


def test_grid_camera_poses_deterministic():
    a = grid_camera_poses(2, 2, radius=0.2, fov_deg=FOV)
    b = grid_camera_poses(2, 2, radius=0.2, fov_deg=FOV)
    for pa, pb in zip(a, b):
        assert np.allclose(pa, pb)


def test_grid_camera_poses_larger_radius_moves_camera_further():
    near = grid_camera_poses(1, 1, radius=0.1, fov_deg=FOV)[0]
    far = grid_camera_poses(1, 1, radius=1.0, fov_deg=FOV)[0]
    near_eye = -near[:3, :3].T @ near[:3, 3]
    far_eye = -far[:3, :3].T @ far[:3, 3]
    assert np.linalg.norm(far_eye) > np.linalg.norm(near_eye)


def test_grid_camera_poses_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        grid_camera_poses(2, 2, radius=0.0, fov_deg=FOV)


def test_grid_camera_poses_bounding_sphere_stays_within_frame():
    # The whole point of the FOV-derived distance: a sphere of the given
    # radius, centered at `center`, must subtend less than the full FOV as
    # seen from every camera in the grid (so nothing at the sphere's silhouette
    # touches/crosses the frame edge).
    radius = 0.25
    center = (0.1, -0.05, 0.12)
    poses = grid_camera_poses(3, 3, radius=radius, fov_deg=FOV, center=center)
    center_v = np.array(center)
    half_fov = math.radians(FOV) / 2.0
    for p in poses:
        eye = -p[:3, :3].T @ p[:3, 3]
        dist_to_center = np.linalg.norm(eye - center_v)
        subtended = math.asin(min(1.0, radius / dist_to_center))
        assert subtended < half_fov


def test_grid_camera_poses_looks_at_given_center_not_origin():
    center = (0.3, -0.2, 0.15)
    poses = grid_camera_poses(1, 1, radius=0.1, fov_deg=FOV, center=center)
    eye = -poses[0][:3, :3].T @ poses[0][:3, 3]
    cam_to_world_rot = poses[0][:3, :3].T
    forward_world = -cam_to_world_rot[:, 2]
    expected = np.array(center) - eye
    expected /= np.linalg.norm(expected)
    assert np.allclose(forward_world, expected, atol=1e-6)


def test_grid_camera_poses_single_cell_looks_down_from_above():
    # rows=1 uses the fixed CAMERA_ELEVATION_DEG (42 deg) as its polar angle,
    # so the camera should end up clearly elevated above the table (positive
    # z, and closer to overhead than to grazing/horizontal).
    poses = grid_camera_poses(1, 1, radius=0.2, fov_deg=FOV)
    eye = -poses[0][:3, :3].T @ poses[0][:3, 3]
    assert eye[2] > 0
    el = math.degrees(math.asin(eye[2] / np.linalg.norm(eye)))
    assert 20.0 < el < 60.0


def test_look_at_points_camera_forward_axis_at_target():
    eye = np.array([1.0, -1.0, 1.0])
    target = np.array([0.0, 0.0, 0.1])
    pose = look_at(eye, target)
    # world_to_camera rotation's -Z row (camera forward) should point toward target
    cam_to_world_rot = pose[:3, :3].T
    forward_world = -cam_to_world_rot[:, 2]
    expected = target - eye
    expected /= np.linalg.norm(expected)
    assert np.allclose(forward_world, expected, atol=1e-6)


def test_grid_camera_poses_azimuth_spans_symmetric_arc():
    poses = grid_camera_poses(1, 3, radius=0.2, fov_deg=FOV)
    eyes = [-p[:3, :3].T @ p[:3, 3] for p in poses]
    # az_span is symmetric around 0 (-50deg, 0deg, +50deg), and x = dist*cos(el)*cos(az)
    # is an even function of az, so the two outer columns land at the same x
    # as each other, both offset from the (az=0) middle column.
    assert math.isclose(eyes[0][0], eyes[2][0], abs_tol=1e-6)
    assert not math.isclose(eyes[0][0], eyes[1][0], abs_tol=1e-3)
