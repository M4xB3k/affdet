"""Clamp / lift / level: the three stages of a pick, read from the raw quantities one grasp test records."""
import numpy as np
import pytest

from affdet.sim.grasp_stages import FAIL, THRESHOLDS, stages


def _map(**over):
    """A synthetic 1 x 1 x 4 map: four gripper yaws at one cell, all of them a clean pick unless overridden."""
    n = 4
    m = {"on_object": np.ones(n, bool), "blocked": np.zeros(n, bool), "force_reached": np.ones(n, bool),
         "f_l_min": np.full(n, 30.0), "f_r_min": np.full(n, 30.0), "f_l_max": np.full(n, 33.0), "f_r_max": np.full(n, 33.0), "gap_m": np.full(n, 0.03), "at_travel": np.zeros(n, bool),
         "closure_drift_m": np.zeros(n), "clamp_creep_m": np.zeros(n), "clamp_rot_deg": np.zeros(n),
         "on_table_end": np.zeros(n, bool), "lifted_m": np.full(n, 0.5), "slip_max_m": np.zeros(n),
         "tilt_max_deg": np.array([1.0, 20.0, 40.0, 70.0]), "params": {"force": 30.0}}
    m.update({k: np.asarray(v) for k, v in over.items()})
    return {k: (v.reshape(1, 1, n) if isinstance(v, np.ndarray) else v) for k, v in m.items()}


def test_default_thresholds_give_the_three_nested_stages():
    s = stages(_map())
    assert s["clamp"].all() and s["lift"].all()
    assert s["level"].ravel().tolist() == [True, True, False, False]
    assert s["fail_stage"].ravel().tolist() == [FAIL["success"]] * 2 + [FAIL["not_level"]] * 2


def test_a_threshold_can_be_changed_without_new_simulation():
    m = _map()
    before = {k: np.copy(v) for k, v in m.items() if isinstance(v, np.ndarray)}
    assert stages(m, max_tilt_deg=60.0)["level"].ravel().tolist() == [True, True, True, False]
    assert stages(m, max_tilt_deg=15.0)["level"].ravel().tolist() == [True, False, False, False]
    assert all((m[k] == v).all() for k, v in before.items())  # the map itself is not touched
    with pytest.raises(ValueError):
        stages(m, max_tilt=60.0)  # a mistyped threshold must not silently run the default


def test_stages_are_nested_and_name_the_first_failure():
    m = _map(force_reached=[True, False, True, True], slip_max_m=[0.0, 0.0, 0.2, 0.0], f_l_min=[30.0, 30.0, 30.0, 10.0],
             tilt_max_deg=[1.0, 1.0, 1.0, 1.0])
    s = stages(m)
    assert s["clamp"].ravel().tolist() == [True, False, True, False]
    assert s["lift"].ravel().tolist() == [True, False, False, False]
    assert s["level"].ravel().tolist() == [True, False, False, False]
    assert s["fail_stage"].ravel().tolist() == [FAIL["success"], FAIL["force_not_reached"], FAIL["lift"], FAIL["clamp_unstable"]]
    # the raw verdicts stay available: yaw 1 was lifted although its clamp failed (scooped or wedged, worth seeing)
    assert s["lift_raw"].ravel().tolist() == [True, True, False, True]
    assert (s["level"] <= s["lift"]).all() and (s["lift"] <= s["clamp"]).all()


def test_cells_without_physics_fail_before_the_clamp():
    nan = np.full(4, np.nan)
    m = _map(on_object=[False, True, True, True], blocked=[False, True, False, False], force_reached=[False, False, True, True],
             f_l_min=[np.nan, np.nan, 30.0, 30.0], tilt_max_deg=[np.nan, np.nan, 1.0, 1.0], lifted_m=[np.nan, np.nan, 0.5, 0.5])
    s = stages(m)
    assert s["fail_stage"].ravel().tolist()[:2] == [FAIL["off_object"], FAIL["blocked"]]
    assert not s["clamp"].ravel()[:2].any() and s["level"].ravel()[2:].all()
    assert nan.shape == (4,)


def test_an_old_map_without_clamp_fields_keeps_its_old_meaning():
    held = np.array([True, True, False]).reshape(1, 1, 3)
    swing = np.array([5.0, 45.0, 5.0]).reshape(1, 1, 3)
    s = stages({"held": held, "swing_deg": swing})
    assert s["clamp"] is None
    assert s["lift"].ravel().tolist() == [True, True, False]
    assert s["level"].ravel().tolist() == [True, False, False]


def test_thresholds_are_listed_with_the_result():
    s = stages(_map(), max_slip_m=0.02)
    assert s["thresholds"] == {**THRESHOLDS, "max_slip_m": 0.02}
