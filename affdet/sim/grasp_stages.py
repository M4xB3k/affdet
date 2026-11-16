"""Clamp, lift, level: the three stages of a pick, decided from the raw quantities a grasp test records.

One simulated grasp closes the pads, waits, lifts and holds (grasp_map.grasp_test). It stores what it measured, not a verdict:
pad forces and creep while the object is clamped on the table, slip and tilt while it is lifted and held. This module turns
those numbers into three yes/no answers, with every limit a named threshold, so a limit can be changed and the maps re-scored
without simulating again.

  clamp  the pads reached the target force and the object sits still between them (no lift yet): grasp in the narrow sense
  lift   it also follows the lift: off the table, risen, and it did not slide in the grip
  level  it also stayed level: during lift and hold its up axis never tilted more than the limit from where it pointed at
         rest, before the pads closed (a spin about the vertical is not a tilt; the old `swing_deg` counted it)

Only numpy is needed here (no MuJoCo), so the model side can re-score maps too.
"""
from __future__ import annotations

import numpy as np

# Every limit is an assumption until swept. Units are in the names.
THRESHOLDS = dict(
    force_floor=0.8,           # each pad keeps at least this share of the target force while it waits clamped
    force_ceiling=1.5,         # and no more than this share: a pad still far above the target has not settled yet
    min_gap_m=0.002,           # something is between the pads: they did not close onto each other
    max_closure_drift_m=0.001,  # the pads stopped closing during the settle window
    max_creep_m=0.001,         # the pinch point did not move in the object's frame during the settle window
    max_clamp_rot_deg=1.0,     # the object did not turn during the settle window
    min_rise_m=0.05,           # centre of mass rose at least this much by the end (the old `held` rule)
    max_slip_m=0.05,           # largest movement of the pinch point in the object's frame during lift and hold
    max_tilt_deg=30.0,         # largest tilt of the up axis during lift and hold, from its pose at rest (spin does not count)
)

FAIL = dict(off_object=0, blocked=1, force_not_reached=2, clamp_unstable=3, lift=4, not_level=5, success=6)


def stages(m: dict, **thr) -> dict:
    """m: one grasp result or a grasp map (arrays of any equal shape) -> clamp, lift, level, their raw (un-nested) verdicts,
    fail_stage (FAIL codes) and the thresholds used. A map from before the clamp fields existed gives clamp=None,
    lift=held and level=held and swing below the limit, which is what those maps meant."""
    unknown = set(thr) - set(THRESHOLDS)
    if unknown:
        raise ValueError(f"unknown thresholds: {sorted(unknown)}")
    t = {**THRESHOLDS, **thr}
    if "force_reached" not in m:
        held = np.asarray(m["held"], bool)
        level = held & (np.asarray(m["swing_deg"], float) < t["max_tilt_deg"])
        return {"clamp": None, "lift": held, "level": level, "thresholds": t}
    a = lambda k: np.asarray(m[k], float)
    b = lambda k: np.asarray(m[k], bool)
    force = float(m["params"]["force"])
    with np.errstate(invalid="ignore"):  # cells without physics hold NaN: every comparison is False there
        reached = b("force_reached")
        still = ((a("f_l_min") >= t["force_floor"] * force) & (a("f_r_min") >= t["force_floor"] * force)
                 & (a("f_l_max") <= t["force_ceiling"] * force) & (a("f_r_max") <= t["force_ceiling"] * force)
                 & (a("gap_m") >= t["min_gap_m"]) & ~b("at_travel") & (a("closure_drift_m") < t["max_closure_drift_m"])
                 & (a("clamp_creep_m") < t["max_creep_m"]) & (a("clamp_rot_deg") < t["max_clamp_rot_deg"]))
        clamp_raw = reached & still
        lift_raw = ~b("on_table_end") & (a("lifted_m") > t["min_rise_m"]) & (a("slip_max_m") < t["max_slip_m"])
        level_raw = a("tilt_max_deg") < t["max_tilt_deg"]
    on = b("on_object")
    blocked = np.asarray(m.get("blocked", False), bool) | np.zeros(on.shape, bool)  # a single test result only has the key when blocked
    clamp = on & ~blocked & clamp_raw
    lift = clamp & lift_raw
    level = lift & level_raw
    fail = np.full(clamp.shape, FAIL["success"], np.int8)
    for code, mask in (("not_level", ~level_raw), ("lift", ~lift_raw), ("clamp_unstable", ~still), ("force_not_reached", ~reached),
                       ("blocked", blocked), ("off_object", ~on)):  # later entries are earlier stages and overwrite
        fail[mask] = FAIL[code]
    return {"clamp": clamp, "lift": lift, "level": level, "clamp_raw": clamp_raw, "lift_raw": lift_raw & on & ~blocked,
            "level_raw": level_raw, "fail_stage": fail, "thresholds": t}
