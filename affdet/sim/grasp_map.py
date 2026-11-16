"""Grasp actionability: where on an object can a parallel-jaw gripper pick it up and keep it?

For each grasp centre (x, y) on a grid over the object's footprint, and each gripper yaw, a two-finger gripper
(fingers at table height, closing horizontally, as when picking a flat object from above) closes with a limited
force, lifts it 50 cm and holds. The label is physical:

- `held`: the object is clear of the table at the end and has not slipped in the grip.
- `swing_deg`: how far the object rotated relative to its starting pose (a pan held far from its centre of mass
  pivots about the pinch until the centre of mass hangs below it; a good grasp keeps this small).
- success = held and swing below `max_swing_deg`.

The map over (x, y) is a grasp-actionability heatmap on the object; the best yaw per cell is kept. This is a
stand-in for a Franka hand (two flat pads, 8 cm opening, 30 N each); the real hand model from MuJoCo Menagerie
is the next step.
"""
from __future__ import annotations

import math
import os
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import affdet.sim.mujoco_support as ms
from affdet.sim.mujoco_support import Body

PAD = dict(half=(0.010, 0.004, 0.0125), open_half=0.040, travel=0.045)  # pad half sizes (m), inner opening, travel
FORCE_N = 30.0
PAD_BELOW_TOP_M = 0.03  # pad bottom this far below the surface under the grasp centre (squeezing above a handle's widest point wedges the pads off)
LIFT_M = 0.50  # a 40 cm pan held by its handle hangs below it: the lift must clear its length
PHASES = dict(close_max=1.2, lift=1.5, hold=0.4)  # seconds
PAD_MASS_KG = 5.0      # pads are welded to their targets, so their mass only sets how stiff their contacts feel (effective mass); 0.2 kg pads sank into a pan handle and lost it
PAD_SOLREF = None      # None: the scene default; a string like "0.004 1" makes pad contacts stiffer (a priority-1 geom's value wins)
SQUEEZE_SPEED = 0.01   # m/s: closing speed once a pad touches (100 um per step), so the force can be regulated to a few N
PARK_Z = 2.0  # pads start parked above the ray cast start so the height probes only see the object
PAD_FRICTION = (1.0, 0.01, 0.001)  # sliding, torsional (m), rolling (m)

# Every setting of one grasp test, so a sweep can vary them (grasp_params) and each result records what it used.
# MuJoCo gives a contact the element-wise MAX of the two geoms' friction when their priorities are equal, so with
# pad_priority 0 the pad's 1.0 wins over the objects' 0.6: lowering grasp friction needs pad_priority 1 (the pad's value is used).
DEFAULTS = dict(force=FORCE_N, pad_half=PAD["half"], open_half=PAD["open_half"], travel=PAD["travel"], pad_friction=PAD_FRICTION,
                pad_priority=0, pad_mass=PAD_MASS_KG, pad_below_top=PAD_BELOW_TOP_M, lift=LIFT_M, lift_time=PHASES["lift"],
                hold=PHASES["hold"], close_max=PHASES["close_max"],
                settle=0.1,           # s between "both pads at full force" (or the closing time running out) and the lift: the clamp is judged here
                min_clearance=0.008)  # m of the pinched surface a pad must be able to reach below its top, else the cell is blocked

# What a test measures besides the old outcome fields (grasp_stages turns these into clamp / lift / level). NaN where no physics ran.
STILL_S = 0.1  # the clamp is judged on the last STILL_S seconds of the settle window (all of it at the default settle)
RAW_FLOAT = ("clearance_m", "t_clamp_s", "f_l_min", "f_r_min", "f_l_max", "f_r_max", "gap_m", "closure_drift_m", "clamp_creep_m", "clamp_rot_deg",
             "slip_max_m", "tilt_max_deg", "tilt_end_deg", "tilt_lift_max_deg", "rot_max_deg", "lifted_max_m", "t_contact_lost_s")
RAW_BOOL = ("force_reached", "at_travel", "on_table_end")


def _angle(Ra, Rb):
    """Rotation angle (deg) between two orientations."""
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def grasp_params(**over) -> dict:
    """DEFAULTS with some settings replaced; an unknown name is an error (a typo must not silently run the default)."""
    unknown = set(over) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown grasp settings: {sorted(unknown)}")
    p = {**DEFAULTS, **over}
    p["pad_half"], p["pad_friction"] = tuple(map(float, p["pad_half"])), tuple(map(float, p["pad_friction"]))
    return p


def _pad_xml(name, x, y, z, yaw, p=None):
    """A dynamic pad welded to a mocap target. (A mocap body has zero velocity as far as contacts are concerned,
    so a teleporting mocap pad can squeeze but never drag an object along with friction.)"""
    p = p or DEFAULTS
    hx, hy, hz = p["pad_half"]
    fr = p["pad_friction"]
    prio = 1 if (PAD_SOLREF or p["pad_priority"]) else 0
    q = f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"
    return (f'<body name="t{name}" mocap="true" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{q}"/>'
            f'<body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{q}"><freejoint/>'
            f'<geom type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" friction="{fr[0]} {fr[1]} '
            f'{fr[2]}" condim="6" mass="{p["pad_mass"]}" solref="{PAD_SOLREF or ms.SIM["solref"]}" priority="{prio}" '
            f'rgba="0.3 0.3 0.3 1"/></body>')


_WELD = ('<equality><weld body1="fl" body2="tfl" solref="0.008 1"/><weld body1="fr" body2="tfr" solref="0.008 1"/></equality>')


def _regulate(s, force, v, dt, target=FORCE_N, travel=PAD["travel"]):
    """One step of the squeeze controller for a pad whose target has closed `s` metres: advance while the contact force is
    below `target` (fast in free air, slowly once touching, so a stiff contact does not overshoot), back off when it is more than
    15 % above. Without the back-off a pad's hold force is set by how far its target overshot, not by the target."""
    if force < target:
        return min(s + (v if force < 3.0 else SQUEEZE_SPEED) * dt, travel)
    if force > 1.15 * target:
        return max(s - SQUEEZE_SPEED * dt, 0.0)
    return s


def _pad_force(m, data, body_id, obj_id):
    """Sum of contact normal forces between one pad and the object."""
    import mujoco
    tot, f = 0.0, np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
        if pair == {body_id, obj_id}:
            mujoco.mj_contactForce(m, data, i, f)
            tot += f[0]
    return tot


def grasp_test(body: Body, x: float, y: float, yaw_deg: float, max_swing_deg: float = 30.0, record=None, params=None) -> dict:
    """One grasp: close at (x, y) with the gripper yawed `yaw_deg` about z, lift, hold.

    The two pads are moved kinematically (mocap): each closes at 10 cm/s until its contact force with the object
    reaches the target force, then regulates it (a force-limited gripper). The object is then lifted and held.
    `params` (see grasp_params; None = DEFAULTS) sets force, pad size and friction, lift and hold. `record(m, data)`
    is called every step (for rendering)."""
    import mujoco
    p = params or DEFAULTS
    yaw = math.radians(yaw_deg)
    u = np.array([-math.sin(yaw), math.cos(yaw)])  # closing direction in the world (hand y axis)
    R = np.eye(3)
    pos = ms._place(body, R)
    c0 = p["open_half"] + p["pad_half"][1]
    with tempfile.TemporaryDirectory() as d:
        ms._write_parts(body, d)
        xml = ms.build_xml([body], [(pos, np.array([1.0, 0, 0, 0]))])
        # pads start open at the grasp centre; the height is set below once the surface under it is known
        xml = xml.replace("</worldbody>", _pad_xml("fl", x - c0 * u[0], y - c0 * u[1], PARK_Z, yaw, p)
                          + _pad_xml("fr", x + c0 * u[0], y + c0 * u[1], PARK_Z, yaw, p) + "</worldbody>")
        xml = xml.replace("</mujoco>", _WELD + "</mujoco>")
        m = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    dt = m.opt.timestep
    il, ir = m.body("tfl").mocapid[0], m.body("tfr").mocapid[0]
    bl, br, oid = m.body("fl").id, m.body("fr").id, m.body(body.name).id
    # lower the pads onto whatever is under the grasp centre (a pan handle floats 3-5 cm above the table)
    gid = np.array([-1], dtype=np.int32)

    def top_at(px, py):
        d_ = mujoco.mj_ray(m, data, np.array([px, py, 1.0]), np.array([0.0, 0, -1.0]), None, 1, -1, gid)
        return (1.0 - d_) if d_ >= 0 and m.geom_bodyid[gid[0]] == oid else 0.0

    z_top = top_at(x, y)
    if z_top <= 0.0:
        return {"x": x, "y": y, "yaw": yaw_deg, "held": False, "lifted_m": 0.0, "swing_deg": 0.0, "slip_m": 0.0,
                "success": False, "com0": data.xipos[oid].copy(), "on_object": False}
    z0 = max(0.003, z_top - p["pad_below_top"])
    # a pad may not start inside the object: whatever is under either pad footprint (a pan's floor next to its rim
    # wall) sets the lowest height the finger can reach; if that is above the surface being pinched, no grasp here
    hx, hy, _ = p["pad_half"]
    t = np.array([-u[1], u[0]])  # the pad's long side runs along the world direction perpendicular to closing
    floor = 0.0
    for sgn in (-1, 1):
        cx, cy = x + sgn * c0 * u[0], y + sgn * c0 * u[1]
        for a_ in (-hx, 0.0, hx):
            for b_ in (-hy, 0.0, hy):
                floor = max(floor, top_at(cx + a_ * t[0] + b_ * u[0], cy + a_ * t[1] + b_ * u[1]))
    z0 = max(z0, floor + 0.001)
    if z0 > z_top - p["min_clearance"]:
        return {"x": x, "y": y, "yaw": yaw_deg, "held": False, "lifted_m": 0.0, "swing_deg": 0.0, "slip_m": 0.0,
                "success": False, "com0": data.xipos[oid].copy(), "on_object": True, "blocked": True, "clearance_m": float(z_top - z0)}
    for i in (il, ir):
        data.mocap_pos[i][2] = z0
    for name in ("fl", "fr"):  # start the pads at their targets: spawning them at table height buries them in the object's hulls
        a = m.jnt_qposadr[m.body(name).jntadr[0]]
        data.qpos[a + 2] = z0
    mujoco.mj_forward(m, data)
    com0 = data.xipos[oid].copy()
    R0 = data.xmat[oid].reshape(3, 3).copy()
    t_lift, t_hold, t_close_max = p["lift_time"], p["hold"], p["close_max"]
    F = p["force"]
    sl = sr = 0.0  # how far each pad has closed (m)
    v = 0.10       # closing speed (m/s)
    rel0 = None
    t_start = None  # time the lift starts: both pads at full force, or the closing time ran out
    f_hold = []    # pad forces during the hold (to check the regulated force)
    # raw quantities for the stages. Reading them does not touch the simulation: the old fields come out the same.
    t_clamp, reached, t_lost = None, False, float("nan")
    win = []  # settle window, one entry per step: (t, force left, force right, pinch point in the object's frame, orientation,
    #          pad closure sum, larger pad closure)
    Rs = None                             # orientation when the lift starts: tilt is measured against it
    slip_max = tilt_max = tilt_lift_max = rot_max = 0.0
    up0 = R0[:, 2].copy()                 # the up axis at rest, before the pads close: `level` is judged against it
    lifted_max = -float("inf")
    k = 0
    while True:
        t = k * dt
        fl_, fr_ = _pad_force(m, data, bl, oid), _pad_force(m, data, br, oid)
        if t_start is None and ((fl_ >= F and fr_ >= F) or t >= t_close_max):
            t_clamp, reached = t, bool(fl_ >= F and fr_ >= F)  # False: the closing time ran out before both pads felt F
            t_start = t + p["settle"]  # let the squeeze settle
        in_win = t_start is not None and t < t_start
        if not in_win and t_start is not None and t_lost != t_lost and min(fl_, fr_) < 1.0:
            t_lost = t  # first time a pad lets go of the object after the lift started
        if t_start is not None and t > t_start + t_lift:
            f_hold.append(0.5 * (fl_ + fr_))
        sl = _regulate(sl, fl_, v, dt, F, p["travel"])
        sr = _regulate(sr, fr_, v, dt, F, p["travel"])
        data.mocap_pos[il][:2] = [x - (c0 - sl) * u[0], y - (c0 - sl) * u[1]]
        data.mocap_pos[ir][:2] = [x + (c0 - sr) * u[0], y + (c0 - sr) * u[1]]
        s_ = 0.0 if t_start is None else np.clip((t - t_start) / t_lift, 0, 1)
        lift = p["lift"] * (3 * s_ * s_ - 2 * s_ ** 3)
        data.mocap_pos[il][2] = data.mocap_pos[ir][2] = z0 + lift
        mujoco.mj_step(m, data)
        Rk = data.xmat[oid].reshape(3, 3)
        pinch = Rk.T @ (0.5 * (data.xpos[bl] + data.xpos[br]) - data.xipos[oid])
        if in_win:
            win.append((t, fl_, fr_, pinch.copy(), Rk.copy(), sl + sr, max(sl, sr)))
        if rel0 is None and t_start is not None and t >= t_start:
            # the pinch point in the object's own frame: it must not move if the object does not slip in the grip
            rel0, Rs = pinch.copy(), Rk.copy()
        elif rel0 is not None:
            slip_max = max(slip_max, float(np.linalg.norm(pinch - rel0)))
            tilt_max = max(tilt_max, math.degrees(math.acos(np.clip(up0 @ Rk[:, 2], -1, 1))))
            tilt_lift_max = max(tilt_lift_max, math.degrees(math.acos(np.clip(Rs[:, 2] @ Rk[:, 2], -1, 1))))
            rot_max = max(rot_max, _angle(Rs, Rk))
            lifted_max = max(lifted_max, float(data.xipos[oid][2] - com0[2]))
        if record is not None:
            record(m, data)
        k += 1
        if t_start is not None and t > t_start + t_lift + t_hold:
            break
    com1 = data.xipos[oid].copy()
    R1 = data.xmat[oid].reshape(3, 3)
    lifted = float(com1[2] - com0[2])
    swing = math.degrees(math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
    rel1 = R1.T @ (0.5 * (data.xpos[bl] + data.xpos[br]) - data.xipos[oid])
    slip = float(np.linalg.norm(rel1 - rel0))
    on_table = any({data.contact[i].geom1, data.contact[i].geom2} & {0} and oid in (m.geom_bodyid[data.contact[i].geom1],
                   m.geom_bodyid[data.contact[i].geom2]) for i in range(data.ncon))
    held = bool(np.isfinite(com1).all()) and not on_table and lifted > 0.05 and slip < 0.05
    win = [w for w in win if w[0] >= t_start - STILL_S - 1e-9]
    win0, win1 = (win[0], win[-1]) if win else (None, None)
    nan = float("nan")
    return {"x": x, "y": y, "yaw": yaw_deg, "held": bool(held), "lifted_m": lifted, "swing_deg": swing, "slip_m": slip,
            "success": bool(held and swing < max_swing_deg), "com0": com0, "on_object": True,
            "hold_force_n": float(np.mean(f_hold)) if f_hold else 0.0, "params": dict(p),
            "clearance_m": float(z_top - z0), "t_clamp_s": float(t_clamp), "force_reached": reached,
            "f_l_min": min(w[1] for w in win) if win else nan, "f_r_min": min(w[2] for w in win) if win else nan,
            "f_l_max": max(w[1] for w in win) if win else nan, "f_r_max": max(w[2] for w in win) if win else nan,
            "gap_m": float(2 * p["open_half"] - win1[5]) if win else nan,
            "at_travel": bool(win1[6] >= p["travel"] - 1e-9) if win else False,  # end of the clamp window, not after the lift
            "closure_drift_m": float(abs(win1[5] - win0[5])) if win else nan,
            "clamp_creep_m": float(np.linalg.norm(win1[3] - win0[3])) if win else nan,
            "clamp_rot_deg": _angle(win0[4], win1[4]) if win else nan,
            "slip_max_m": max(slip_max, slip), "tilt_max_deg": tilt_max,
            "tilt_end_deg": math.degrees(math.acos(np.clip(up0 @ R1[:, 2], -1, 1))), "tilt_lift_max_deg": tilt_lift_max,
            "rot_max_deg": rot_max,
            "lifted_max_m": lifted_max, "on_table_end": bool(on_table), "t_contact_lost_s": t_lost}


_G = {}


def _init(body, params=None):
    _G["body"] = body
    _G["params"] = params


def _one(a):
    return grasp_test(_G["body"], *a, params=_G["params"])


def grasp_map(body: Body, extent_xy, step=0.015, yaws=(0, 30, 60, 90, 120, 150), procs=8, max_swing_deg=30.0, params=None) -> dict:
    """`params` (grasp_params) is handed to every worker through the pool initializer; each test reports what it used and
    the map keeps the workers' copy, so a stored map proves its own settings."""
    params = params or DEFAULTS
    xs = np.arange(-extent_xy[0] / 2, extent_xy[0] / 2 + 1e-9, step)
    ys = np.arange(-extent_xy[1] / 2, extent_xy[1] / 2 + 1e-9, step)
    jobs = [(x, y, yaw, max_swing_deg) for y in ys for x in xs for yaw in yaws]
    # an executor, not multiprocessing.Pool: when a worker dies without an exception (a native crash in MuJoCo), Pool.map
    # waits forever, the executor raises BrokenProcessPool. Fork, as before: workers inherit the parent's modules.
    with ProcessPoolExecutor(procs, mp_context=mp.get_context("fork"), initializer=_init, initargs=(body, params)) as ex:
        res = list(ex.map(_one, jobs, chunksize=4))
    used = {r["params"].__repr__() for r in res if "params" in r}
    if len(used) > 1 or (used and next(iter(used)) != dict(params).__repr__()):
        raise RuntimeError(f"workers ran with other grasp settings than requested: {used}")
    shape = (len(ys), len(xs), len(yaws))
    ok = np.array([r["success"] for r in res]).reshape(shape)
    swing = np.array([r["swing_deg"] for r in res]).reshape(shape)
    held = np.array([r["held"] for r in res]).reshape(shape)
    blocked = np.array([r.get("blocked", False) for r in res]).reshape(shape)  # a pad could not reach the pinched surface
    lifted = np.array([r["lifted_m"] for r in res]).reshape(shape)
    slip = np.array([r["slip_m"] for r in res]).reshape(shape)
    on_obj = np.array([r["on_object"] for r in res]).reshape(shape)
    hold_f = np.array([r.get("hold_force_n", 0.0) for r in res]).reshape(shape)
    raw = {k: np.array([r.get(k, np.nan) for r in res], np.float32).reshape(shape) for k in RAW_FLOAT}
    raw.update({k: np.array([r.get(k, False) for r in res], bool).reshape(shape) for k in RAW_BOOL})
    from affdet.sim.grasp_stages import stages
    st = stages({**raw, "lifted_m": lifted, "on_object": on_obj, "blocked": blocked, "params": params})
    raw.update({k: st[k] for k in ("clamp", "lift", "level", "fail_stage")})
    raw.update(schema=2, thresholds=st["thresholds"])
    return {**raw, "xs": xs, "ys": ys, "yaws": list(yaws), "success": ok, "held": held, "swing_deg": swing,
            "blocked": blocked, "lifted_m": lifted, "slip_m": slip, "score": ok.mean(2), "best_yaw_idx": ok.argmax(2), "com": res[0]["com0"], "step": step,
            "on_object": on_obj, "hold_force_n": hold_f, "params": dict(params)}
