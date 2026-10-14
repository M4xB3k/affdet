"""Grasp actionability: where on an object can a parallel-jaw gripper pick it up and keep it?

For each grasp centre (x, y) on a grid over the object's footprint, and each gripper yaw, a two-finger gripper
(fingers at table height, closing horizontally, as when picking a flat object from above) closes with a limited
force, lifts the object 15 cm and holds. The label is physical:

- `held`: the object left the table with the gripper and is still carried at the end.
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
from multiprocessing import Pool

import numpy as np

import affdet.sim.mujoco_support as ms
from affdet.sim.mujoco_support import Body

PAD = dict(half=(0.010, 0.004, 0.0125), open_half=0.040, travel=0.045)  # pad half sizes (m), inner opening, travel
FORCE_N = 30.0
PAD_BELOW_TOP_M = 0.03  # pad bottom this far below the surface under the grasp centre (squeezing above a handle's widest point wedges the pads off)
LIFT_M = 0.15
PHASES = dict(close_max=1.2, lift=0.6, hold=0.3)  # seconds
PAD_FRICTION = (1.0, 0.01, 0.001)  # sliding, torsional (m), rolling (m)


def _pad_xml(name, x, y, z, yaw):
    """A dynamic pad welded to a mocap target. (A mocap body has zero velocity as far as contacts are concerned,
    so a teleporting mocap pad can squeeze but never drag an object along with friction.)"""
    hx, hy, hz = PAD["half"]
    q = f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"
    return (f'<body name="t{name}" mocap="true" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{q}"/>'
            f'<body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{q}"><freejoint/>'
            f'<geom type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" friction="{PAD_FRICTION[0]} {PAD_FRICTION[1]} '
            f'{PAD_FRICTION[2]}" condim="6" mass="0.2" solref="{ms.SIM["solref"]}" rgba="0.3 0.3 0.3 1"/></body>')


_WELD = ('<equality><weld body1="fl" body2="tfl" solref="0.008 1"/><weld body1="fr" body2="tfr" solref="0.008 1"/></equality>')


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


def grasp_test(body: Body, x: float, y: float, yaw_deg: float, max_swing_deg: float = 30.0, record=None) -> dict:
    """One grasp: close at (x, y) with the gripper yawed `yaw_deg` about z, lift, hold.

    The two pads are moved kinematically (mocap): each closes at 5 cm/s until its contact force with the object
    reaches FORCE_N, then stops (a force-limited gripper). The object is then lifted LIFT_M and held. `record(m,
    data)` is called every step (for rendering)."""
    import mujoco
    yaw = math.radians(yaw_deg)
    u = np.array([-math.sin(yaw), math.cos(yaw)])  # closing direction in the world (hand y axis)
    R = np.eye(3)
    pos = ms._place(body, R)
    c0 = PAD["open_half"] + PAD["half"][1]
    with tempfile.TemporaryDirectory() as d:
        ms._write_parts(body, d)
        xml = ms.build_xml([body], [(pos, np.array([1.0, 0, 0, 0]))])
        # pads start open at the grasp centre; the height is set below once the surface under it is known
        xml = xml.replace("</worldbody>", _pad_xml("fl", x - c0 * u[0], y - c0 * u[1], 0.003, yaw)
                          + _pad_xml("fr", x + c0 * u[0], y + c0 * u[1], 0.003, yaw) + "</worldbody>")
        xml = xml.replace("</mujoco>", _WELD + "</mujoco>")
        m = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    dt = m.opt.timestep
    il, ir = m.body("tfl").mocapid[0], m.body("tfr").mocapid[0]
    bl, br, oid = m.body("fl").id, m.body("fr").id, m.body(body.name).id
    # lower the pads onto whatever is under the grasp centre (a pan handle floats 3-5 cm above the table)
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, data, np.array([x, y, 1.0]), np.array([0.0, 0, -1.0]), None, 1, bl, gid)
    if dist < 0 or m.geom_bodyid[gid[0]] != oid:
        return {"x": x, "y": y, "yaw": yaw_deg, "held": False, "lifted_m": 0.0, "swing_deg": 0.0, "slip_m": 0.0,
                "success": False, "com0": data.xipos[oid].copy(), "on_object": False}
    z_top = 1.0 - dist
    z0 = max(0.003, z_top - PAD_BELOW_TOP_M)
    for i in (il, ir):
        data.mocap_pos[i][2] = z0
    mujoco.mj_forward(m, data)
    com0 = data.xipos[oid].copy()
    R0 = data.xmat[oid].reshape(3, 3).copy()
    t_lift, t_hold, t_close_max = PHASES["lift"], PHASES["hold"], PHASES["close_max"]
    sl = sr = 0.0  # how far each pad has closed (m)
    v = 0.10       # closing speed (m/s)
    rel0 = None
    t_start = None  # time the lift starts: both pads at full force, or the closing time ran out
    k = 0
    while True:
        t = k * dt
        fl_, fr_ = _pad_force(m, data, bl, oid), _pad_force(m, data, br, oid)
        if t_start is None and ((fl_ >= FORCE_N and fr_ >= FORCE_N) or t >= t_close_max):
            t_start = t + 0.1  # let the squeeze settle
        if sl < PAD["travel"] and fl_ < FORCE_N:
            sl += v * dt
        if sr < PAD["travel"] and fr_ < FORCE_N:
            sr += v * dt
        data.mocap_pos[il][:2] = [x - (c0 - sl) * u[0], y - (c0 - sl) * u[1]]
        data.mocap_pos[ir][:2] = [x + (c0 - sr) * u[0], y + (c0 - sr) * u[1]]
        s_ = 0.0 if t_start is None else np.clip((t - t_start) / t_lift, 0, 1)
        lift = LIFT_M * (3 * s_ * s_ - 2 * s_ ** 3)
        data.mocap_pos[il][2] = data.mocap_pos[ir][2] = z0 + lift
        mujoco.mj_step(m, data)
        if rel0 is None and t_start is not None and t >= t_start:
            rel0 = data.xipos[oid] - data.mocap_pos[il]
        if record is not None:
            record(m, data)
        k += 1
        if t_start is not None and t > t_start + t_lift + t_hold:
            break
    com1 = data.xipos[oid].copy()
    R1 = data.xmat[oid].reshape(3, 3)
    lifted = float(com1[2] - com0[2])
    swing = math.degrees(math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
    slip = float(np.linalg.norm((data.xipos[oid] - data.mocap_pos[il]) - rel0))
    held = bool(np.isfinite(com1).all()) and lifted > 0.8 * LIFT_M and slip < 0.05
    return {"x": x, "y": y, "yaw": yaw_deg, "held": bool(held), "lifted_m": lifted, "swing_deg": swing, "slip_m": slip,
            "success": bool(held and swing < max_swing_deg), "com0": com0, "on_object": True}


_G = {}


def _init(body):
    _G["body"] = body


def _one(a):
    return grasp_test(_G["body"], *a)


def grasp_map(body: Body, extent_xy, step=0.015, yaws=(0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5), procs=8, max_swing_deg=30.0) -> dict:
    xs = np.arange(-extent_xy[0] / 2, extent_xy[0] / 2 + 1e-9, step)
    ys = np.arange(-extent_xy[1] / 2, extent_xy[1] / 2 + 1e-9, step)
    jobs = [(x, y, yaw, max_swing_deg) for y in ys for x in xs for yaw in yaws]
    with Pool(procs, initializer=_init, initargs=(body,)) as p:
        res = p.map(_one, jobs, chunksize=4)
    shape = (len(ys), len(xs), len(yaws))
    ok = np.array([r["success"] for r in res]).reshape(shape)
    swing = np.array([r["swing_deg"] for r in res]).reshape(shape)
    held = np.array([r["held"] for r in res]).reshape(shape)
    return {"xs": xs, "ys": ys, "yaws": list(yaws), "success": ok, "held": held, "swing_deg": swing,
            "score": ok.mean(2), "best_yaw_idx": ok.argmax(2), "com": res[0]["com0"], "step": step}
