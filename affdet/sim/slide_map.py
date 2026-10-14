"""canSlide: where on an object can a pusher push it so that it slides across the table (and does not tip or spin)?

A flat pusher approaches the object from outside at azimuth `a` (degrees around the object's centre) and height
`frac` (fraction of the object's height), and pushes it horizontally toward the centre for `push_m` at `speed`.
Outcome (measured on the object after the push):

- `moved_frac`: how far the object travelled along the push, as a fraction of the pusher's travel (1 = it slid with
  the pusher, 0 = it did not move, the pusher slipped past or over it).
- `tilt_deg`: how far its up axis rotated (a tall object pushed high topples).
- `yaw_deg`: how far it turned about the vertical (a push off the centre of friction spins it).
- success = moved_frac >= 0.8, tilt < 10 deg, yaw < 25 deg.

The map is over (azimuth, height); `slide_labels` paints it on the object's side surface. Like the grasp test, the pusher is a
heavy body welded to a mocap target (a mocap body alone has zero velocity as far as contacts are concerned).
"""
from __future__ import annotations

import math
import tempfile
from multiprocessing import Pool

import numpy as np

import affdet.sim.mujoco_support as ms
from affdet.sim.mujoco_support import Body

PUSHER = dict(half=(0.004, 0.025, 0.010), mass=5.0, friction=(0.6, 0.01, 0.001))  # x is the push direction: a 2 cm tall, 5 cm wide plate
SETTLE_S = 1.0    # the object is let go from its upright pose and settles first: a round-bottomed bowl is not at rest upright
PUSH_M = 0.12
SPEED = 0.05      # m/s
LEAD_M = 0.03     # the pusher starts this far from the surface
MOVED_MIN, TILT_MAX, YAW_MAX = 0.8, 10.0, 25.0


def _pusher_xml(x, y, z, yaw, hz=None):
    hx, hy, hz0 = PUSHER["half"]
    hz = hz0 if hz is None else hz
    q = f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"
    f = PUSHER["friction"]
    return (f'<body name="tpush" mocap="true" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{q}"/>'
            f'<body name="push" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{q}"><freejoint/>'
            f'<geom type="box" size="{hx} {hy} {hz}" friction="{f[0]} {f[1]} {f[2]}" condim="6" mass="{PUSHER["mass"]}" '
            f'solref="{ms.SIM["solref"]}" rgba="0.8 0.2 0.2 1"/></body>')


def slide_test(body: Body, azimuth_deg: float, frac: float, push_m=PUSH_M, speed=SPEED, record=None, xml_hook=None) -> dict:
    import mujoco
    a = math.radians(azimuth_deg)
    out = np.array([math.cos(a), math.sin(a)])  # from the object toward the pusher
    pos = ms._place(body, np.eye(3))
    z_lo = float(min(p.vertices[:, 2].min() for p in body.parts)); z_hi = float(max(p.vertices[:, 2].max() for p in body.parts))
    hz = min(PUSHER["half"][2], 0.4 * (z_hi - z_lo))  # a pusher no taller than the object (a plate is 1 cm high)
    with tempfile.TemporaryDirectory() as d:
        ms._write_parts(body, d)
        xml = ms.build_xml([body], [(pos, np.array([1.0, 0, 0, 0]))])
        xml = xml.replace("</worldbody>", _pusher_xml(0.0, 0.0, 2.0, a + math.pi, hz) + "</worldbody>")  # parked above while the object settles
        xml = xml.replace("</mujoco>", '<equality><weld body1="push" body2="tpush" solref="0.008 1"/></equality></mujoco>')
        if xml_hook is not None:
            xml = xml_hook(xml)
        m = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    oid = m.body(body.name).id
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, data)
    Rs, ps = data.xmat[oid].reshape(3, 3).copy(), data.xpos[oid].copy()
    verts = np.concatenate([p.vertices for p in body.parts]) @ Rs.T + ps
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    z = max(zmin + frac * (zmax - zmin), hz + 0.001)  # the pusher centre stays above the table
    com0 = data.xipos[oid].copy()
    # where the pusher's line meets the object: cast from far outside toward the centre of mass at the push height
    gid = np.array([-1], dtype=np.int32)
    origin = np.array([com0[0] + out[0], com0[1] + out[1], z])
    dist = mujoco.mj_ray(m, data, origin, np.array([-out[0], -out[1], 0.0]), None, 1, m.body("push").id, gid)
    if dist < 0 or m.geom_bodyid[gid[0]] != oid:
        return {"azimuth": azimuth_deg, "frac": frac, "on_object": False, "success": False, "moved_frac": 0.0, "tilt_deg": 0.0, "yaw_deg": 0.0}
    hit = origin[:2] - out * dist
    hx = PUSHER["half"][0]
    start = hit + out * (LEAD_M + hx)
    qa = a + math.pi  # pusher x axis points toward the object
    jid = m.body("push").jntadr[0]
    qadr = m.jnt_qposadr[jid]
    data.qpos[qadr:qadr + 3] = [start[0], start[1], z]
    data.qpos[qadr + 3:qadr + 7] = [math.cos(qa / 2), 0, 0, math.sin(qa / 2)]
    data.mocap_pos[0] = [start[0], start[1], z]
    data.mocap_quat[0] = [math.cos(qa / 2), 0, 0, math.sin(qa / 2)]
    mujoco.mj_forward(m, data)
    R0 = data.xmat[oid].reshape(3, 3).copy()
    com_start = data.xipos[oid].copy()
    dt = m.opt.timestep
    travel = LEAD_M + push_m
    steps = int((travel / speed + 0.8) / dt)
    for k in range(steps):
        s = min(speed * k * dt, travel)
        p = start - out * s
        data.mocap_pos[0] = [p[0], p[1], z]
        mujoco.mj_step(m, data)
        if record is not None and k % 16 == 0:
            record(m, data)
    R1 = data.xmat[oid].reshape(3, 3)
    com1 = data.xipos[oid].copy()
    moved = float(np.dot((com1 - com_start)[:2], -out))
    tilt = math.degrees(math.acos(np.clip(R0[:, 2] @ R1[:, 2], -1, 1)))
    yaw = math.degrees(abs(math.atan2((R0.T @ R1)[1, 0], (R0.T @ R1)[0, 0])))
    mf = moved / push_m
    ok = bool(np.isfinite(com1).all() and mf >= MOVED_MIN and tilt < TILT_MAX and yaw < YAW_MAX)
    return {"azimuth": azimuth_deg, "frac": frac, "on_object": True, "moved_frac": float(mf), "tilt_deg": float(tilt),
            "yaw_deg": float(yaw), "success": ok, "hit_xy": hit.tolist(), "z": z}


_G = {}


def _init(body):
    _G["body"] = body


def _one(a):
    return slide_test(_G["body"], *a)


def slide_map(body: Body, azimuths=tuple(range(0, 360, 30)), fracs=(0.15, 0.4, 0.65, 0.9), procs=8) -> dict:
    jobs = [(az, f) for f in fracs for az in azimuths]
    with Pool(procs, initializer=_init, initargs=(body,)) as p:
        res = p.map(_one, jobs, chunksize=2)
    shape = (len(fracs), len(azimuths))
    return {"azimuths": list(azimuths), "fracs": list(fracs),
            "success": np.array([r["success"] for r in res]).reshape(shape),
            "moved_frac": np.array([r["moved_frac"] for r in res]).reshape(shape),
            "tilt_deg": np.array([r["tilt_deg"] for r in res]).reshape(shape),
            "yaw_deg": np.array([r["yaw_deg"] for r in res]).reshape(shape),
            "on_object": np.array([r["on_object"] for r in res]).reshape(shape)}
