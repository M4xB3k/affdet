"""MuJoCo prototype: simulation-checked support labels and contact heatmaps.

Objects come from affdet.geometry.prepare (upright, metric, base at the
origin). Collision shapes are a CoACD convex decomposition of the visual mesh,
so concave objects (mugs, pans, trays) keep their cavities. Mass is spread
uniformly over the decomposed volume (a solid-object approximation; hollow
objects need a shell model later).

- rest_test: put an object on a table with a given box face down, let go,
  simulate, and report whether it stays (tilt and drift below thresholds).
- stack_test: put object A on the table and object B centred on A's top
  (the placement UTAMP-2 assumes), simulate, and report whether B stays on A.
  It also returns the contact points between A and B, which become the
  contact heatmap labels.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import trimesh

FACE_DOWN = {  # rotation (axis, angle) that puts this face of the object down
    "under": ([1, 0, 0], 0.0),
    "on": ([1, 0, 0], math.pi),
    "front": ([1, 0, 0], math.pi / 2),
    "back": ([1, 0, 0], -math.pi / 2),
    "left": ([0, 1, 0], -math.pi / 2),
    "right": ([0, 1, 0], math.pi / 2),
}


DISTURBANCE_DEG = 5.0   # tilt a rest must survive; equals the static labeller's Params.min_tip_deg
SETTLE_DEG = 2.0        # the face must return within this angle of its rest pose (a 64-gon rests 2.8 deg off: faceted-round must fail)
LATE_MOTION_M = 0.005   # centre-of-mass travel allowed in the last second (a tall object may sway a few mm)
SETTLE_CANDIDATE_DEG = 20.0  # like the static labeller: a face counts if the object rests on a facet this close to it


def decompose(mesh: trimesh.Trimesh, threshold: float = 0.05, max_hulls: int = 24) -> list[trimesh.Trimesh]:
    """Convex parts (CoACD). Falls back to the convex hull when decomposition fails."""
    import coacd
    coacd.set_log_level("error")
    try:
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold,
                                max_convex_hull=max_hulls, preprocess_mode="auto")
        hulls = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
        hulls = [h for h in hulls if h.volume > 1e-9]
        if hulls:
            return hulls
    except Exception:
        pass
    return [mesh.convex_hull]


@dataclass
class Body:
    name: str
    parts: list  # convex trimesh parts in the object frame
    mass: float
    friction: float = 0.6
    part_mass: list | None = None  # kg per part (sums to `mass`); None spreads `mass` uniformly over the part volumes
    mesh_files: list = field(default_factory=list)

    @property
    def volume(self):
        return sum(p.volume for p in self.parts)


def _write_parts(body: Body, d: str):
    body.mesh_files = []
    for i, p in enumerate(body.parts):
        f = os.path.join(d, f"{body.name}_{i}.obj")
        p.export(f)
        body.mesh_files.append(f)


def _quat(axis, ang):
    axis = np.asarray(axis, float)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis / np.linalg.norm(axis)]


SIM = {"timestep": 0.001, "solref": "0.006 1", "roll": 0.0005, "noslip": 5}  # stiff contacts (the default 0.02 lets multi-part CoACD bodies sink 9 mm into the table) + small rolling friction (m)


def build_xml(bodies: list[Body], poses: list[tuple[np.ndarray, np.ndarray]], timestep=None) -> str:
    timestep = timestep or SIM["timestep"]
    assets, worldbodies = [], []
    for b, (pos, quat) in zip(bodies, poses):
        dens = b.mass / b.volume
        geoms = []
        for i, f in enumerate(b.mesh_files):
            assets.append(f'<mesh name="{b.name}_{i}" file="{f}"/>')
            weight = f'mass="{b.part_mass[i]:.6f}"' if b.part_mass is not None else f'density="{dens:.3f}"'
            geoms.append(f'<geom type="mesh" mesh="{b.name}_{i}" {weight} '
                         f'friction="{b.friction} 0.005 {SIM["roll"]}" condim="6" solref="{SIM["solref"]}" rgba="0.6 0.7 0.9 1"/>')
        worldbodies.append(f'<body name="{b.name}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
                           f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">'
                           f'<freejoint/>{"".join(geoms)}</body>')
    return f"""<mujoco>
  <size memory="256M"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="{SIM["noslip"]}"/>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <geom name="table" type="plane" size="2 2 0.01" friction="0.6 0.005 {SIM["roll"]}" condim="6" solref="{SIM["solref"]}"/>
    {''.join(worldbodies)}
  </worldbody>
</mujoco>"""


def _lowest_z(parts, R):
    return min((p.vertices @ R.T)[:, 2].min() for p in parts)


def _place(body: Body, R: np.ndarray, xy=(0.0, 0.0), z_base=0.0, gap=0.001):
    """Position so the rotated body's lowest point sits `gap` above z_base."""
    z = z_base - _lowest_z(body.parts, R) + gap
    return np.array([xy[0], xy[1], z])


def _tilt_deg(q0, q1):
    import mujoco
    r0 = np.zeros(9); r1 = np.zeros(9)
    mujoco.mju_quat2Mat(r0, q0); mujoco.mju_quat2Mat(r1, q1)
    z0, z1 = r0.reshape(3, 3)[:, 2], r1.reshape(3, 3)[:, 2]
    return math.degrees(math.acos(np.clip(z0 @ z1, -1, 1)))


STABILITY = dict(max_tilt=10.0, max_drift=0.03, max_excess_drop=0.01, max_w=1.5, max_v=0.1, max_release_penetration=0.003)


def placement_stable(n_contacts, tilt_deg, drift_m, excess_drop_m, w_end, v_end, release_penetration_m, **thr):
    """The one definition of "B stays on A" (works on numbers or arrays): still touching A, not tipped more than `max_tilt` degrees
    relative to A, not slid more than `max_drift` m relative to A, not fallen further than expected, not still tumbling (`max_w` rad/s,
    `max_v` m/s relative to A: soft contacts leave ~0.3 rad/s of jitter on a stack at rest), and released without overlapping A.
    Used by `stack_test` and by `placement_map.finalize`, so a single placement and a stored map always agree."""
    t = {**STABILITY, **thr}
    return ((np.asarray(n_contacts) > 0) & (np.asarray(tilt_deg) < t["max_tilt"]) & (np.asarray(drift_m) < t["max_drift"])
            & (np.asarray(excess_drop_m) < t["max_excess_drop"]) & (np.asarray(w_end) < t["max_w"]) & (np.asarray(v_end) < t["max_v"])
            & (np.asarray(release_penetration_m) <= t["max_release_penetration"]))


def lower_until_touching(m, data, body_name: str, z_lo: float, z_hi: float, tol=0.0002) -> float:
    """Lower a free body straight down from z_hi to the lowest height in [z_lo, z_hi] at which it overlaps nothing (bisection on
    its height; overlap is monotone in z), then leave it 0.5 mm above that height. Returns that height.

    This is how a robot puts an object down: descend until the first contact. Shared by `stack_test` and the videos."""
    import mujoco
    bid = m.body(body_name).id
    adr = m.jnt_qposadr[m.body(body_name).jntadr[0]]

    def overlap_at(z):
        data.qpos[adr + 2] = z
        mujoco.mj_forward(m, data)
        return max([-c.dist for c in data.contact[: data.ncon] if bid in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])], default=0.0)

    lo, hi = z_lo, z_hi
    if overlap_at(lo) <= tol:
        z_star = lo                      # nothing in the way down to the table
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if overlap_at(mid) <= tol:
                hi = mid
            else:
                lo = mid
        z_star = hi
    data.qpos[adr + 2] = z_star + 0.0005
    mujoco.mj_forward(m, data)
    return z_star


def _simulate(bodies, poses, seconds=2.0, kick=None, watch=None, window=1.0, info=None, skip_if_penetration=None, descend=None,
              abort=None, skip_on_table=False):
    """Build and step the scene. watch(data) is called every step of the last `window` seconds.

    If `info` is a dict it receives `initial_penetration_m`: how deeply the last body overlaps anything at t = 0.
    With `skip_if_penetration` set, a start deeper than that returns without stepping (the result is already decided).
    With `descend=(z_lo, z_hi)` the last body is first lowered straight down from z_hi to the lowest height in [z_lo, z_hi]
    at which it overlaps nothing (bisection on the body's height), i.e. until it first touches something, as a robot does
    when it puts an object down; it is then released 0.5 mm above that height.

    Speed options (off by default, so stored results are unchanged): `skip_on_table` returns without stepping when the descent
    reached the table, i.e. the body is not over anything; `abort(m, data, q_start, p_start)` is called every 25 steps and stops
    the run when it returns True (`info["aborted_at"]` = seconds simulated)."""
    import mujoco
    with tempfile.TemporaryDirectory() as d:
        for b in bodies:
            _write_parts(b, d)
        m = mujoco.MjModel.from_xml_string(build_xml(bodies, poses))
    data = mujoco.MjData(m)
    if info is not None:
        info["model"] = m
    if kick is not None:  # initial (angular, linear) velocity of the last body
        w, v = kick
        a = m.jnt_dofadr[m.body(bodies[-1].name).jntadr[0]]
        data.qvel[a:a + 3] = v
        data.qvel[a + 3:a + 6] = w
    mujoco.mj_forward(m, data)
    if descend is not None:
        z_star = lower_until_touching(m, data, bodies[-1].name, *descend)
        if info is not None:
            info["descent_m"] = float(descend[1] - z_star)
            info["release_z"] = float(z_star + 0.0005)
    if info is not None:
        last = m.body(bodies[-1].name).id
        pen = [-c.dist for c in data.contact[: data.ncon] if last in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
        info["initial_penetration_m"] = float(max(pen, default=0.0))
        if skip_if_penetration is not None and info["initial_penetration_m"] > skip_if_penetration:
            seconds = 0.0
            info["skipped"] = True
    if skip_on_table and descend is not None and info is not None and info["descent_m"] >= descend[1] - descend[0] - 1e-6:
        seconds = 0.0
        info["on_table"] = True
    q_start = [data.body(b.name).xquat.copy() for b in bodies]
    p_start = [data.body(b.name).xpos.copy() for b in bodies]
    n = int(seconds / m.opt.timestep)
    n_watch = int(window / m.opt.timestep)
    for k in range(n):
        mujoco.mj_step(m, data)
        if abort is not None and k % 25 == 24 and abort(m, data, q_start, p_start):
            if info is not None:
                info["aborted_at"] = (k + 1) * m.opt.timestep
            break
        if watch is not None and k >= n - n_watch:
            watch(data)
    return m, data, q_start, p_start


def settled_margin(body: Body, data, tol=0.0005) -> float:
    """Signed distance (m) from the centre of mass to the edge of the contact patch, measured on the settled pose.

    The patch is every vertex of the collision hulls within `tol` of the table. Positive: the centre of mass is
    over the patch. This is a purely geometric check of what the simulation actually rests on.
    """
    from shapely.geometry import MultiPoint, Point
    Rw, x = data.body(body.name).xmat.reshape(3, 3), data.body(body.name).xpos
    V = np.concatenate([p.vertices for p in body.parts]) @ Rw.T + x
    pts = V[V[:, 2] < V[:, 2].min() + tol][:, :2]
    poly = MultiPoint(pts.tolist()).convex_hull
    c = Point(*data.body(body.name).xipos[:2])
    if poly.area <= 0:
        return float(-poly.distance(c))
    return float(poly.exterior.distance(c) * (1 if poly.contains(c) else -1))


def rest_test(body: Body, face: str, tilt_deg: float = DISTURBANCE_DEG, seed: int = 0, tilt_dir_deg: float | None = None,
              ref_normal=None, settle_deg=SETTLE_DEG, max_drift=LATE_MOTION_M, seconds=3.0, down_dir=None) -> dict:
    """Set the object down on `face` (tilted by tilt_deg about a horizontal axis), release it,
    and watch the last second. It holds if the face normal ends within settle_deg of the
    reference rest direction and the centre of mass has stopped moving.

    ref_normal: the face normal's world direction in the untilted rest pose (from a first,
    untilted drop); defaults to straight down. Tapered bodies rest a few degrees off their
    box face, so the reference pose matters. This is the dynamic version of the static
    tip-angle rule: a flat face rocks back; a round side stays tilted or rolls away.
    """
    from affdet.geometry.face_labels import FACES
    axis, ang = FACE_DOWN[face]
    R = trimesh.transformations.rotation_matrix(ang, axis)[:3, :3]
    n_face = np.asarray(FACES[face], float)
    if down_dir is not None:  # rest on a nearby facet instead of exactly on the box face
        n_face = np.asarray(down_dir, float)
        R = trimesh.geometry.align_vectors(n_face, [0, 0, -1])[:3, :3]
    if tilt_deg:
        if tilt_dir_deg is None:
            ax = np.random.default_rng(seed).normal(size=3); ax[2] = 0
        else:
            a = math.radians(tilt_dir_deg); ax = np.array([math.cos(a), math.sin(a), 0.0])
        R = trimesh.transformations.rotation_matrix(math.radians(tilt_deg), ax)[:3, :3] @ R
    q = trimesh.transformations.quaternion_from_matrix(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    pos = _place(body, R)
    n_obj = n_face
    ref = np.array([0, 0, -1.0]) if ref_normal is None else np.asarray(ref_normal, float)
    offs, coms, last = [], [], {}

    def watch(data):
        n_world = data.body(body.name).xmat.reshape(3, 3) @ n_obj
        offs.append(math.degrees(math.acos(np.clip(n_world @ ref, -1, 1))))
        coms.append(data.body(body.name).xipos[:2].copy())
        last["n"] = n_world

    m, data, _, _ = _simulate([body], [(pos, q)], seconds, watch=watch)
    moving = float(np.linalg.norm(coms[-1] - coms[0]))  # centre-of-mass travel in the last second
    worst = float(max(offs))
    down_off = math.degrees(math.acos(np.clip(-last["n"][2], -1, 1)))
    return {"face": face, "face_off_deg": worst, "late_motion_m": moving, "n_final": last["n"],
            "settled_margin_m": settled_margin(body, data),
            "down_off_deg": down_off, "stable": worst < settle_deg and moving < max_drift}


def _holds_on(body: Body, face: str, down_dir, tilt_deg, directions, max_rest_off_deg, settle_deg, max_drift) -> dict:
    ref = rest_test(body, face, tilt_deg=0, settle_deg=max_rest_off_deg, max_drift=max_drift, down_dir=down_dir)
    if not ref["stable"]:
        return {"holds": False, "rest_off_deg": ref["down_off_deg"], "worst_off_deg": 180.0,
                "settled_margin_m": ref["settled_margin_m"], "pass_frac": 0.0, "per_dir": []}
    res = [rest_test(body, face, tilt_deg=tilt_deg, tilt_dir_deg=d, ref_normal=ref["n_final"], settle_deg=settle_deg,
                     max_drift=max_drift, down_dir=down_dir) for d in directions]
    return {"holds": all(r["stable"] for r in res), "rest_off_deg": ref["down_off_deg"],
            "worst_off_deg": max(r["face_off_deg"] for r in res), "settled_margin_m": ref["settled_margin_m"],
            "pass_frac": sum(r["stable"] for r in res) / len(res),
            "per_dir": [(r["face_off_deg"], r["late_motion_m"]) for r in res]}


def face_holds(body: Body, face: str, tilt_deg=DISTURBANCE_DEG, directions=(0, 45, 90, 135, 180, 225, 270, 315),
               max_rest_off_deg=SETTLE_CANDIDATE_DEG, settle_deg=SETTLE_DEG, max_drift=LATE_MOTION_M,
               max_candidates=4) -> dict:
    """The face holds if the object can rest with that face down, within max_rest_off_deg, and survive a tilt.

    Like the static labeller, the object may settle onto a facet near the box face: the exact face-down pose is
    tried first, then the largest hull facets within max_rest_off_deg of it. A candidate holds if (1) a drop
    settles on it and stops moving and (2) after tilting by tilt_deg in every direction it returns to that rest
    pose. The face holds if any candidate does. Returns the best candidate's metrics (raw, so thresholds can be
    re-applied offline) and `candidate_deg`, the angle between that facet and the box face.
    """
    from affdet.geometry.face_labels import FACES, _resting_directions
    nrm = np.asarray(FACES[face], float)
    hull = trimesh.util.concatenate([p for p in body.parts]).convex_hull
    cands = _resting_directions(hull, nrm, max_rest_off_deg, max_n=max_candidates)[:max_candidates]
    best = None
    for k, d in enumerate(cands):
        r = _holds_on(body, face, None if k == 0 else d, tilt_deg, directions, max_rest_off_deg, settle_deg, max_drift)
        r["candidate_deg"] = math.degrees(math.acos(np.clip(d @ nrm, -1, 1)))
        if best is None or (r["holds"], r["pass_frac"]) > (best["holds"], best["pass_frac"]):
            best = r
        if r["holds"]:
            break
    return {"face": face, "tilt_deg": tilt_deg, "n_candidates_tried": k + 1, **best,
            "worst_late_motion_m": max((m for _, m in best["per_dir"]), default=float("nan"))}


def surface_height(base: Body, Rb, pb, points) -> float:
    """Highest point of `base` (placed with rotation Rb at pb) below any of the world (x, y) `points`."""
    import mujoco
    with tempfile.TemporaryDirectory() as d:
        _write_parts(base, d)
        m = mujoco.MjModel.from_xml_string(build_xml(
            [base], [(pb, trimesh.transformations.quaternion_from_matrix(np.block([[Rb, np.zeros((3, 1))], [np.zeros((1, 3)), 1]])))]))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    gid = np.array([-1], dtype=np.int32)
    best = 0.0
    for x, y in points:
        dist = mujoco.mj_ray(m, data, np.array([x, y, 2.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        if dist >= 0 and gid[0] >= 0 and m.geom_bodyid[gid[0]] == m.body(base.name).id:
            best = max(best, 2.0 - dist)
    return best


def stack_test(base: Body, top: Body, base_face="under", top_face="under", offset=(0.0, 0.0),
               max_tilt=STABILITY["max_tilt"], max_drift=STABILITY["max_drift"], seconds=2.0, release="descend",
               max_release_penetration=STABILITY["max_release_penetration"], fast=None) -> dict:
    """Put `top` at `offset` on `base`'s upward side; does it stay on?

    release="descend" (default): B is held above the offset point and lowered straight down until it first touches
    anything (bisection on its height), then released: a lying shaker over a mug opening comes to rest across the rim,
    a small one goes down to the floor. release="surface": B starts 2 mm above the highest of 9 probe points of A under its
    footprint (kept for comparison; a probe can miss a rim, hence the overlap rule below). release="top": drop B from the
    height of A's highest point (UTAMP-2's centred placement; B falls into anything concave).

    A placement where B already overlaps A by more than `max_release_penetration` at release (a 9 cm shaker laid
    down across a 8 cm mug) cannot be made by a robot, and the solver pushes it out in arbitrary directions: it does
    not count as stable. `seconds=0` runs only this check.

    `fast` (default None = the full run that made the stored labels) is a dict of speed options: `on_table` (skip placements
    where B is not over A at all), `abort_drift_m` / `abort_tilt_deg` (stop once B has clearly left A: drifted or tilted far
    beyond the limits). A cut-short placement is reported with 0 contacts, so the map's finalizer rejects it too.
    """
    import mujoco
    Rb = trimesh.transformations.rotation_matrix(FACE_DOWN[base_face][1], FACE_DOWN[base_face][0])
    Rt = trimesh.transformations.rotation_matrix(FACE_DOWN[top_face][1], FACE_DOWN[top_face][0])
    pb = _place(base, Rb[:3, :3])
    base_top = max((p.vertices @ Rb[:3, :3].T)[:, 2].max() for p in base.parts) + pb[2]
    # centre of the base's top box face; the top object drops onto whatever is there
    z_release = base_top
    if release == "surface":
        ext = np.ptp(np.concatenate([p.vertices for p in top.parts]) @ Rt[:3, :3].T, axis=0)[:2]
        hx, hy = 0.5 * ext
        probe = [(offset[0] + fx * hx, offset[1] + fy * hy)
                 for fx, fy in [(0, 0), (.8, 0), (-.8, 0), (0, .8), (0, -.8), (.6, .6), (-.6, .6), (.6, -.6), (-.6, -.6)]]
        z_release = surface_height(base, Rb[:3, :3], pb, probe)
    descend = None
    if release == "descend":
        z_floor = _place(top, Rt[:3, :3], xy=offset, z_base=0.0, gap=0.0)[2]       # resting on the table
        z_high = _place(top, Rt[:3, :3], xy=offset, z_base=base_top, gap=0.02)[2]    # lowest point 2 cm above A's highest point
        descend = (float(z_floor), float(z_high))
        pt = _place(top, Rt[:3, :3], xy=offset, z_base=base_top, gap=0.02)
    else:
        pt = _place(top, Rt[:3, :3], xy=offset, z_base=z_release, gap=0.002)
    qb = trimesh.transformations.quaternion_from_matrix(Rb)
    qt = trimesh.transformations.quaternion_from_matrix(Rt)
    info = {}
    touching = [0]  # most base-top contacts seen at any step of the last second: one step can show none while B rests on A (soft contacts jitter)
    window_pts = []  # world positions of those contacts, every 20th step (the whole contact patch, not one step's few points)
    step_no = [0]

    def watch(data):
        mm = info["model"]
        ids = {mm.body(base.name).id, mm.body(top.name).id}
        here = [c.pos.copy() for c in data.contact[: data.ncon] if {mm.geom_bodyid[c.geom1], mm.geom_bodyid[c.geom2]} == ids]
        touching[0] = max(touching[0], len(here))
        step_no[0] += 1
        if step_no[0] % 20 == 0:
            window_pts.extend(here)

    fast = fast or {}
    abort = None
    if "abort_drift_m" in fast or "abort_tilt_deg" in fast:
        lim_d, lim_t = fast.get("abort_drift_m", 1e9), fast.get("abort_tilt_deg", 1e9)

        def abort(mm, dd, q_s, p_s):
            # B's motion relative to A, the same measures as the verdict below
            ra, rt = np.zeros(9), np.zeros(9)
            mujoco.mju_quat2Mat(ra, q_s[0]); mujoco.mju_quat2Mat(rt, q_s[1])
            ra, rt = ra.reshape(3, 3), rt.reshape(3, 3)
            xa, xt = dd.body(base.name), dd.body(top.name)
            Ra = xa.xmat.reshape(3, 3)
            tilt_now = math.degrees(math.acos(np.clip((np.trace((ra.T @ rt).T @ (Ra.T @ xt.xmat.reshape(3, 3))) - 1) / 2, -1, 1)))
            d = Ra.T @ (xt.xpos - xa.xpos) - ra.T @ (p_s[1] - p_s[0])
            return tilt_now > lim_t or float(np.linalg.norm(d[:2])) > lim_d
    m, data, q0, p0 = _simulate([base, top], [(pb, qb), (pt, qt)], seconds, info=info, skip_if_penetration=max_release_penetration, watch=watch, descend=descend,
                                abort=abort, skip_on_table=bool(fast.get("on_table")))
    # motion of the top object relative to the base: soft contacts let the whole stack creep a few mm,
    # which says nothing about whether B stays on A
    def _mat(q):
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, q)
        return r.reshape(3, 3)
    Rb0, Rt0 = _mat(q0[0]), _mat(q0[1])
    Rb1, Rt1 = data.body(base.name).xmat.reshape(3, 3), data.body(top.name).xmat.reshape(3, 3)
    rel0, rel1 = Rb0.T @ Rt0, Rb1.T @ Rt1
    tilt = math.degrees(math.acos(np.clip((np.trace(rel0.T @ rel1) - 1) / 2, -1, 1)))
    d_end = Rb1.T @ (data.body(top.name).xpos - data.body(base.name).xpos)
    d_start = Rb0.T @ (p0[1] - p0[0])
    drift = float(np.linalg.norm((d_end - d_start)[:2]))
    drop = float(p0[1][2] - data.body(top.name).xpos[2])  # p0 is the pose after the descent
    base_id, top_id = m.body(base.name).id, m.body(top.name).id
    contacts = []
    for c in data.contact[: data.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if {b1, b2} == {base_id, top_id}:
            contacts.append(c.pos.copy())
    contacts = np.array(contacts).reshape(-1, 3)
    # contact points in each object's own (upright) frame, for heatmaps on the meshes
    def to_local(name, R0):
        # body-local coordinates are the object's own (upright) frame: geoms were built there
        x = data.body(name).xpos; Rw = data.body(name).xmat.reshape(3, 3)
        return (contacts - x) @ Rw
    # height of A's surface under the release point, and how far B should fall to reach it (a cup is deep,
    # a rim is not). B has "stayed" if it ended no lower than that surface allows.
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, data, np.array([offset[0], offset[1], 2.0]), np.array([0.0, 0.0, -1.0]), None, 1,
                         top_id, gid)
    surface_z = 2.0 - dist if dist >= 0 else 0.0
    expected_fall = 0.0 if release == "descend" else float(z_release - surface_z)  # lowered until touching: no fall is expected
    excess_drop = float(drop - expected_fall - 0.002)
    # still moving at the end (a shaker balanced on a rim and slowly tipping off) is not "stays"
    vt, vb = data.body(top.name).cvel, data.body(base.name).cvel
    w_end = float(np.linalg.norm(vt[:3] - vb[:3]))
    v_end = float(np.linalg.norm(vt[3:] - vb[3:]))
    n_touch = max(len(contacts), touching[0])
    if "aborted_at" in info or info.get("on_table"):
        n_touch = 0  # B left A (or was never over it): the verdict is 'does not stay'
    on_top = bool(placement_stable(n_touch, tilt, drift, excess_drop, w_end, v_end, info["initial_penetration_m"], max_tilt=max_tilt,
                                   max_drift=max_drift, max_release_penetration=max_release_penetration))
    poses = {k: (data.body(n).xpos.copy(), data.body(n).xmat.reshape(3, 3).copy(), R0[:3, :3])
             for k, n, R0 in (("base", base.name, Rb), ("top", top.name, Rt))}
    return {"tilt_deg": tilt, "drift_m": drift, "drop_m": drop, "n_contacts": n_touch, "aborted_at": info.get("aborted_at"), "on_table": info.get("on_table", False), "poses": poses,
            "stable": bool(on_top), "release_penetration_m": info["initial_penetration_m"], "expected_fall_m": expected_fall, "excess_drop_m": excess_drop, "w_end": w_end, "v_end": v_end,
            "surface_depth_m": float(base_top - (np.concatenate([p.vertices for p in top.parts]) @ data.body(top.name).xmat.reshape(3, 3).T + data.body(top.name).xpos)[:, 2].min()) if release == "descend" else float(base_top - surface_z),
            "release_z": info.get("release_z"), "descent_m": info.get("descent_m"),
            "contacts_window_world": np.array(window_pts).reshape(-1, 3),
            "contacts_window_base": _compact(window_pts, data.body(base.name)), "contacts_window_top": _compact(window_pts, data.body(top.name)), "contacts_world": contacts,
            "contacts_base": to_local(base.name, Rb) if len(contacts) else contacts,
            "contacts_top": to_local(top.name, Rt) if len(contacts) else contacts}


def _compact(world_pts, body, cell=0.002, cap=64) -> np.ndarray:
    """The contact points collected over the last second, in `body`'s own frame, merged to one per 2 mm cell and capped (float16)."""
    if not len(world_pts):
        return np.zeros((0, 3), np.float16)
    local = (np.asarray(world_pts) - body.xpos) @ body.xmat.reshape(3, 3)
    _, keep = np.unique(np.round(local / cell).astype(np.int64), axis=0, return_index=True)
    local = local[np.sort(keep)]
    if len(local) > cap:
        local = local[np.linspace(0, len(local) - 1, cap).astype(int)]
    return local.astype(np.float16)


def heat_on_mesh(mesh: trimesh.Trimesh, points: np.ndarray, radius=0.008) -> np.ndarray:
    """Per-vertex heat in [0, 1]: Gaussian falloff from the nearest contact point."""
    if len(points) == 0:
        return np.zeros(len(mesh.vertices))
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points).query(mesh.vertices)
    return np.exp(-0.5 * (d / radius) ** 2)


def contact_region_heat(mesh_a: trimesh.Trimesh, pose_a, mesh_b: trimesh.Trimesh, pose_b,
                        eps=0.003, max_edge=0.004):
    """Heat on A's surface where it touches B in the settled pose (distance to B below eps).

    MuJoCo reports only a few contact points per convex pair (3 for flat on flat), so the
    label uses proximity instead: A is subdivided to max_edge and every vertex within eps of
    B's surface is hot. pose = (xpos, xmat) of each body after the simulation.
    Returns (subdivided A mesh in its own frame, per-vertex heat in [0, 1]).
    """
    v, f = trimesh.remesh.subdivide_to_size(mesh_a.vertices, mesh_a.faces, max_edge=max_edge, max_iter=12)
    a = trimesh.Trimesh(v, f, process=False)
    xa, Ra = pose_a[0], pose_a[1]
    xb, Rb = pose_b[0], pose_b[1]
    pts_world = a.vertices @ Ra.T + xa
    b_world = mesh_b.copy()
    b_world.apply_transform(np.block([[Rb, xb[:, None]], [np.zeros((1, 3)), 1]]))
    # only vertices near B's bounding box need an exact distance
    lo, hi = b_world.bounds
    near = np.all((pts_world > lo - 2 * eps) & (pts_world < hi + 2 * eps), axis=1)
    d = np.full(len(pts_world), np.inf)
    if near.any():
        _, dist, _ = trimesh.proximity.closest_point(b_world, pts_world[near])
        d[near] = dist
    heat = np.clip(1 - d / eps, 0, 1)
    return a, heat
