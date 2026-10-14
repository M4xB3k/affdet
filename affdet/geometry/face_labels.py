"""Per-face UTAMP-2 affordance labels computed from an object's mesh.

UTAMP-2 states its predicates over a member's bounding-box faces ("parts"):
on (+z), under (-z), front (-y), back (+y), left (-x), right (+x), in the
object's upright frame (z up; glTF models are converted from +Y up, and the
glTF front, +Z, becomes -y). The labels here are computed per object from
its geometry, never per category, so two shakers that differ only in their
cap get different `isSupportable on` labels.

For each face:

- **isSupportable(face)**: the object rests stably on a plane with that face
  down. The support polygon is the convex hull of the contact region
  (convex-hull facets within `flat_angle_deg` of the resting direction and
  within `contact_tol` of the extreme plane). The object is stable if the
  centre of mass projects inside the polygon at least `min_margin` from its
  edge and tipping it over needs at least `min_tip_deg` (atan(margin / COM
  height)). Put down with the face roughly down, an object settles onto the
  nearest stable facet, so resting directions up to `settle_deg` from the
  face normal count (a jar whose side is two facets meeting at a shallow
  ridge lies on either facet). A round body lying on its side only touches
  along a line, so it fails; a faceted one passes. A low-poly body the
  artist shaded smooth is meant to be round but passes on geometry, so the
  contact faces' vertex-normal deviation is reported (`smooth_deg`), and a
  narrow contact (a few facets, not a flat base) that is shaded round gets
  `review=True` instead of a silent label change.
- **isSupportive(face)**: something can be put on the face: the flat region
  on it is at least `min_support_width` across in both directions, and the
  object stands stably on the opposite face.
- **canGraspFrom(body, palm, f1, f2)**: a parallel-jaw gripper approaching
  with its palm against `palm` and fingers on the opposite faces f1/f2. All
  material within `finger_depth` of the palm face must fit between the open
  fingers (extent along the closing axis <= `max_opening`), and the object
  must be at least `min_width` thick there. Grasps from every face are
  listed; whether a face is reachable in a given pose is the planner's job.
- **canLift(body)**: mass <= payload.

The centre of mass is that of the solid convex hull (uniform density).
Objaverse meshes are rarely watertight, and a filled shaker is close to
solid, so this is a fair approximation for compact objects; hollow objects
(mugs, pans) will need a better estimate.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Point

# face name -> outward normal in the upright object frame
FACES = {
    "on": (0, 0, 1),
    "under": (0, 0, -1),
    "front": (0, -1, 0),
    "back": (0, 1, 0),
    "left": (-1, 0, 0),
    "right": (1, 0, 0),
}
OPPOSITE = {"on": "under", "under": "on", "front": "back", "back": "front", "left": "right", "right": "left"}


@dataclass
class Params:
    # support / stability
    flat_angle_deg: float = 3.0      # hull facets this close to the face normal count as contact
    contact_tol: float = 0.001       # m; and lie this close to the extreme plane
    min_tip_deg: float = 5.0         # tilt needed to tip the object over
    min_margin: float = 0.005        # m; COM distance inside the support polygon edge
    settle_deg: float = 20.0         # resting directions this far from the face normal count
    smooth_review_deg: float = 8.0   # narrow supportable contacts shaded smoother than this get review
    narrow_contact: float = 0.025    # m; contact regions narrower than this rest on a few facets
    min_support_width: float = 0.03  # m; flat region needed to put something on a face
    # grasp (Franka Hand defaults)
    max_opening: float = 0.08        # m
    finger_depth: float = 0.04       # m; how far the fingers reach past the palm face
    min_width: float = 0.005         # m
    payload: float = 3.0             # kg
    n_samples: int = 30000           # surface samples for grasp widths


@dataclass
class FaceLabel:
    face: str
    box_dims: tuple[float, float]           # extents of the box face (m)
    support_area: float                     # flat contact region area when resting on this face (m^2)
    support_min_width: float                # its narrowest width (m)
    com_margin: float                       # signed COM distance to the support polygon edge (m)
    tip_deg: float                          # tilt needed to tip over (negative: falls immediately)
    settle_deg: float                       # angle between the face normal and the best resting direction
    smooth_deg: float                       # mean vertex-normal deviation of the mesh faces in contact
    review: bool                            # supportable on geometry, but the contact surface is shaded round
    supportable: bool
    supportive: bool


@dataclass
class GraspLabel:
    palm: str
    finger1: str
    finger2: str
    width: float      # extent along the closing axis within finger reach (m)
    graspable: bool


@dataclass
class ObjectLabels:
    uid: str
    extents: tuple[float, float, float]
    mass: float | None
    com: tuple[float, float, float]
    faces: list[FaceLabel] = field(default_factory=list)
    grasps: list[GraspLabel] = field(default_factory=list)
    can_lift: bool | None = None

    def predicates(self, obj: str, member: str = "body") -> list[str]:
        """UTAMP-2 facts for a single-member object."""
        out = []
        for f in self.faces:
            if f.supportable:
                out.append(f"(isSupportable {obj} {f.face})")
            if f.supportive:
                out.append(f"(isSupportive {obj} {f.face})")
        for g in self.grasps:
            if g.graspable:
                out.append(f"(canGraspFrom {obj} {member} {g.palm} {g.finger1} {g.finger2})")
        if self.can_lift:
            out.append(f"(canLift {obj} {member})")
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _frame(normal: np.ndarray) -> np.ndarray:
    """Rows: two in-plane axes and the normal (right-handed)."""
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u), n])


def support_polygon(hull: trimesh.Trimesh, normal, p: Params):
    """Flat contact region of the convex hull on its extreme plane along `normal`.

    Returns (polygon in the frame's 2D coordinates, frame, extreme plane offset).
    """
    R = _frame(np.asarray(normal, float))
    n = R[2]
    d = hull.vertices @ n
    top = d.max()
    cos_t = math.cos(math.radians(p.flat_angle_deg))
    pts = [hull.vertices[d >= top - 1e-6]]  # the extreme point(s) always touch
    for fi in np.nonzero(hull.face_normals @ n >= cos_t)[0]:
        tri = hull.vertices[hull.faces[fi]]
        if (tri @ n >= top - p.contact_tol).all():
            pts.append(tri)
    pts = np.concatenate(pts)
    poly = MultiPoint((pts @ R[:2].T).tolist()).convex_hull
    return poly, R, top


def _min_width(poly) -> float:
    if poly.area <= 0:
        return 0.0
    rect = poly.minimum_rotated_rectangle
    xy = np.asarray(rect.exterior.coords)[:4]
    sides = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    return float(sides.min())


def _resting_directions(hull: trimesh.Trimesh, normal: np.ndarray, settle_deg: float, max_n: int = 40):
    """The face normal plus hull facet normals within settle_deg of it, largest facets first."""
    cos_s = math.cos(math.radians(settle_deg))
    idx = np.nonzero(hull.face_normals @ normal >= cos_s)[0]
    idx = idx[np.argsort(-hull.area_faces[idx])]
    dirs = [normal]
    for n in hull.face_normals[idx]:
        if all(n @ d < math.cos(math.radians(2)) for d in dirs):
            dirs.append(n)
            if len(dirs) >= max_n:
                break
    return dirs


def _stability(poly, R, top, com):
    """(signed COM margin to the support polygon edge, tip angle in degrees)."""
    c2 = Point(*(com @ R[:2].T))
    h = top - com @ R[2]  # COM height above the plane the object rests on
    if poly.area > 0:
        margin = poly.exterior.distance(c2) * (1 if poly.contains(c2) else -1)
    else:
        margin = -poly.distance(c2) if not poly.is_empty else -np.inf
    return float(margin), math.degrees(math.atan2(margin, h))
def _contact_smoothness(mesh: trimesh.Trimesh, d: np.ndarray, p: Params) -> float:
    """Area-weighted mean angle (deg) between vertex and face normals of the mesh faces
    touching the plane when resting along d. Flat-shaded facets give ~0."""
    fn = mesh.face_normals
    tri = mesh.vertices[mesh.faces]
    h = tri @ d
    near = (fn @ d >= math.cos(math.radians(p.settle_deg))) & (h.min(1) >= h.max() - p.contact_tol)
    if not near.any():
        return 0.0
    vn = mesh.vertex_normals[mesh.faces[near]]
    ang = np.degrees(np.arccos(np.clip(np.einsum("kij,kj->ki", vn, fn[near]), -1, 1))).mean(1)
    return float(np.average(ang, weights=mesh.area_faces[near] + 1e-12))


def label_faces(mesh: trimesh.Trimesh, p: Params = Params(), mass: float | None = None, uid: str = "") -> ObjectLabels:
    """Label an upright, metric-scale mesh (z up, frame as in FACES)."""
    hull = mesh.convex_hull
    com = hull.center_mass if hull.is_volume else hull.centroid
    lo, hi = mesh.bounds
    ext = hi - lo
    labels = ObjectLabels(uid=uid, extents=tuple(map(float, ext)), mass=mass, com=tuple(map(float, com)))

    supportable = {}
    flat = {}
    for name, nrm in FACES.items():
        nrm = np.asarray(nrm, float)
        best = None
        for d in _resting_directions(hull, nrm, p.settle_deg):
            poly, R, top = support_polygon(hull, d, p)
            margin, tip = _stability(poly, R, top, com)
            score = min(margin / p.min_margin, tip / p.min_tip_deg)
            if best is None or score > best[0]:
                best = (score, margin, tip, d, _min_width(poly))
        _, margin, tip, d, rest_width = best
        poly, _, _ = support_polygon(hull, nrm, p)  # flat region of the face itself
        supportable[name] = bool(margin >= p.min_margin and tip >= p.min_tip_deg)
        smooth = _contact_smoothness(mesh, d, p)
        flat[name] = _min_width(poly)
        axes = [i for i in range(3) if nrm[i] == 0]
        labels.faces.append(FaceLabel(
            face=name, box_dims=(float(ext[axes[0]]), float(ext[axes[1]])),
            support_area=float(poly.area), support_min_width=flat[name],
            com_margin=float(margin), tip_deg=float(tip),
            settle_deg=float(math.degrees(math.acos(np.clip(d @ nrm, -1, 1)))),
            smooth_deg=smooth, review=bool(supportable[name] and smooth > p.smooth_review_deg and rest_width < p.narrow_contact),
            supportable=supportable[name], supportive=False))
    for f in labels.faces:
        f.supportive = bool(flat[f.face] >= p.min_support_width and supportable[OPPOSITE[f.face]])

    pts, _ = trimesh.sample.sample_surface(mesh, p.n_samples, seed=0)
    pts = np.vstack([pts, mesh.vertices])
    for palm, nrm in FACES.items():
        nrm = np.asarray(nrm, float)
        d = pts @ nrm
        slab = pts[d >= d.max() - p.finger_depth]
        for f1, f2 in (("left", "right"), ("front", "back"), ("under", "on")):
            if palm in (f1, f2):
                continue
            axis = np.asarray(FACES[f2], float)
            s = slab @ axis
            width = float(s.max() - s.min())
            labels.grasps.append(GraspLabel(palm, f1, f2, width, p.min_width <= width <= p.max_opening))
    if mass is not None:
        labels.can_lift = mass <= p.payload
    return labels
