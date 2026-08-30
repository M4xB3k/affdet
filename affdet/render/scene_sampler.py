from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

from affdet.intersection_matrix import OBJECT_CLASSES


TABLE_SIZE_M = (1.00, 0.75)
TABLE_HEIGHT_M = 0.0


_FOOTPRINTS_M: dict[str, tuple[float, float]] = {
    "apple":         (0.075, 0.075),
    "banana":        (0.178, 0.109),
    "bowl":          (0.161, 0.161),
    "cutting_board": (0.459, 0.270),
    "knife":         (0.215, 0.021),
    "lemon":         (0.061, 0.059),
    "mug":           (0.117, 0.093),
    "mustard_bottle":(0.097, 0.067),
    "orange":        (0.074, 0.072),
    "peach":         (0.063, 0.062),
    "pear":          (0.100, 0.067),
    "pitcher_base":  (0.149, 0.145),
    "plate":         (0.261, 0.260),
    "plum":          (0.057, 0.055),
    "skillet":       (0.441, 0.267),  # includes handle
    "skillet_lid":   (0.269, 0.268),
    "spatula":       (0.306, 0.125),
    "spoon":         (0.186, 0.083),
    "strawberry":    (0.045, 0.045),
    "sugar_box":     (0.094, 0.049),
}


# 11 templates designed for 100% coverage of all 79 structurally-possible
# affordance triples given available meshes (see tools/docs/shortcuts.md S4).
# Missing objects (pitcher_lid, salt_shaker, pot, pot_lid, tray) excluded — see S3.
_TEMPLATES: list[list[str]] = [
    ["cutting_board", "knife", "plate", "mug", "apple"],                  # T12: canCut+canSupport+canSlideOn (cutting_board as support surface)
    ["skillet", "spoon", "bowl", "mug", "strawberry"],                   # T1: canScoop+canContain+canPourInto+canSlideOn+canSupport
    ["pitcher_base", "bowl", "mug", "spoon", "lemon"],                   # T2: canPourInto+canScoop+canContain
    ["plate", "knife", "banana", "peach", "spoon"],                      # T3: canCut+canSupport (plate first — largest)
    ["skillet", "plate", "mug", "bowl", "plum"],                         # T4: canContain+canPourInto+canSlideOn+canSupport
    ["skillet", "mustard_bottle", "pitcher_base", "sugar_box", "mug"],   # T5: canPourInto+canSlideOn+canSupport
    ["skillet", "knife", "mustard_bottle", "bowl", "mug", "strawberry"], # T6: canCut+canContain+canPourInto+canSlideOn+canSupport
    ["skillet", "plate", "sugar_box", "spoon", "spatula"],                # T7: canScoop+canSlideOn+canSupport
    ["knife", "mustard_bottle", "pitcher_base", "sugar_box", "pear", "mug"],    # T8: canCut+canPourInto+canSupport
    ["plate", "knife", "mustard_bottle", "sugar_box", "orange", "spatula"],     # T9: canCut+canPourInto+canSlideOn+canSupport
    ["skillet", "plate", "knife", "pitcher_base", "sugar_box", "apple", "plum", "strawberry"],  # T10: canCut+canContain+canPourInto+canSupport
    ["skillet", "knife", "sugar_box", "bowl", "lemon"],                  # T11: canCut+canContain+canSupport (closes gaps)
    ["cutting_board", "knife", "plate", "mug", "apple"],                  # T12: canCut+canSupport+canSlideOn (cutting_board as support surface)
]

# Objects that lie flat — only z-rotation (spin), zero x/y tilt.
_FLAT_OBJECTS = {
    "knife", "spoon", "spatula", "banana", "plate",
    "pitcher_lid", "cutting_board", "tray",
}

# Objects that stand upright — also zero x/y tilt (they sit on their base).
_UPRIGHT_OBJECTS = {
    "bowl", "mug", "pitcher_base", "skillet", "sugar_box",
    "mustard_bottle", "apple", "orange", "peach", "pear",
    "plum", "strawberry", "lemon",
}

# Only flat surfaces where smaller objects can realistically rest on top.
# All objects placed separately — no stacking or overlap.
# See tools/docs/shortcuts.md S6.
_OVERLAP_ALLOWED: set[str] = set()


@dataclass
class ObjectPlacement:
    obj_class: str
    mesh_path: str
    position: list[float]
    rotation_euler: list[float]


@dataclass
class SceneConfig:
    objects: list[ObjectPlacement]
    camera_pose: np.ndarray
    camera_fov_deg: float
    n_samples: int = 128
    resolution: tuple = (640, 480)


@dataclass
class SceneSampler:
    seed: int = 0
    n_objects_range: tuple[int, int] = (3, 6)
    table_size: tuple[float, float] = TABLE_SIZE_M
    mesh_dir: Path | str = "data/meshes"
    allow_missing_assets: bool = True
    min_gap_m: float = 0.05

    def __post_init__(self) -> None:
        self._rng = np.random.RandomState(self.seed)
        self._sample_idx = 0
        self.mesh_dir = Path(self.mesh_dir)
        self._available_classes = [
            cls for cls in OBJECT_CLASSES
            if (self.mesh_dir / cls / "model.obj").exists()
        ]

    def sample(self) -> SceneConfig:
        template = _TEMPLATES[self._sample_idx % len(_TEMPLATES)]
        self._sample_idx += 1
        # Use template objects only — no random filler to avoid duplicates and
        # ensure affordance pairs are always present.
        classes = [c for c in template if c in self._available_classes]

        objects = self._sample_object_placements(classes)
        camera_pose = self.sample_camera_pose()
        return SceneConfig(
            objects=objects,
            camera_pose=camera_pose,
            camera_fov_deg=35.0,
            n_samples=256,
            resolution=(960, 720),
        )

    def sample_camera_pose(self) -> np.ndarray:
        az = float(self._rng.uniform(0.0, 2.0 * math.pi))
        el = float(self._rng.uniform(math.radians(55.0), math.radians(70.0)))
        dist = float(self._rng.uniform(1.8, 2.2))
        eye = np.array([
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el),
        ], dtype=np.float64)
        target = np.array([0.0, 0.0, TABLE_HEIGHT_M], dtype=np.float64)
        return _look_at(eye, target)

    def _sample_object_placements(self, classes: list[str]) -> list[ObjectPlacement]:
        table_w, table_d = self.table_size
        margin = 0.08  # keep objects away from table edge so camera doesn't clip them
        placements: list[ObjectPlacement] = []
        placed: list[tuple[float, float, float, float, str]] = []  # x,y,hw,hd,cls
        for cls in classes:
            fw, fd = _FOOTPRINTS_M.get(cls, (0.10, 0.10))
            hw, hd = fw / 2, fd / 2  # half-extents for AABB check
            gap = self.min_gap_m
            placed_pos: tuple[float, float] | None = None
            for _ in range(500):
                x = float(self._rng.uniform(-table_w / 2 + hw + margin, table_w / 2 - hw - margin))
                y = float(self._rng.uniform(-table_d / 2 + hd + margin, table_d / 2 - hd - margin))
                overlap = False
                for px, py, phw, phd, pcls in placed:
                    if cls in _OVERLAP_ALLOWED or pcls in _OVERLAP_ALLOWED:
                        continue
                    # AABB separation check with gap
                    if (abs(x - px) < hw + phw + gap and
                            abs(y - py) < hd + phd + gap):
                        overlap = True
                        break
                if not overlap:
                    placed_pos = (x, y)
                    break
            if placed_pos is None:
                # genuinely doesn't fit — skip this object rather than force-place
                continue
            x, y = placed_pos
            placed.append((x, y, hw, hd, cls))
            if cls in _FLAT_OBJECTS or cls in _UPRIGHT_OBJECTS:
                rx, ry = 0.0, 0.0
            else:
                rx = float(self._rng.uniform(-math.pi / 12, math.pi / 12))
                ry = float(self._rng.uniform(-math.pi / 12, math.pi / 12))
            rz = float(self._rng.uniform(0.0, 2.0 * math.pi))
            placements.append(ObjectPlacement(
                obj_class=cls,
                mesh_path=str(self.mesh_dir / cls / "model.obj"),
                position=[x, y, TABLE_HEIGHT_M],
                rotation_euler=[rx, ry, rz],
            ))
        return placements

    @staticmethod
    def _list_assets(root: Path, suffixes: set[str]) -> list[Path]:
        if not root.exists():
            return []
        return sorted([
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ])

    def _choose_asset(self, assets: list[Path], root: Path, fallback: str) -> str:
        if assets:
            return str(self._rng.choice(assets))
        if self.allow_missing_assets:
            return str(root / fallback)
        raise FileNotFoundError(f"No assets found in {root}")


def _look_at(eye: np.ndarray, target: np.ndarray,
             up: np.ndarray | None = None) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else up
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, -forward], axis=0)
    trans = -rot @ eye
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose
