"""Pydantic v2 schemas for all on-disk artifacts in the affdet pipeline.

Every module reads and writes typed artifacts validated by these schemas. The
schemas are the module boundary contracts.
"""
from __future__ import annotations
from typing import Literal
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Scene metadata (written by render/, read by extract/ and label/)
# ---------------------------------------------------------------------------

class GTAttrs(BaseModel):
    """Ground-truth object attributes (present for Blender renders only)."""
    model_config = ConfigDict(extra="forbid")
    members: list[Literal["body", "handle", "edge"]] = Field(
        ..., description="Members present on this object instance"
    )
    bbox_size: list[float] = Field(
        ..., min_length=3, max_length=3,
        description="Metric bounding-box size [w, h, d] in metres"
    )
    weight_kg: float = Field(..., gt=0)
    edge_type: Literal["serrated", "smooth", "none"] | None = None


class SceneObject(BaseModel):
    """One detected (or rendered) object in a scene."""
    model_config = ConfigDict(extra="forbid")
    obj_id: str = Field(..., description="Unique per-scene identifier")
    obj_class: str = Field(..., description="Taxonomy class name (e.g. 'knife')")
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Pixel-space bbox [x1, y1, x2, y2]"
    )
    mask_path: str | None = Field(
        default=None,
        description="Relative path to instance mask PNG (Blender renders only)"
    )
    gt_attrs: GTAttrs | None = None
    # Post-physics render annotations (omni pipeline).
    visible: bool = Field(
        default=True,
        description="Object occupies >= MIN_VISIBLE_FRAC of the frame (detectable)"
    )
    visible_px: int = Field(
        default=0, description="Visible pixel count in the final render"
    )
    not_upright: bool = Field(
        default=False,
        description="Container settled not opening-up (upside-down or on its side)"
    )


class Camera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose: list[list[float]] = Field(..., description="4x4 world-to-camera pose")
    intrinsics: list[list[float]] = Field(..., description="3x3 intrinsics matrix")


class Scene(BaseModel):
    """Top-level scene artifact written by render/, read by extract/ and label/."""
    # Render/sampler bookkeeping keys (template, instance_ids, source_worker,
    # not_upright_containers, realised_affordances) are attached to the JSON but
    # not core scene data — ignore rather than forbid so readers stay robust.
    model_config = ConfigDict(extra="ignore")
    scene_id: str
    image_path: str = Field(..., description="Relative path to RGB image")
    image_size: list[int] = Field(..., min_length=2, max_length=2,
                                   description="[width, height] in pixels")
    source: Literal["blender", "real"]
    objects: list[SceneObject]
    camera: Camera


# ---------------------------------------------------------------------------
# Features (written by extract/, read by model/)
# ---------------------------------------------------------------------------

class Features(BaseModel):
    """768-d frozen-DINOv2 features per object in a scene.

    Stored as an NPZ for tensor efficiency; not a JSON artifact.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    obj_ids: list[str]
    features: np.ndarray  # shape (N, 768), float32

    def model_post_init(self, _):
        if len(self.obj_ids) != self.features.shape[0]:
            raise ValueError(
                f"length mismatch: {len(self.obj_ids)} obj_ids vs "
                f"{self.features.shape[0]} feature rows"
            )

    def save(self, path: Path | str) -> None:
        np.savez(path, obj_ids=np.array(self.obj_ids), features=self.features)

    @classmethod
    def load(cls, path: Path | str) -> "Features":
        npz = np.load(path, allow_pickle=False)
        return cls(obj_ids=list(npz["obj_ids"]), features=npz["features"])


class FeaturesMeta(BaseModel):
    """Sidecar metadata for features.npz: which bbox source, which backbone."""
    model_config = ConfigDict(extra="forbid")
    bbox_source: Literal["oracle", "yolo"]
    dinov2_variant: Literal["vitb14", "vits14", "vitl14"] = "vitb14"
    extractor_version: str = "v1"


# ---------------------------------------------------------------------------
# Labels (written by label/, read by model/ and eval/)
# ---------------------------------------------------------------------------

TIER1_PREDICATES = Literal[
    "canCut", "canPierce", "canPeel", "canContain", "canSupport", "canPourInto",
]


class LabelEntry(BaseModel):
    """One per-pair Tier-1 label."""
    model_config = ConfigDict(extra="forbid")
    actor: str
    recipient: str
    predicate: TIER1_PREDICATES
    verdict: Literal[0, 1]
    actor_part: Literal["body", "handle", "edge"] | None = Field(
        default=None,
        description="None for whole-object affordances (canContain, canFitOnto)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(..., description="e.g. 'vlm_gpt4o_v1', 'mock', 'human'")
    score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Soft distillation target = VLM score/9; None for binary labels")
    rationale: str | None = None


class Labels(BaseModel):
    """Per-scene labels artifact."""
    model_config = ConfigDict(extra="forbid")
    scene_id: str
    tier1: list[LabelEntry]
    tier2_placeholder: None = None  # reserved for Tier 2 when added back


# ---------------------------------------------------------------------------
# Predictions (written by model/, read by adapter/ and eval/)
# ---------------------------------------------------------------------------

class PredictionEntry(BaseModel):
    """One per-pair Tier-1 prediction with model scores."""
    model_config = ConfigDict(extra="forbid")
    actor: str
    recipient: str
    predicate: TIER1_PREDICATES
    verdict: Literal[0, 1]
    actor_part: Literal["body", "handle", "edge"] | None = None
    score: float = Field(..., ge=0.0, le=1.0)
    actor_part_logits: dict[str, float] | None = None


class Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str
    tier1: list[PredictionEntry]
    tier2_placeholder: None = None
