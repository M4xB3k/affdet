import json
import pytest
from pydantic import ValidationError
from affdet.schemas import Scene, SceneObject, Camera, GTAttrs


def test_scene_object_minimal_real_image():
    obj = SceneObject(
        obj_id="knife_0",
        obj_class="knife",
        bbox=[10.0, 20.0, 100.0, 200.0],
    )
    assert obj.obj_id == "knife_0"
    assert obj.mask_path is None
    assert obj.gt_attrs is None


def test_scene_object_blender_with_gt():
    obj = SceneObject(
        obj_id="knife_0",
        obj_class="knife",
        bbox=[10.0, 20.0, 100.0, 200.0],
        mask_path="scene_0001_knife_0.png",
        gt_attrs=GTAttrs(
            members=["body", "handle", "edge"],
            bbox_size=[0.20, 0.03, 0.02],
            weight_kg=0.15,
            edge_type="serrated",
        ),
    )
    assert obj.gt_attrs.edge_type == "serrated"


def test_scene_rejects_bad_bbox():
    with pytest.raises(ValidationError):
        SceneObject(obj_id="k", obj_class="knife", bbox=[1.0, 2.0, 3.0])  # too short


def test_scene_round_trip_json(tmp_path):
    scene = Scene(
        scene_id="scene_0001",
        image_path="scene_0001.png",
        image_size=[1280, 960],
        source="blender",
        objects=[
            SceneObject(obj_id="k_0", obj_class="knife", bbox=[1.0, 2.0, 3.0, 4.0]),
        ],
        camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                      intrinsics=[[500,0,640],[0,500,480],[0,0,1]]),
    )
    p = tmp_path / "scene.json"
    p.write_text(scene.model_dump_json())
    loaded = Scene.model_validate_json(p.read_text())
    assert loaded == scene


# ---------------------------------------------------------------------------
# Features tests (Tasks 3)
# ---------------------------------------------------------------------------
import numpy as np
from affdet.schemas import Features, FeaturesMeta


def test_features_round_trip_npz(tmp_path):
    feats = Features(
        obj_ids=["knife_0", "banana_0"],
        features=np.random.RandomState(0).randn(2, 768).astype(np.float32),
    )
    npz_path = tmp_path / "features.npz"
    feats.save(npz_path)
    loaded = Features.load(npz_path)
    assert loaded.obj_ids == ["knife_0", "banana_0"]
    np.testing.assert_array_equal(loaded.features, feats.features)


def test_features_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        Features(
            obj_ids=["a", "b", "c"],
            features=np.zeros((2, 768), dtype=np.float32),
        )


def test_features_meta_json(tmp_path):
    meta = FeaturesMeta(bbox_source="oracle", dinov2_variant="vitb14", extractor_version="v1")
    p = tmp_path / "features.meta.json"
    p.write_text(meta.model_dump_json())
    loaded = FeaturesMeta.model_validate_json(p.read_text())
    assert loaded.bbox_source == "oracle"


# ---------------------------------------------------------------------------
# Labels + Predictions tests (Task 4)
# ---------------------------------------------------------------------------
from affdet.schemas import Labels, LabelEntry, Predictions, PredictionEntry


def test_label_entry_minimal():
    entry = LabelEntry(
        actor="knife_0", recipient="banana_0", predicate="canCut",
        verdict=1, actor_part="edge", confidence=0.95,
        source="vlm_gpt4o_v1", rationale="serrated blade visible",
    )
    assert entry.verdict == 1


def test_label_entry_whole_object_no_actor_part():
    entry = LabelEntry(
        actor="bowl_0", recipient="apple_0", predicate="canContain",
        verdict=1, actor_part=None, confidence=0.9, source="vlm_gpt4o_v1",
    )
    assert entry.actor_part is None


def test_labels_round_trip(tmp_path):
    labels = Labels(
        scene_id="scene_0001",
        tier1=[
            LabelEntry(actor="k_0", recipient="b_0", predicate="canCut",
                       verdict=1, actor_part="edge", confidence=0.9,
                       source="mock"),
        ],
        tier2_placeholder=None,
    )
    p = tmp_path / "labels.json"
    p.write_text(labels.model_dump_json(indent=2))
    loaded = Labels.model_validate_json(p.read_text())
    assert loaded == labels


def test_prediction_entry_has_score_and_logits():
    entry = PredictionEntry(
        actor="k_0", recipient="b_0", predicate="canCut",
        verdict=1, actor_part="edge", score=0.87,
        actor_part_logits={"edge": 0.81, "body": 0.12, "handle": 0.07},
    )
    assert entry.score == 0.87
    assert entry.actor_part_logits["edge"] > 0.5
