"""Tests for affdet.detect.scene_from_predictions — build a synthetic Scene
directly from raw YOLO detections on an arbitrary photo (no gold objects to
match against, unlike yolo_boxes.rewrite_scene which needs a pre-existing
Scene with gold obj_ids)."""
from affdet.schemas import Scene
from affdet.detect.scene_from_predictions import scene_from_predictions


def _preds():
    return [
        {"bbox": [10.0, 10.0, 50.0, 90.0], "cls": "knife", "conf": 0.91},
        {"bbox": [60.0, 20.0, 140.0, 100.0], "cls": "apple", "conf": 0.77},
    ]


def test_builds_scene_with_one_object_per_prediction():
    scene = scene_from_predictions(
        scene_id="live_0001", image_path="photo.jpg",
        image_size=(200, 150), preds=_preds(),
    )
    assert isinstance(scene, Scene)
    assert scene.scene_id == "live_0001"
    assert scene.image_path == "photo.jpg"
    assert scene.image_size == [200, 150]
    assert scene.source == "real"
    assert len(scene.objects) == 2


def test_object_classes_and_bboxes_carried_through():
    scene = scene_from_predictions(
        scene_id="live_0001", image_path="photo.jpg",
        image_size=(200, 150), preds=_preds(),
    )
    classes = {o.obj_class for o in scene.objects}
    assert classes == {"knife", "apple"}
    knife = next(o for o in scene.objects if o.obj_class == "knife")
    assert knife.bbox == [10.0, 10.0, 50.0, 90.0]


def test_obj_ids_are_unique_and_stable_per_class():
    preds = [
        {"bbox": [0.0, 0.0, 10.0, 10.0], "cls": "apple", "conf": 0.9},
        {"bbox": [20.0, 0.0, 30.0, 10.0], "cls": "apple", "conf": 0.8},
    ]
    scene = scene_from_predictions(
        scene_id="live_0002", image_path="photo.jpg",
        image_size=(100, 100), preds=preds,
    )
    ids = [o.obj_id for o in scene.objects]
    assert len(ids) == len(set(ids))
    assert all(oid.startswith("apple_") for oid in ids)


def test_below_conf_threshold_predictions_are_dropped():
    preds = _preds() + [{"bbox": [0.0, 0.0, 5.0, 5.0], "cls": "bowl", "conf": 0.1}]
    scene = scene_from_predictions(
        scene_id="live_0003", image_path="photo.jpg",
        image_size=(200, 150), preds=preds, conf_thresh=0.25,
    )
    assert {o.obj_class for o in scene.objects} == {"knife", "apple"}


def test_no_predictions_gives_empty_object_list():
    scene = scene_from_predictions(
        scene_id="live_0004", image_path="photo.jpg",
        image_size=(200, 150), preds=[],
    )
    assert scene.objects == []
