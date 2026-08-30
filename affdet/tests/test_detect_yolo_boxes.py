"""Tests for affdet.detect.yolo_boxes — rewrite Scene bboxes with matched YOLO boxes.

Produces the "YOLO arm" scene set for the detector-influence ablation: each gold
object keeps its obj_id (so labels stay aligned) but its bbox is replaced by the
IoU-matched YOLO prediction. Gold objects YOLO missed are dropped.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from affdet.schemas import Scene, SceneObject, Camera
from affdet.detect.yolo_boxes import (
    iou,
    match_boxes,
    rewrite_scene,
    rewrite_scenes,
    dedup_contained_boxes,
    _ultralytics_predictor,
)


def _obj(obj_class: str, bbox: list[float], idx: int = 0) -> SceneObject:
    return SceneObject(obj_id=f"{obj_class}_{idx}", obj_class=obj_class, bbox=bbox)


def _scene(scene_id: str, objects: list[SceneObject],
           image_size=(128, 96)) -> Scene:
    return Scene(
        scene_id=scene_id,
        image_path=f"{scene_id}.png",
        image_size=list(image_size),
        source="blender",
        objects=objects,
        camera=Camera(
            pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]],
            intrinsics=[[500, 0, 64], [0, 500, 48], [0, 0, 1]],
        ),
    )


def _write_scene(out_dir: Path, scene: Scene) -> None:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tuple(scene.image_size), color=(128, 128, 128)).save(
        out_dir / scene.image_path)
    (out_dir / f"{scene.scene_id}.json").write_text(scene.model_dump_json())


class TestIoU:
    def test_identical_boxes_iou_one(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_half_overlap(self):
        # A=[0,0,10,10] area 100; B=[5,0,15,10] area 100; inter=[5,0,10,10]=50
        # union = 100 + 100 - 50 = 150 -> 50/150 = 1/3
        assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)

    def test_touching_edges_iou_zero(self):
        # share only an edge -> zero-area intersection
        assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

    def test_contained_box(self):
        # inner fully inside outer: inter = inner area = 25, union = outer 100
        assert iou([0, 0, 10, 10], [2, 2, 7, 7]) == pytest.approx(25 / 100)


class TestMatchBoxes:
    def test_perfect_match_assigns_pred_box(self):
        gold = [_obj("knife", [0, 0, 10, 10], 0)]
        preds = [{"bbox": [0, 0, 10, 10], "cls": "knife"}]
        matched = match_boxes(gold, preds, iou_thresh=0.5)
        assert matched == {"knife_0": [0, 0, 10, 10]}

    def test_below_threshold_is_dropped(self):
        gold = [_obj("knife", [0, 0, 10, 10], 0)]
        # IoU 1/3 < 0.5
        preds = [{"bbox": [5, 0, 15, 10], "cls": "knife"}]
        assert match_boxes(gold, preds, iou_thresh=0.5) == {}

    def test_unmatched_gold_absent_from_result(self):
        gold = [_obj("knife", [0, 0, 10, 10], 0),
                _obj("apple", [50, 50, 60, 60], 1)]
        preds = [{"bbox": [0, 0, 10, 10], "cls": "knife"}]  # apple missed
        matched = match_boxes(gold, preds, iou_thresh=0.5)
        assert "knife_0" in matched and "apple_1" not in matched

    def test_each_pred_used_at_most_once(self):
        # two gold objects overlap the same single pred; only the best-IoU gold wins
        gold = [_obj("bowl", [0, 0, 10, 10], 0),
                _obj("dish", [0, 0, 9, 9], 1)]  # slightly smaller, lower IoU vs pred
        preds = [{"bbox": [0, 0, 10, 10], "cls": "bowl"}]
        matched = match_boxes(gold, preds, iou_thresh=0.5)
        assert len(matched) == 1
        assert "bowl_0" in matched  # exact overlap wins over the 9x9

    def test_best_of_multiple_preds_chosen(self):
        gold = [_obj("knife", [0, 0, 10, 10], 0)]
        preds = [
            {"bbox": [5, 0, 15, 10], "cls": "knife"},   # IoU 1/3
            {"bbox": [0, 0, 10, 10], "cls": "knife"},   # IoU 1.0
        ]
        assert match_boxes(gold, preds, iou_thresh=0.5) == {"knife_0": [0, 0, 10, 10]}

    def test_matching_ignores_class_labels(self):
        # matching is geometric only — a mislabelled pred still supplies its box,
        # because the ablation keeps the gold obj_id/class and only swaps pixels.
        gold = [_obj("knife", [0, 0, 10, 10], 0)]
        preds = [{"bbox": [0, 0, 10, 10], "cls": "fork"}]
        assert match_boxes(gold, preds, iou_thresh=0.5) == {"knife_0": [0, 0, 10, 10]}


class TestRewriteScene:
    def test_matched_object_gets_yolo_box_keeps_id_and_class(self):
        scene = _scene("scene_0001", [_obj("knife", [0, 0, 10, 10], 0)])
        preds = [{"bbox": [1, 1, 11, 11], "cls": "knife"}]  # IoU high enough
        out = rewrite_scene(scene, preds, iou_thresh=0.5)
        assert len(out.objects) == 1
        o = out.objects[0]
        assert o.obj_id == "knife_0" and o.obj_class == "knife"
        assert o.bbox == [1, 1, 11, 11]

    def test_missed_object_dropped(self):
        scene = _scene("scene_0001", [
            _obj("knife", [0, 0, 10, 10], 0),
            _obj("apple", [50, 50, 60, 60], 1),
        ])
        preds = [{"bbox": [0, 0, 10, 10], "cls": "knife"}]
        out = rewrite_scene(scene, preds, iou_thresh=0.5)
        assert [o.obj_id for o in out.objects] == ["knife_0"]

    def test_scene_metadata_preserved(self):
        scene = _scene("scene_0001", [_obj("knife", [0, 0, 10, 10], 0)])
        preds = [{"bbox": [0, 0, 10, 10], "cls": "knife"}]
        out = rewrite_scene(scene, preds, iou_thresh=0.5)
        assert out.scene_id == scene.scene_id
        assert out.image_path == scene.image_path
        assert out.image_size == scene.image_size


class TestDedupContainedBoxes:
    """A sub-part detection (e.g. a knife blade's reflective tip) sits almost
    entirely inside another box for the same physical object. Standard
    IoU-based NMS doesn't catch this: IoU is low because the union is
    dominated by the big box's area, even though the small box has no
    territory of its own outside it. This dedup checks containment (fraction
    of the SMALLER box's own area covered by a bigger one) instead of IoU."""

    def test_small_box_inside_big_box_is_dropped(self):
        preds = [
            {"bbox": [0, 0, 100, 100], "cls": "knife", "confidence": 0.5},
            {"bbox": [80, 80, 95, 95], "cls": "kettle", "confidence": 0.3},  # fully inside
        ]
        out = dedup_contained_boxes(preds)
        assert len(out) == 1
        assert out[0]["bbox"] == [0, 0, 100, 100]

    def test_disjoint_boxes_both_kept(self):
        preds = [
            {"bbox": [0, 0, 10, 10], "cls": "apple", "confidence": 0.9},
            {"bbox": [50, 50, 60, 60], "cls": "knife", "confidence": 0.8},
        ]
        out = dedup_contained_boxes(preds)
        assert len(out) == 2

    def test_partial_overlap_below_containment_threshold_both_kept(self):
        # B's area is 100 (10x10); overlap with A is [5,0,10,10] = 50 -> only
        # 50% of B contained in A, below the 0.85 default threshold.
        preds = [
            {"bbox": [0, 0, 10, 10], "cls": "dish", "confidence": 0.9},
            {"bbox": [5, 0, 15, 10], "cls": "bowl", "confidence": 0.8},
        ]
        out = dedup_contained_boxes(preds)
        assert len(out) == 2

    def test_giant_near_full_image_box_never_absorbs_real_detections(self):
        # A spurious box covering nearly the whole frame must not cause every
        # real, smaller detection to be dropped as its "sub-part" -- that's
        # backwards, the giant box is the spurious one here.
        preds = [
            {"bbox": [0, 0, 1600, 1067], "cls": "knife", "confidence": 0.2},
            {"bbox": [100, 100, 300, 300], "cls": "apple", "confidence": 0.9},
            {"bbox": [900, 400, 1100, 700], "cls": "dish", "confidence": 0.85},
        ]
        out = dedup_contained_boxes(preds, image_size=(1600, 1067))
        kept_classes = {o["cls"] for o in out}
        assert "apple" in kept_classes
        assert "dish" in kept_classes

    def test_higher_confidence_box_wins_when_both_could_be_container(self):
        # Two boxes, one contained in the other -- must keep the bigger one
        # (the container), regardless of which has higher confidence, since
        # the smaller one has no independent territory.
        preds = [
            {"bbox": [10, 10, 20, 20], "cls": "kettle", "confidence": 0.9},
            {"bbox": [0, 0, 100, 100], "cls": "knife", "confidence": 0.3},
        ]
        out = dedup_contained_boxes(preds)
        assert len(out) == 1
        assert out[0]["bbox"] == [0, 0, 100, 100]


class TestRewriteScenes:
    def test_writes_parallel_scene_set_with_yolo_boxes(self, tmp_path):
        scenes = tmp_path / "scenes"
        _write_scene(scenes, _scene("scene_0001", [
            _obj("knife", [0, 0, 10, 10], 0),
            _obj("apple", [50, 50, 60, 60], 1),
        ]))
        out = tmp_path / "scenes_yolo"

        # fake predictor: perfect knife box, apple missed
        def predictor(image_path, scene):
            return [{"bbox": [0, 0, 10, 10], "cls": "knife"}]

        stats = rewrite_scenes(scenes, out, predictor, iou_thresh=0.5)

        written = Scene.model_validate_json(
            (out / "scene_0001.json").read_text())
        assert [o.obj_id for o in written.objects] == ["knife_0"]
        assert written.objects[0].bbox == [0, 0, 10, 10]
        # image is available alongside the rewritten json (symlink or copy)
        assert (out / "scene_0001.png").exists()
        # stats report the drop so the ablation write-up can cite recall
        assert stats["n_gold_objects"] == 2
        assert stats["n_matched"] == 1
        assert stats["n_dropped"] == 1

    def test_empty_scenes_dir_raises(self, tmp_path):
        (tmp_path / "scenes").mkdir()
        with pytest.raises(ValueError, match="No scenes"):
            rewrite_scenes(tmp_path / "scenes", tmp_path / "out",
                           lambda p, s: [], iou_thresh=0.5)


def _install_fake_ultralytics(monkeypatch, yolo_side_effect=None):
    """Same pattern as affdet/tests/test_detect_train.py -- fake ultralytics
    module with mocked YOLO/YOLOWorld classes, no real network/file load."""
    fake_module = MagicMock(name="ultralytics_module")
    fake_yolo = MagicMock(name="YOLO")
    if yolo_side_effect is not None:
        fake_yolo.side_effect = yolo_side_effect
    fake_module.YOLO = fake_yolo
    fake_module.YOLOWorld = MagicMock(name="YOLOWorld")
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    return fake_module


class TestUltralyticsPredictorWorldFallback:
    """A fine-tuned YOLO-World checkpoint's filename doesn't reliably contain
    "world" (the actual deployed checkpoint is plain best.pt), so
    _ultralytics_predictor can't dispatch on the weights string the way
    affdet.detect.train._ultralytics_factory does at training time. Instead:
    try the plain YOLO loader first, and if THAT specific checkpoint-format
    TypeError is raised (real error seen loading a YOLO-World checkpoint via
    plain YOLO()), fall back to YOLOWorld."""

    def test_plain_yolo_checkpoint_uses_yolo_class(self, monkeypatch):
        fake = _install_fake_ultralytics(monkeypatch)
        _ultralytics_predictor("yolo11s_final/best.pt")
        fake.YOLO.assert_called_once_with("yolo11s_final/best.pt")
        fake.YOLOWorld.assert_not_called()

    def test_world_checkpoint_that_yolo_rejects_falls_back_to_yoloworld(self, monkeypatch):
        def _reject_world_checkpoint(weights):
            raise TypeError(
                f"ERROR ❌️ {weights} references types outside the "
                "supported Ultralytics checkpoint format."
            )
        fake = _install_fake_ultralytics(monkeypatch, yolo_side_effect=_reject_world_checkpoint)
        _ultralytics_predictor("yolov8s_world_full/weights/best.pt")
        fake.YOLO.assert_called_once_with("yolov8s_world_full/weights/best.pt")
        fake.YOLOWorld.assert_called_once_with("yolov8s_world_full/weights/best.pt")

    def test_unrelated_yolo_error_is_not_swallowed(self, monkeypatch):
        """Only the specific checkpoint-format TypeError should trigger a
        fallback -- any other failure (missing file, corrupt weights) must
        propagate normally, not silently retry as YOLOWorld."""
        def _raise_file_not_found(weights):
            raise FileNotFoundError(f"no such file: {weights}")
        fake = _install_fake_ultralytics(monkeypatch, yolo_side_effect=_raise_file_not_found)
        with pytest.raises(FileNotFoundError):
            _ultralytics_predictor("missing.pt")
        fake.YOLOWorld.assert_not_called()
