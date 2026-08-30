"""Tests for affdet.detect.yolo_dataset — Scene JSONs → Ultralytics dataset."""
from pathlib import Path

import pytest
import yaml

from affdet.render.mock import MockRenderer
from affdet.schemas import Scene, SceneObject, Camera
from affdet.detect.yolo_dataset import convert


def _write_scene(out_dir: Path, scene_id: str, objects: list[SceneObject],
                 image_size=(128, 96)) -> None:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(128, 128, 128)).save(out_dir / f"{scene_id}.png")
    scene = Scene(
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
    (out_dir / f"{scene_id}.json").write_text(scene.model_dump_json())


def _obj(obj_class: str, bbox: list[float], idx: int = 0) -> SceneObject:
    return SceneObject(obj_id=f"{obj_class}_{idx}", obj_class=obj_class, bbox=bbox)


class TestConvertLayout:
    def test_mock_scenes_produce_full_dataset_layout(self, tmp_path):
        scenes = tmp_path / "scenes"
        MockRenderer(seed=0, n_scenes=10).run(scenes)
        out = tmp_path / "dataset"

        convert(scenes, out, val_fraction=0.2, seed=0)

        assert (out / "data.yaml").is_file()
        train_imgs = list((out / "images" / "train").iterdir())
        val_imgs = list((out / "images" / "val").iterdir())
        assert len(train_imgs) == 8
        assert len(val_imgs) == 2
        # every image has a label file with one line per object
        for img in train_imgs + val_imgs:
            split = img.parent.name
            label = out / "labels" / split / f"{img.stem}.txt"
            assert label.is_file()
            scene = Scene.model_validate_json(
                (scenes / f"{img.stem}.json").read_text())
            lines = label.read_text().strip().splitlines()
            assert len(lines) == len(scene.objects)

    def test_images_are_symlinked_not_copied(self, tmp_path):
        scenes = tmp_path / "scenes"
        MockRenderer(seed=0, n_scenes=5).run(scenes)
        out = tmp_path / "dataset"

        convert(scenes, out, seed=0)

        imgs = list((out / "images" / "train").iterdir()) + \
               list((out / "images" / "val").iterdir())
        assert imgs and all(p.is_symlink() for p in imgs)

    def test_data_yaml_names_are_sorted_and_ids_match_labels(self, tmp_path):
        scenes = tmp_path / "scenes"
        _write_scene(scenes, "scene_0001", [
            _obj("knife", [10, 10, 50, 30], 0),
            _obj("apple", [60, 40, 90, 70], 1),
        ])
        _write_scene(scenes, "scene_0002", [_obj("bowl", [5, 5, 40, 40], 0)])
        out = tmp_path / "dataset"

        convert(scenes, out, val_fraction=0.5, seed=0)

        data = yaml.safe_load((out / "data.yaml").read_text())
        assert data["names"] == ["apple", "bowl", "knife"]
        # class id in the label file matches the index in names
        label_files = list((out / "labels").rglob("scene_0001.txt"))
        assert len(label_files) == 1
        ids = sorted(int(l.split()[0]) for l in
                     label_files[0].read_text().strip().splitlines())
        assert ids == [data["names"].index("apple"), data["names"].index("knife")]


class TestLabelGeometry:
    def test_bbox_normalized_to_xywh(self, tmp_path):
        # image 128x96, bbox [32, 24, 96, 72] → cx=0.5 cy=0.5 w=0.5 h=0.5
        scenes = tmp_path / "scenes"
        _write_scene(scenes, "scene_0001",
                     [_obj("apple", [32.0, 24.0, 96.0, 72.0])])
        out = tmp_path / "dataset"

        convert(scenes, out, val_fraction=0.0, seed=0)

        label = out / "labels" / "train" / "scene_0001.txt"
        cls, cx, cy, w, h = label.read_text().split()
        assert cls == "0"
        assert (float(cx), float(cy), float(w), float(h)) == \
               pytest.approx((0.5, 0.5, 0.5, 0.5))


class TestSplit:
    def test_split_is_deterministic_and_disjoint(self, tmp_path):
        scenes = tmp_path / "scenes"
        MockRenderer(seed=0, n_scenes=10).run(scenes)

        out1 = tmp_path / "d1"
        out2 = tmp_path / "d2"
        convert(scenes, out1, val_fraction=0.3, seed=42)
        convert(scenes, out2, val_fraction=0.3, seed=42)

        def split_names(out):
            train = {p.stem for p in (out / "images" / "train").iterdir()}
            val = {p.stem for p in (out / "images" / "val").iterdir()}
            return train, val

        t1, v1 = split_names(out1)
        t2, v2 = split_names(out2)
        assert t1 == t2 and v1 == v2
        assert not (t1 & v1)
        assert len(t1) + len(v1) == 10

    def test_worker_layout_converts_all_workers_into_one_dataset(self, tmp_path):
        # cluster layout: scenes/worker_N/scene_XXXX/scene_XXXX.{json,png}
        scenes = tmp_path / "scenes"
        for worker, scene_id in [(0, "scene_0001"), (0, "scene_0002"),
                                 (1, "scene_0003")]:
            _write_scene(scenes / f"worker_{worker}" / scene_id, scene_id,
                         [_obj("apple", [10, 10, 50, 50])])
        out = tmp_path / "dataset"

        convert(scenes, out, val_fraction=0.0, seed=0)

        all_imgs = {p.stem for p in (out / "images").rglob("*.png")}
        assert all_imgs == {"scene_0001", "scene_0002", "scene_0003"}

    def test_exclude_scene_ids(self, tmp_path):
        scenes = tmp_path / "scenes"
        MockRenderer(seed=0, n_scenes=5).run(scenes)
        out = tmp_path / "dataset"

        convert(scenes, out, seed=0, exclude_scene_ids={"scene_0001", "scene_0002"})

        all_imgs = {p.stem for p in (out / "images").rglob("*.png")}
        assert "scene_0001" not in all_imgs and "scene_0002" not in all_imgs
        assert len(all_imgs) == 3


class TestValidation:
    def test_unknown_class_raises_when_allowed_given(self, tmp_path):
        scenes = tmp_path / "scenes"
        _write_scene(scenes, "scene_0001", [_obj("spaceship", [10, 10, 50, 50])])
        with pytest.raises(ValueError, match="spaceship"):
            convert(scenes, tmp_path / "d", allowed_classes={"apple", "knife"})

    def test_degenerate_bbox_raises(self, tmp_path):
        scenes = tmp_path / "scenes"
        _write_scene(scenes, "scene_0001", [_obj("apple", [50, 10, 50, 40])])
        with pytest.raises(ValueError, match="degenerate"):
            convert(scenes, tmp_path / "d")

    def test_out_of_bounds_bbox_raises(self, tmp_path):
        scenes = tmp_path / "scenes"
        _write_scene(scenes, "scene_0001", [_obj("apple", [10, 10, 200, 40])])
        with pytest.raises(ValueError, match="bounds"):
            convert(scenes, tmp_path / "d")

    def test_scene_without_objects_raises(self, tmp_path):
        scenes = tmp_path / "scenes"
        _write_scene(scenes, "scene_0001", [])
        with pytest.raises(ValueError, match="no objects"):
            convert(scenes, tmp_path / "d")

    def test_empty_scenes_dir_raises(self, tmp_path):
        (tmp_path / "scenes").mkdir()
        with pytest.raises(ValueError, match="No scenes"):
            convert(tmp_path / "scenes", tmp_path / "d")
