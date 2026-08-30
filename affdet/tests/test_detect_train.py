"""Tests for affdet.detect.train — thin wrapper around Ultralytics training."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from affdet.detect.train import _ultralytics_factory, train_detector


class FakeModel:
    def __init__(self):
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return {"ok": True}


def test_train_detector_passes_settings_to_model(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [apple]\n")
    fake = FakeModel()

    result = train_detector(
        data_yaml, model="yolo11s.pt", epochs=5, imgsz=640, seed=7,
        project=tmp_path / "runs", name="smoke",
        model_factory=lambda weights: fake,
    )

    assert result == {"ok": True}
    assert fake.train_kwargs["data"] == str(data_yaml)
    assert fake.train_kwargs["epochs"] == 5
    assert fake.train_kwargs["imgsz"] == 640
    assert fake.train_kwargs["seed"] == 7
    assert fake.train_kwargs["name"] == "smoke"


def test_model_factory_receives_weights_string(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [apple]\n")
    seen = []

    def factory(weights):
        seen.append(weights)
        return FakeModel()

    train_detector(data_yaml, model="rtdetr-l.pt", model_factory=factory)
    assert seen == ["rtdetr-l.pt"]


def _install_fake_ultralytics(monkeypatch):
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = MagicMock(name="YOLO")
    fake_module.RTDETR = MagicMock(name="RTDETR")
    fake_module.YOLOWorld = MagicMock(name="YOLOWorld")
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    return fake_module


class TestUltralyticsFactory:
    def test_rtdetr_weights_use_rtdetr_class(self, monkeypatch):
        fake = _install_fake_ultralytics(monkeypatch)
        _ultralytics_factory("rtdetr-l.pt")
        fake.RTDETR.assert_called_once_with("rtdetr-l.pt")
        fake.YOLO.assert_not_called()
        fake.YOLOWorld.assert_not_called()

    def test_world_weights_use_yoloworld_class(self, monkeypatch):
        fake = _install_fake_ultralytics(monkeypatch)
        _ultralytics_factory("yolov8s-worldv2.pt")
        fake.YOLOWorld.assert_called_once_with("yolov8s-worldv2.pt")
        fake.YOLO.assert_not_called()
        fake.RTDETR.assert_not_called()

    def test_plain_weights_use_yolo_class(self, monkeypatch):
        fake = _install_fake_ultralytics(monkeypatch)
        _ultralytics_factory("yolo11s.pt")
        fake.YOLO.assert_called_once_with("yolo11s.pt")
        fake.RTDETR.assert_not_called()
        fake.YOLOWorld.assert_not_called()
