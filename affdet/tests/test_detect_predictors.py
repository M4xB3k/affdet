"""Predictor construction for the E5 detector arms.

The heavy model loading is injected, so what is tested here is the wiring
that actually differs between arms — which vocabulary a zero-shot detector
is given, and that its emitted labels are mapped back into the taxonomy's
own label space before scoring.
"""
import pytest

from affdet.detect.predictors import (
    build_predictor,
    normalise_predictions,
    taxonomy_classes,
)
from affdet.experiment.component_config import ComponentEvalConfig


def test_taxonomy_read_from_the_dataset_yaml(tmp_path):
    """The class list must come from the same data.yaml the detectors were
    trained against, not a hardcoded copy that can drift out of sync."""
    (tmp_path / "data.yaml").write_text(
        "path: /x\ntrain: images/train\nval: images/val\n"
        "names:\n- apple\n- red_wine_glass\n")

    assert taxonomy_classes(tmp_path / "data.yaml") == ["apple", "red_wine_glass"]


def test_zero_shot_arm_is_given_the_despaced_vocabulary():
    """A text-prompted detector reads class names as English; underscores
    would measure its tolerance for underscores, not its detection."""
    captured = {}

    class FakeWorldModel:
        def set_classes(self, names):
            captured["names"] = names

        def predict(self, *a, **kw):
            return []

    cfg = ComponentEvalConfig(name="zs", scenes="/s",
                              detector="yolov8s-worldv2", zero_shot=True)
    build_predictor(cfg, model=FakeWorldModel(),
                    classes=["apple", "red_wine_glass"])

    assert captured["names"] == ["apple", "red wine glass"]


def test_finetuned_arm_does_not_override_its_vocabulary():
    """A fine-tuned checkpoint carries its own learned class indices;
    calling set_classes on it would discard the fine-tuning."""
    calls = []

    class FakeModel:
        def set_classes(self, names):
            calls.append(names)

        def predict(self, *a, **kw):
            return []

    cfg = ComponentEvalConfig(name="ft", scenes="/s",
                              detector="yolov8s-worldv2", weights="/w.pt")
    build_predictor(cfg, model=FakeModel(), classes=["apple"])

    assert calls == [], "fine-tuned arm must keep its trained vocabulary"


def test_emitted_labels_map_back_to_canonical_ids():
    """Scoring happens in the taxonomy's label space, so 'red wine glass'
    coming back from a text model must become 'red_wine_glass'."""
    preds = [{"bbox": [0, 0, 1, 1], "cls": "red wine glass", "conf": 0.9}]

    out = normalise_predictions(preds, ["red_wine_glass", "apple"])

    assert out[0]["cls"] == "red_wine_glass"


def test_out_of_vocabulary_label_is_kept_and_marked_not_dropped():
    """An open-vocab detector can emit something outside the taxonomy. That
    box still exists and still costs precision, so dropping it would flatter
    the detector; it is kept with a None class, which cannot match any gold
    object."""
    preds = [{"bbox": [0, 0, 1, 1], "cls": "spaceship", "conf": 0.9}]

    out = normalise_predictions(preds, ["apple"])

    assert len(out) == 1
    assert out[0]["cls"] is None


def test_unknown_detector_name_fails_loudly():
    cfg = ComponentEvalConfig(name="x", scenes="/s", detector="not-a-detector",
                              weights="/w.pt")
    with pytest.raises(ValueError, match="not-a-detector"):
        build_predictor(cfg)
