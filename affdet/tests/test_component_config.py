"""ComponentEvalConfig — one config names one detector arm of E5."""
import pytest
import yaml

from affdet.experiment.component_config import ComponentEvalConfig


def _cfg(**kw):
    base = dict(name="e5-yoloworld-ft", scenes="/scratch/scenes",
                detector="yolov8s-worldv2", weights="/scratch/best.pt")
    return ComponentEvalConfig(**{**base, **kw})


def test_minimal_config_has_thesis_defaults():
    cfg = _cfg()
    assert cfg.iou_thresh == 0.5, "the thesis reports IoU >= 0.5"
    assert cfg.split == "test", "component eval scores held-out scenes only"
    assert cfg.seed == 0


def test_zero_shot_arm_needs_no_weights():
    """A zero-shot arm runs from pretrained weights and has no fine-tuned
    checkpoint of its own."""
    cfg = ComponentEvalConfig(name="e5-yoloworld-zs", scenes="/s",
                              detector="yolov8s-worldv2", zero_shot=True)
    assert cfg.zero_shot is True
    assert cfg.weights is None


def test_finetuned_arm_requires_weights():
    """Silently scoring pretrained weights as if they were the fine-tuned
    checkpoint would be an unnoticed wrong table row."""
    with pytest.raises(ValueError, match="weights"):
        ComponentEvalConfig(name="e5-bad", scenes="/s", detector="yolo11s")


def test_unknown_field_is_rejected():
    """extra='forbid', as ExperimentConfig -- a typo'd knob must fail loudly
    rather than be silently ignored while the run reports a default."""
    with pytest.raises(Exception):
        _cfg(iou_threshold=0.75)


def test_roundtrips_through_yaml(tmp_path):
    p = tmp_path / "arm.yaml"
    p.write_text(yaml.safe_dump({
        "name": "e5-yolo11s-ft", "scenes": "/scratch/scenes",
        "detector": "yolo11s", "weights": "/scratch/best.pt", "seed": 2,
    }))
    cfg = ComponentEvalConfig.from_yaml(p)
    assert cfg.name == "e5-yolo11s-ft"
    assert cfg.seed == 2


def test_arm_key_distinguishes_detectors_and_shot_mode():
    """Two arms differing only in zero-shot must not collapse into one
    family when results are aggregated -- the same class of bug as the
    v3/v3 collision."""
    ft = ComponentEvalConfig(name="a", scenes="/s", detector="yolov8s-worldv2",
                             weights="/w.pt")
    zs = ComponentEvalConfig(name="b", scenes="/s", detector="yolov8s-worldv2",
                             zero_shot=True)
    assert ft.arm_key() != zs.arm_key()


def _shipped_configs():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    d = root / "experiments" / "configs" / "e5_component"
    return sorted(d.glob("e5-*.yaml"))


def test_every_shipped_arm_scores_on_the_dataset_split():
    """Every E5 arm must take its eval scenes from the built dataset's own
    val split.

    The seeded scene shuffle and the dataset split are INDEPENDENT random
    partitions of the same corpus, so they barely agree: measured on the
    scene-disjoint dataset, 90.5% of the shuffle's "test" scenes are scenes
    the detector actually trained on. Scoring there measures memorisation.

    This bit D-b first (its split is defined by mesh identity, so the
    mismatch was obvious) and the fix was applied only there, on the wrong
    assumption that "scene-disjoint" made D-a's shuffle equivalent to the
    dataset split. It does not — the dataset was built with its own split,
    and that is the one that decided what the detector saw.
    """
    offenders = []
    for p in _shipped_configs():
        spec = yaml.safe_load(p.read_text())
        if not spec.get("split_from_dataset"):
            offenders.append(p.name)

    assert not offenders, (
        "these arms would be scored on a seeded shuffle that overlaps the "
        f"detector's training scenes: {offenders}")


def test_every_shipped_arm_names_a_data_yaml():
    """split_from_dataset is meaningless without one, and the class list is
    read from it so every arm predicts over the same vocabulary."""
    missing = [p.name for p in _shipped_configs()
               if not yaml.safe_load(p.read_text()).get("data_yaml")]
    assert not missing
