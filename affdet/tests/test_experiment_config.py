import yaml
from affdet.experiment.config import ExperimentConfig


def _cfg(**kw):
    base = dict(name="t", scenes="/s", labels="/l")
    base.update(kw)
    return ExperimentConfig(**base)


def test_defaults():
    c = _cfg()
    assert c.bbox_source == "oracle" and c.head == "mlp" and c.crop_padding == 0.0
    assert c.unbounded is False


def test_unbounded_is_not_an_extract_field():
    """unbounded changes training/eval, not extraction -- must not force a
    re-extract when toggled (same reasoning as head/epochs/seed)."""
    a = _cfg(unbounded=False)
    b = _cfg(unbounded=True)
    assert a.extract_hash() == b.extract_hash()


def test_extract_key_only_extract_fields():
    c = _cfg()
    assert set(c.extract_key()) == {"scenes", "bbox_source", "crop_padding", "backbone"}


def test_hash_ignores_non_extract_fields():
    a = _cfg(head="mlp", epochs=50, seed=0)
    b = _cfg(head="attention", epochs=999, seed=7)
    assert a.extract_hash() == b.extract_hash()   # same extract config


def test_hash_changes_on_padding():
    assert _cfg(crop_padding=0.0).extract_hash() != _cfg(crop_padding=0.1).extract_hash()


def test_from_yaml(tmp_path):
    p = tmp_path/"c.yaml"
    p.write_text(yaml.safe_dump({"name":"g","scenes":"/x","labels":"/l","head":"mlp-big"}))
    c = ExperimentConfig.from_yaml(p)
    assert c.name == "g" and c.head == "mlp-big"


def test_unknown_field_rejected():
    import pytest
    with pytest.raises(Exception):
        _cfg(bogus=1)


# --- holdout_mode/holdout_frac (E1c/E1c') -----------------------------------

def test_holdout_defaults():
    c = _cfg()
    assert c.holdout_mode is None
    assert c.holdout_frac == 0.20


def test_holdout_mode_is_settable():
    c = _cfg(holdout_mode="strict", holdout_frac=0.30)
    assert c.holdout_mode == "strict"
    assert c.holdout_frac == 0.30


def test_holdout_is_not_an_extract_field():
    """holdout_mode changes the train/val/test split, not extraction -- must
    not force a re-extract when toggled (same reasoning as unbounded)."""
    a = _cfg(holdout_mode=None)
    b = _cfg(holdout_mode="strict", holdout_frac=0.30)
    assert a.extract_hash() == b.extract_hash()
