"""run_classification_eval — probe vs. detector class head on identical crops.

Ported from an uncommitted cluster script (which produced the 98.76%/93.12%
figure but had no test, no config and no ledger row). The methodology it
established is kept; what is added is that it is now reproducible.

Feature extraction and the detector are injected, so the split -> fit ->
score -> ledger path is testable without a GPU or the render corpus.
"""
import json

import numpy as np
import pytest

from affdet.experiment.component_config import ComponentEvalConfig
from affdet.experiment.classification_runner import run_classification_eval
from affdet.render.mock import MockRenderer


@pytest.fixture
def scenes(tmp_path):
    """Enough scenes that every class in the held-out split also appears in
    train. With too few, a class can land only in test, where the probe
    cannot possibly predict it -- correct behaviour, but it would cap
    accuracy below 1.0 and make these tests measure the fixture."""
    d = tmp_path / "scenes"
    MockRenderer(seed=0, n_scenes=300).run(d)
    return d


def _cfg(scenes, **kw):
    base = dict(name="e5-cls", scenes=str(scenes), detector="fake",
                weights="/fake.pt")
    return ComponentEvalConfig(**{**base, **kw})


def _perfect_detector(image_path, scene):
    return [{"bbox": list(o.bbox), "cls": o.obj_class, "conf": 0.99}
            for o in scene.objects]


def _wrong_class_detector(image_path, scene):
    """Right boxes, always the wrong name -- isolates the class head."""
    return [{"bbox": list(o.bbox), "cls": "definitely_wrong", "conf": 0.9}
            for o in scene.objects]


#: Stable, collision-free class -> feature-index map. Hashing into a fixed
#: number of slots would alias two classes onto one feature and cap the
#: probe below 100%, making the test measure the fixture rather than the
#: code under test.
_CLASS_SLOTS: dict[str, int] = {}


def _separable_features(crops):
    """One perfectly class-separable feature per crop, so the probe can
    reach 100% and the test asserts on wiring rather than on learning."""
    # Fixed width: the train pass and the eval pass must return matrices of
    # the same dimensionality, and eval can surface a class train never saw.
    dim = 128
    out = []
    for c in crops:
        slot = _CLASS_SLOTS.setdefault(c["true_cls"], len(_CLASS_SLOTS))
        assert slot < dim, "test fixture ran out of feature slots"
        v = np.zeros(dim, dtype=np.float32)
        v[slot] = 1.0
        out.append(v)
    return np.array(out)


class _NearestCentroidProbe:
    """Dependency-free stand-in for the sklearn probe.

    scikit-learn is an optional dependency (extras: `probe`), and the point
    of these tests is the split/fit/score/ledger wiring, not which linear
    model is used -- production injects the real one.
    """

    def fit(self, X, y):
        self._centroids = {}
        for cls in set(y):
            rows = np.array([x for x, label in zip(X, y) if label == cls])
            self._centroids[cls] = rows.mean(axis=0)
        return self

    def predict(self, X):
        classes = list(self._centroids)
        mat = np.array([self._centroids[c] for c in classes])
        return np.array([classes[int(np.argmax(mat @ x))] for x in X])


def _probe_factory():
    return _NearestCentroidProbe()


def test_probe_learns_from_train_and_scores_heldout(scenes, tmp_path):
    r = run_classification_eval(
        _cfg(scenes), tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory,
    )

    assert r["classification"]["n"] > 0
    assert r["classification"]["probe"]["accuracy"] == 1.0


def test_probe_and_classhead_are_scored_on_the_same_objects(scenes, tmp_path):
    """The whole point of the comparison: any difference must be the
    classifier, not a different set of crops."""
    r = run_classification_eval(
        _cfg(scenes), tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory,
    )
    c = r["classification"]

    assert c["probe"]["n"] == c["classhead"]["n"] == c["n"]


def test_wrong_class_head_loses_to_the_probe(scenes, tmp_path):
    r = run_classification_eval(
        _cfg(scenes), tmp_path / "root",
        detector_factory=lambda cfg: _wrong_class_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory,
    )

    assert r["classification"]["classhead"]["accuracy"] == 0.0
    assert r["classification"]["probe"]["accuracy"] == 1.0
    assert r["classification"]["delta_accuracy"] == 1.0


def test_detector_misses_are_reported_not_folded_into_accuracy(scenes, tmp_path):
    """An object the detector never boxed cannot be classified by EITHER
    method; counting it as a classification error would blame the
    classifier for a detection failure."""
    def half_blind(image_path, scene):
        return [{"bbox": list(o.bbox), "cls": o.obj_class, "conf": 0.9}
                for o in scene.objects[:1]]

    r = run_classification_eval(
        _cfg(scenes), tmp_path / "root",
        detector_factory=lambda cfg: half_blind,
        feature_fn=_separable_features,
        probe_factory=_probe_factory,
    )

    assert r["n_detector_misses"] > 0
    assert r["classification"]["n"] == r["n_matched_objects"]
    assert r["n_matched_objects"] < r["n_gold_objects"]


def test_probe_never_trains_on_heldout_scenes(scenes, tmp_path):
    """Fitting on the scenes it is scored on would make the probe's
    advantage meaningless."""
    r = run_classification_eval(
        _cfg(scenes), tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory,
    )

    assert not (set(r["train_scene_ids"]) & set(r["eval_scene_ids"]))


def test_run_is_appended_to_the_ledger(scenes, tmp_path):
    root = tmp_path / "root"
    run_classification_eval(
        _cfg(scenes), root,
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory,
    )

    rows = [json.loads(l) for l in
            (root / "runs.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "classification"
    assert "probe" in rows[0]["metrics"]["classification"]


def test_seed_changes_the_split(scenes, tmp_path):
    """Unlike detection, the probe is FITTED, so seeds matter here and three
    are reported."""
    a = run_classification_eval(
        _cfg(scenes, seed=0), tmp_path / "a",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)
    b = run_classification_eval(
        _cfg(scenes, seed=1), tmp_path / "b",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)

    assert a["eval_scene_ids"] != b["eval_scene_ids"]


def _write_dataset_yaml(tmp_path, scenes_dir, val_stems):
    """Minimal Ultralytics-style dataset whose val split names given stems."""
    import yaml

    base = tmp_path / "ds"
    (base / "images" / "val").mkdir(parents=True)
    (base / "images" / "train").mkdir(parents=True)
    for stem in val_stems:
        (base / "images" / "val" / f"{stem}.png").write_bytes(b"")
    p = base / "data.yaml"
    p.write_text(yaml.safe_dump({"path": str(base), "train": "images/train",
                                 "val": "images/val", "names": ["a"]}))
    return p


def test_split_from_dataset_keeps_the_probe_off_the_detectors_training_scenes(
        scenes, tmp_path):
    """The instance-disjoint arm's test set is defined by MESH IDENTITY at
    dataset-build time. A seeded scene shuffle does not reproduce it, so the
    probe would be fitted on scenes holding the very meshes the arm reserves
    for testing — the same contamination already fixed in the detection
    runner, which this runner never received."""
    from affdet.detect.yolo_boxes import _find_scene_jsons

    all_stems = sorted(p.stem for p in _find_scene_jsons(scenes))
    val_stems = set(all_stems[:40])
    data_yaml = _write_dataset_yaml(tmp_path, scenes, val_stems)

    r = run_classification_eval(
        _cfg(scenes, data_yaml=str(data_yaml), split_from_dataset=True),
        tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)

    assert set(r["eval_scene_ids"]) == val_stems
    # and nothing the probe trained on may appear in the eval set
    assert not (set(r["train_scene_ids"]) & val_stems)


def test_max_train_scenes_caps_the_probe_fit(scenes, tmp_path):
    """max_scenes caps only the EVAL set, so a "smoke run" still pushed the
    entire train split through DINOv2 — nearly as expensive as the real run,
    which defeats the purpose of smoke-testing before a GPU job."""
    r = run_classification_eval(
        _cfg(scenes, max_scenes=10, max_train_scenes=25), tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)

    assert r["n_train_scenes"] == 25
    assert r["n_eval_scenes"] == 10


@pytest.mark.parametrize("module", [
    "affdet.scripts.classification_eval",
    "affdet.scripts.component_eval",
    "affdet.scripts.component_tables",
])
def test_cli_accepts_its_subcommand_name(module):
    """A Typer app with exactly one command collapses into a bare CLI, so
    `... run --config X` dies with "unexpected extra argument (run)" — while
    every SLURM script invokes exactly that shape.

    An earlier version of this test asserted on `run --help`, which exits 0
    even in the collapsed form (Typer handles --help before it validates
    positionals), so it passed against a CLI that was in fact broken. Invoke
    the subcommand with a real argument instead: the command must be REACHED,
    which a missing config proves (exit 2 from the missing file, not from an
    unexpected-argument parse error)."""
    import importlib

    from typer.testing import CliRunner

    app = importlib.import_module(module).app
    if module.endswith("component_tables"):
        sub, args = "export", ["--ledger", "/nonexistent.jsonl"]
    else:
        sub, args = "run", ["--config", "/nonexistent.yaml",
                            "--root", "/tmp/nope"]

    result = CliRunner().invoke(app, [sub, *args])

    assert "unexpected extra argument" not in result.output.lower(), (
        f"{module}: the {sub!r} subcommand is not reachable — the Typer app "
        "collapsed into a bare CLI. Add a second @app.command().")


def test_features_are_extracted_in_bounded_batches(scenes, tmp_path):
    """The extractor stacked EVERY crop into one tensor before running the
    backbone. On the real train split (11,478 scenes, ~43k crops) that is
    ~26 GB of float32 before the GPU is touched, and all three seeds were
    OOM-killed at 24 GB. Nothing may see an unbounded batch."""
    seen_batch_sizes = []

    def counting_features(crops):
        seen_batch_sizes.append(len(crops))
        return _separable_features(crops)

    run_classification_eval(
        _cfg(scenes, feature_batch=32), tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=counting_features,
        probe_factory=_probe_factory)

    assert seen_batch_sizes, "feature_fn was never called"
    assert max(seen_batch_sizes) <= 32, (
        f"unbounded batch: {max(seen_batch_sizes)} crops in one call")


def test_classification_splits_novel_from_seen_meshes(scenes, tmp_path):
    """The sharper question the detection result raises: the detector's class
    head loses 0.98 -> 0.67 on a mesh it never saw. Whether the DINOv2 probe
    degrades the same way is the actual case for re-classifying crops, and a
    pooled accuracy cannot answer it."""
    from affdet.detect.yolo_boxes import _find_scene_jsons

    stems = sorted(p.stem for p in _find_scene_jsons(scenes))
    val_stems = set(stems[:60])
    data_yaml = _write_dataset_yaml(tmp_path, scenes, val_stems)

    r = run_classification_eval(
        _cfg(scenes, data_yaml=str(data_yaml), split_from_dataset=True,
             split_novel_meshes=True),
        tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)

    groups = r["classification"]["subgroups"]
    assert set(groups) == {"novel", "rest"}
    for g in groups.values():
        assert "probe" in g and "classhead" in g
    assert (groups["novel"]["probe"]["n"] + groups["rest"]["probe"]["n"]
            == r["classification"]["n"])


def test_classification_subgroups_are_opt_in(scenes, tmp_path):
    r = run_classification_eval(
        _cfg(scenes), tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)
    assert "subgroups" not in r["classification"]


def test_novel_means_unseen_by_the_probe_not_just_by_the_detector(
        scenes, tmp_path):
    """"Novel" is derived from the dataset's TRAIN directory, while the probe
    fits on every scene outside val. Those coincide only when train+val cover
    the corpus, which is true of the E5 datasets but is not guaranteed. If a
    scene is in neither directory, the probe still trains on it, so calling
    its meshes "novel" would overstate the probe's generalisation."""
    from affdet.detect.yolo_boxes import _find_scene_jsons

    stems = sorted(p.stem for p in _find_scene_jsons(scenes))
    val_stems = set(stems[:60])
    # deliberately leave 40 scenes out of BOTH dataset directories
    train_stems = set(stems[60:-40])

    import yaml
    base = tmp_path / "ds"
    (base / "images" / "val").mkdir(parents=True)
    (base / "images" / "train").mkdir(parents=True)
    for s in val_stems:
        (base / "images" / "val" / f"{s}.png").touch()
    for s in train_stems:
        (base / "images" / "train" / f"{s}.png").touch()
    dy = base / "data.yaml"
    dy.write_text(yaml.safe_dump({"path": str(base), "train": "images/train",
                                  "val": "images/val", "names": ["a"]}))

    cfg = _cfg(scenes, data_yaml=str(dy), split_from_dataset=True,
               split_novel_meshes=True)
    r = run_classification_eval(
        cfg, tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)

    from affdet.detect.yolo_boxes import _find_scene_jsons as _f
    import json as _json
    trained_ids = set()
    for p in _f(scenes):
        if p.stem not in val_stems:                     # what the probe fit on
            for o in _json.loads(p.read_text())["objects"]:
                trained_ids.add(o["obj_id"])

    assert r["classification"]["subgroups"]["novel"]["probe"]["n"] > 0
    # no mesh the probe actually trained on may be labelled novel
    assert not (set(r["novel_mesh_ids"]) & trained_ids)


def test_seed_is_recorded_as_inert_under_split_from_dataset(scenes, tmp_path):
    """Three "seeds" of the real arm returned one identical number, because
    the seed only feeds scene_split_map -- which split_from_dataset bypasses
    -- and the probe (LogisticRegression, lbfgs, fixed data) is
    deterministic. Reporting those as three samples would invent error bars
    around a single measurement. The run must say so rather than leave the
    reader to infer it from suspiciously equal numbers."""
    stems = sorted(p.stem for p in __import__(
        "affdet.detect.yolo_boxes", fromlist=["x"])._find_scene_jsons(scenes))
    data_yaml = _write_dataset_yaml(tmp_path, scenes, set(stems[:60]))

    r = run_classification_eval(
        _cfg(scenes, data_yaml=str(data_yaml), split_from_dataset=True,
             seed=7),
        tmp_path / "root",
        detector_factory=lambda cfg: _perfect_detector,
        feature_fn=_separable_features,
        probe_factory=_probe_factory)

    assert r["seed_affects_result"] is False
