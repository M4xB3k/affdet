"""run_component_eval — E5 end to end, against an injected fake detector.

The detector is injected so the whole pipeline (split -> predict -> score ->
ledger) is testable without weights, a GPU, or the render corpus.
"""
import json

import pytest

from affdet.experiment.component_config import ComponentEvalConfig
from affdet.experiment.component_runner import run_component_eval
from affdet.render.mock import MockRenderer
from affdet.schemas import Scene


@pytest.fixture
def scenes(tmp_path):
    d = tmp_path / "scenes"
    MockRenderer(seed=0, n_scenes=20).run(d)
    return d


def _perfect_detector(image_path, scene):
    """Returns exactly the gold boxes -- an oracle detector."""
    return [{"bbox": list(o.bbox), "cls": o.obj_class, "conf": 0.99}
            for o in scene.objects]


def _blind_detector(image_path, scene):
    return []


def _cfg(scenes, **kw):
    base = dict(name="e5-test", scenes=str(scenes), detector="fake",
                weights="/fake.pt")
    return ComponentEvalConfig(**{**base, **kw})


def test_oracle_detector_scores_perfectly(scenes, tmp_path):
    r = run_component_eval(_cfg(scenes), tmp_path / "root",
                           detector_factory=lambda cfg: _perfect_detector)

    assert r["detection"]["f1"] == 1.0
    assert r["detection"]["fp"] == 0
    assert r["detection"]["fn"] == 0


def test_blind_detector_scores_zero(scenes, tmp_path):
    r = run_component_eval(_cfg(scenes), tmp_path / "root",
                           detector_factory=lambda cfg: _blind_detector)

    assert r["detection"]["tp"] == 0
    assert r["detection"]["recall"] == 0.0
    assert r["detection"]["n_gold"] > 0, "gold objects must still be counted"


def test_only_the_held_out_split_is_scored(scenes, tmp_path):
    """Scoring every scene would mix scenes the detector trained on into the
    reported number -- the same defect that was fixed for the affordance
    model."""
    r = run_component_eval(_cfg(scenes, split="test"), tmp_path / "root",
                           detector_factory=lambda cfg: _perfect_detector)

    all_scenes = 20
    assert 0 < r["detection"]["n_scenes"] < all_scenes
    assert r["split"] == "test"


def test_run_is_appended_to_the_ledger(scenes, tmp_path):
    root = tmp_path / "root"
    run_component_eval(_cfg(scenes), root,
                       detector_factory=lambda cfg: _perfect_detector)

    rows = [json.loads(l) for l in
            (root / "runs.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["kind"] == "component", \
        "E5 rows must be distinguishable from affordance rows in one ledger"
    assert rec["config"]["detector"] == "fake"
    assert "git_sha" in rec and "timestamp" in rec
    assert rec["metrics"]["detection"]["f1"] == 1.0


def test_ledger_records_the_scored_scene_set(scenes, tmp_path):
    """Without knowing WHICH scenes were scored the run is not reproducible."""
    root = tmp_path / "root"
    run_component_eval(_cfg(scenes), root,
                       detector_factory=lambda cfg: _perfect_detector)

    rec = json.loads((root / "runs.jsonl").read_text().splitlines()[0])
    assert rec["n_scenes_scored"] > 0
    assert rec["scenes_hash"], "content hash of the scored split"


def test_class_agnostic_detection_is_reported_alongside(scenes, tmp_path):
    """Box quality alone is the number that governs this pipeline, since
    crops are re-classified downstream. Both must be present."""
    def wrong_class(image_path, scene):
        return [{"bbox": list(o.bbox), "cls": "definitely_not_the_class",
                 "conf": 0.9} for o in scene.objects]

    r = run_component_eval(_cfg(scenes), tmp_path / "root",
                           detector_factory=lambda cfg: wrong_class)

    assert r["detection"]["f1"] == 0.0, "class-aware: every label is wrong"
    assert r["detection_class_agnostic"]["f1"] == 1.0, "boxes are perfect"


def test_deterministic_across_repeat_runs(scenes, tmp_path):
    """Detection against a fixed checkpoint and a fixed split has no random
    component; this is what licenses reporting it without seed error bars."""
    a = run_component_eval(_cfg(scenes), tmp_path / "a",
                           detector_factory=lambda cfg: _perfect_detector)
    b = run_component_eval(_cfg(scenes), tmp_path / "b",
                           detector_factory=lambda cfg: _perfect_detector)

    assert a["detection"] == b["detection"]


def test_different_seed_selects_a_different_split(scenes, tmp_path):
    a = run_component_eval(_cfg(scenes, seed=0), tmp_path / "a",
                           detector_factory=lambda cfg: _perfect_detector)
    b = run_component_eval(_cfg(scenes, seed=1), tmp_path / "b",
                           detector_factory=lambda cfg: _perfect_detector)

    assert a["scene_ids"] != b["scene_ids"]


def test_max_scenes_caps_the_run(scenes, tmp_path):
    r = run_component_eval(_cfg(scenes, max_scenes=2), tmp_path / "root",
                           detector_factory=lambda cfg: _perfect_detector)

    assert r["detection"]["n_scenes"] == 2


def test_per_class_results_survive_to_the_record(scenes, tmp_path):
    r = run_component_eval(_cfg(scenes), tmp_path / "root",
                           detector_factory=lambda cfg: _perfect_detector)

    assert r["detection"]["per_class"], "per-class breakdown is a deliverable"
    for stats in r["detection"]["per_class"].values():
        assert "precision" in stats and "recall" in stats and "f1" in stats


# ---------------------------------------------------------------------------
# Scene selection must honour an explicitly-built split
# ---------------------------------------------------------------------------

def test_eval_scenes_come_from_the_dataset_val_split_when_given(scenes, tmp_path):
    """The instance-disjoint arm's test set is defined by MESH IDENTITY at
    dataset-build time, not by a seeded scene shuffle. Falling back to
    scene_split_map there would score the detector on scenes it trained on
    -- the exact contamination this experiment exists to avoid."""
    from affdet.experiment.component_runner import _select_scenes

    ds = tmp_path / "ds"
    (ds / "images" / "val").mkdir(parents=True)
    (ds / "images" / "train").mkdir(parents=True)
    ds.joinpath("data.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n- apple\n")

    chosen = ["scene_0003", "scene_0007"]
    for stem in chosen:
        (ds / "images" / "val" / f"{stem}.png").touch()
    (ds / "images" / "train" / "scene_0001.png").touch()

    cfg = _cfg(scenes, data_yaml=str(ds / "data.yaml"), split_from_dataset=True)
    selected = [p.stem for p in _select_scenes(cfg)]

    assert sorted(selected) == sorted(chosen)


def test_split_from_dataset_is_opt_in(scenes, tmp_path):
    """Without the flag, behaviour is unchanged: the seeded scene split."""
    from affdet.experiment.component_runner import _select_scenes

    cfg = _cfg(scenes)
    selected = _select_scenes(cfg)

    assert 0 < len(selected) < 20


def test_novel_mesh_subgroup_is_reported_when_requested(scenes, tmp_path):
    """The D-b arm must separate novel meshes from the seen ones sharing
    their scenes, or its headline number is a blend it does not name."""
    import json

    from affdet.detect.yolo_boxes import _find_scene_jsons
    from affdet.experiment.component_runner import run_component_eval

    stems = sorted(p.stem for p in _find_scene_jsons(scenes))
    val_stems, train_stems = set(stems[:3]), set(stems[3:])

    ds = tmp_path / "ds"
    (ds / "images" / "val").mkdir(parents=True)
    (ds / "images" / "train").mkdir(parents=True)
    ds.joinpath("data.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n- apple\n")
    for s in val_stems:
        (ds / "images" / "val" / f"{s}.png").touch()
    for s in train_stems:
        (ds / "images" / "train" / f"{s}.png").touch()

    cfg = _cfg(scenes, data_yaml=str(ds / "data.yaml"),
               split_from_dataset=True, split_novel_meshes=True)
    r = run_component_eval(cfg, tmp_path / "root",
                           detector_factory=lambda c: _perfect_detector)

    groups = r["detection"]["subgroups"]
    assert set(groups) == {"novel", "rest"}
    # every gold object is accounted for in exactly one group
    assert (groups["novel"]["n_gold"] + groups["rest"]["n_gold"]
            == r["detection"]["n_gold"])


def test_novel_mesh_subgroup_is_opt_in(scenes, tmp_path):
    from affdet.experiment.component_runner import run_component_eval

    r = run_component_eval(_cfg(scenes), tmp_path / "root2",
                           detector_factory=lambda c: _perfect_detector)
    assert "subgroups" not in r["detection"]
