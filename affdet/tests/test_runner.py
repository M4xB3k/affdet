import json
from pathlib import Path

import numpy as np
import pytest

from affdet.experiment.config import ExperimentConfig
from affdet.experiment.runner import run_experiment
from affdet.render.mock import MockRenderer
from affdet.schemas import Scene, Features


def _scenes_and_labels(tmp_path):
    """MockRenderer writes scenes only; build a matching labels dir so the mock
    train/infer/eval path has labels to consume."""
    from affdet.schemas import Scene, Labels, LabelEntry
    s = tmp_path/"scenes"; MockRenderer(seed=0, n_scenes=8).run(s)
    lab = tmp_path/"labels"; lab.mkdir()
    for sj in sorted(s.glob("scene_*.json")):
        scene = Scene.model_validate_json(sj.read_text())
        entries = []
        for a in scene.objects:
            for b in scene.objects:
                if a.obj_id != b.obj_id:
                    entries.append(LabelEntry(actor=a.obj_id, recipient=b.obj_id,
                        predicate="canContain", verdict=1, confidence=0.9,
                        source="mock", score=0.9))
                    break
            break
        (lab/sj.name).write_text(Labels(scene_id=scene.scene_id, tier1=entries).model_dump_json())
    return s, lab


def test_runner_mock_end_to_end_writes_ledger(tmp_path):
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels), epochs=1)
    root = tmp_path/"exp"
    rec = run_experiment(cfg, root, mock=True)
    # metrics is split into raw (threshold=0.5) and calibrated (thresholds fit
    # on val) blocks -- the calibrated one is what the thesis reports.
    assert "macro_f1" in rec["metrics"]["raw"]
    assert (root/"runs.jsonl").exists()
    from affdet.experiment.ledger import load_runs
    assert len(load_runs(root/"runs.jsonl")) == 1
    assert rec["git_sha"]


def test_runner_evaluates_on_test_split_only(tmp_path, monkeypatch):
    """The reported metrics must come from the model's held-out TEST split
    only, not train+val+test combined -- otherwise F1 is inflated by scenes
    the model was trained/early-stopped on. Real (mock=False) path: heavy
    deps (DINOv2 extraction, real Trainer) are avoided the same way
    test_runner_real_branch_passes_resolved_instances_to_inferencer does
    (pre-seeded fake features, 1 epoch), but the real Inferencer + its new
    split="test" filtering run for real, so this actually exercises the fix."""
    scenes = tmp_path / "scenes"
    MockRenderer(seed=0, n_scenes=20).run(scenes)
    from affdet.schemas import Scene, Labels, LabelEntry
    lab = tmp_path / "labels"; lab.mkdir()
    for sj in sorted(scenes.glob("scene_*.json")):
        scene = Scene.model_validate_json(sj.read_text())
        entries = []
        for a in scene.objects:
            for b in scene.objects:
                if a.obj_id != b.obj_id:
                    entries.append(LabelEntry(actor=a.obj_id, recipient=b.obj_id,
                        predicate="canContain", verdict=1, confidence=0.9,
                        source="mock", score=0.9))
                    break
            break
        (lab / sj.name).write_text(Labels(scene_id=scene.scene_id, tier1=entries).model_dump_json())

    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(lab), epochs=1)
    root = tmp_path / "exp"

    feat_dir = root / "features" / cfg.extract_hash()
    feat_dir.mkdir(parents=True, exist_ok=True)
    for sj in sorted(scenes.glob("scene_*.json")):
        scene = Scene.model_validate_json(sj.read_text())
        Features(obj_ids=[o.obj_id for o in scene.objects],
                  features=np.zeros((len(scene.objects), 768), dtype=np.float32)
                  ).save(feat_dir / f"{sj.stem}.npz")

    import torch
    from affdet.model.train import Trainer
    from affdet.model.mlp import AffordanceMLP

    def _fake_fit(self, scenes_dir, features_dir, labels_dir, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    monkeypatch.setattr(Trainer, "fit", _fake_fit)

    rec = run_experiment(cfg, root, mock=False)

    from affdet.model.dataset import scene_split_map
    split_map = scene_split_map(scenes, seed=cfg.seed)
    test_scene_ids = {sid for sid, sp in split_map.items() if sp == "test"}

    pred_dir = root / "runs" / rec["run_id"] / "pred"
    written = {p.stem for p in pred_dir.glob("scene_*.json")}
    assert written == test_scene_ids
    assert len(written) < 20  # strictly a subset, proving it's not "score everything"


def test_runner_reuses_cached_features(tmp_path):
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels), epochs=1)
    root = tmp_path/"exp"
    run_experiment(cfg, root, mock=True)
    feat_dir = root/"features"/cfg.extract_hash()
    mtime = max(p.stat().st_mtime for p in feat_dir.glob("*.npz"))
    # second run, same extract_key -> must NOT re-extract (npz mtimes unchanged)
    run_experiment(ExperimentConfig(name="t2", scenes=str(scenes), labels=str(labels), epochs=1), root, mock=True)
    mtime2 = max(p.stat().st_mtime for p in feat_dir.glob("*.npz"))
    assert mtime2 == mtime


def test_runner_real_branch_passes_resolved_instances_to_inferencer(tmp_path, monkeypatch):
    """Regression test for I1: the real (mock=False) path must thread
    cfg.resolved_instances into the Inferencer, same as it does for the
    Trainer, so inference gates candidate pairs on effective class exactly
    like training and the production scripts/infer.py path.

    Real DINOv2 extraction is avoided by pre-seeding the extract-hash features
    dir with fake (2, 768) npz features so the runner's cache-skip check
    (`if not any(feat_dir.glob("*.npz"))`) short-circuits extraction. Training
    runs for real (1 epoch, tiny fixture). Only the Inferencer construction is
    intercepted, by monkeypatching the `Inferencer` symbol that
    affdet.model.infer exposes (the runner imports it locally at call time via
    `from affdet.model.infer import Inferencer`, so patching the source
    module's attribute is what the runner actually picks up).
    """
    scenes, labels = _scenes_and_labels(tmp_path)
    resolved = tmp_path / "resolved_instances.json"
    resolved.write_text(json.dumps({"knife_1": {"effective_class": "peeler"}}))

    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels),
                            epochs=1, resolved_instances=str(resolved))
    root = tmp_path / "exp"

    # Pre-seed the extract-hash features dir so the real DINOv2Extractor
    # (network call) is never invoked.
    feat_dir = root / "features" / cfg.extract_hash()
    feat_dir.mkdir(parents=True, exist_ok=True)
    for sj in sorted(scenes.glob("scene_*.json")):
        scene = Scene.model_validate_json(sj.read_text())
        Features(obj_ids=[o.obj_id for o in scene.objects],
                  features=np.zeros((len(scene.objects), 768), dtype=np.float32)
                  ).save(feat_dir / f"{sj.stem}.npz")

    captured = {}

    class _FakeInferencer:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def predict(self, *args, **kwargs):
            for p in args[-1] if args else []:
                pass  # predictions_dir already created by caller if needed

    import affdet.model.infer as infer_mod
    monkeypatch.setattr(infer_mod, "Inferencer", _FakeInferencer)

    # predictions_dir must exist with per-scene prediction files for eval to
    # read; write trivial empty Predictions so write_metrics_csv doesn't choke.
    from affdet.schemas import Predictions

    def _fake_predict(self, scenes_dir, features_dir, labels_dir, ckpt_dir, predictions_dir):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for sj in sorted(Path(scenes_dir).glob("scene_*.json")):
            scene = Scene.model_validate_json(sj.read_text())
            (predictions_dir / sj.name).write_text(
                Predictions(scene_id=scene.scene_id, tier1=[]).model_dump_json())

    _FakeInferencer.predict = _fake_predict

    run_experiment(cfg, root, mock=False)

    assert captured.get("resolved_instances") == cfg.resolved_instances


# --- H1/H2: runner must calibrate on val and record full metrics ------------

def test_runner_records_raw_and_calibrated_metrics(tmp_path):
    """The headline thesis number is the CALIBRATED one. If run_experiment
    only ever emits raw threshold-0.5 metrics, no committed code path can
    reproduce the reported result."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels), epochs=1)
    root = tmp_path/"exp"
    rec = run_experiment(cfg, root, mock=True)

    assert "raw" in rec["metrics"], "raw metrics block missing"
    assert "calibrated" in rec["metrics"], "calibrated metrics block missing"
    assert "macro_f1" in rec["metrics"]["raw"]
    assert "macro_f1" in rec["metrics"]["calibrated"]
    # the thresholds used must be recorded, not just the numbers they produced
    thr = rec["metrics"]["calibrated"]["thresholds"]
    assert isinstance(thr, dict) and thr, "calibrated thresholds not recorded"


def test_runner_records_per_predicate_metrics(tmp_path):
    """A per-predicate results table cannot be built from macro/micro alone."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels), epochs=1)
    rec = run_experiment(cfg, tmp_path/"exp", mock=True)

    per_pred = rec["metrics"]["raw"]["per_predicate"]
    assert "canContain" in per_pred
    for key in ("precision", "recall", "f1"):
        assert key in per_pred["canContain"]
    assert "counts" in rec["metrics"]["raw"]


def test_runner_records_labels_hash_and_split_sizes(tmp_path):
    """extract_hash() deliberately excludes `labels`, so without a separate
    label-corpus hash two runs on DIFFERENT label corpora are
    indistinguishable in the ledger -- exactly the v3/v3 naming collision that
    already produced one wrong results table."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels), epochs=1)
    rec = run_experiment(cfg, tmp_path/"exp", mock=True)

    assert rec.get("labels_hash"), "no label-corpus hash recorded"
    splits = rec.get("split_sizes")
    assert splits and {"train", "val", "test"} <= set(splits)
    assert rec.get("timestamp")


# --- E1c/E1c': runner must thread holdout_mode/holdout_frac through ---------

def test_runner_passes_holdout_mode_to_trainer_and_inferencer(tmp_path, monkeypatch):
    """ExperimentConfig.holdout_mode/holdout_frac must reach both Trainer.fit
    and Inferencer (test AND val calls), or E1c/E1c' configs silently fall
    back to split_strategy="scene" -- exactly the collapse H5 fixed."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels),
                            epochs=1, holdout_mode="strict", holdout_frac=0.3)
    root = tmp_path / "exp"

    feat_dir = root / "features" / cfg.extract_hash()
    feat_dir.mkdir(parents=True, exist_ok=True)
    for sj in sorted(scenes.glob("scene_*.json")):
        scene = Scene.model_validate_json(sj.read_text())
        Features(obj_ids=[o.obj_id for o in scene.objects],
                  features=np.zeros((len(scene.objects), 768), dtype=np.float32)
                  ).save(feat_dir / f"{sj.stem}.npz")

    trainer_kwargs = {}
    import affdet.model.train as train_mod
    real_init = train_mod.Trainer.__init__

    def _capturing_init(self, *args, **kwargs):
        trainer_kwargs.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(train_mod.Trainer, "__init__", _capturing_init)

    import torch
    from affdet.model.mlp import AffordanceMLP

    def _fake_fit(self, scenes_dir, features_dir, labels_dir, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    monkeypatch.setattr(train_mod.Trainer, "fit", _fake_fit)

    inferencer_calls = []

    class _FakeInferencer:
        def __init__(self, *args, **kwargs):
            inferencer_calls.append(kwargs)

        def predict(self, scenes_dir, features_dir, labels_dir, ckpt_dir, predictions_dir):
            predictions_dir = Path(predictions_dir)
            predictions_dir.mkdir(parents=True, exist_ok=True)
            from affdet.schemas import Predictions
            for sj in sorted(Path(scenes_dir).glob("scene_*.json")):
                scene = Scene.model_validate_json(sj.read_text())
                (predictions_dir / sj.name).write_text(
                    Predictions(scene_id=scene.scene_id, tier1=[]).model_dump_json())

    import affdet.model.infer as infer_mod
    monkeypatch.setattr(infer_mod, "Inferencer", _FakeInferencer)

    run_experiment(cfg, root, mock=False)

    assert trainer_kwargs.get("holdout_mode") == "strict"
    assert trainer_kwargs.get("holdout_frac") == 0.3
    assert len(inferencer_calls) == 2  # test split + val split
    for kwargs in inferencer_calls:
        assert kwargs.get("holdout_mode") == "strict"
        assert kwargs.get("holdout_frac") == 0.3


def test_runner_passes_all_pairs_to_inferencer(tmp_path, monkeypatch):
    """ExperimentConfig.all_pairs must reach both Inferencer calls (test AND
    val), or an "ungated" E1 arm silently evaluates through the gate anyway --
    unbounded=True already trains on every pair, so without this the eval
    denominator stays gated while training is not."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels),
                            epochs=1, unbounded=True, all_pairs=True)
    root = tmp_path / "exp"

    feat_dir = root / "features" / cfg.extract_hash()
    feat_dir.mkdir(parents=True, exist_ok=True)
    for sj in sorted(scenes.glob("scene_*.json")):
        scene = Scene.model_validate_json(sj.read_text())
        Features(obj_ids=[o.obj_id for o in scene.objects],
                  features=np.zeros((len(scene.objects), 768), dtype=np.float32)
                  ).save(feat_dir / f"{sj.stem}.npz")

    import torch
    from affdet.model.train import Trainer
    from affdet.model.mlp import AffordanceMLP

    def _fake_fit(self, scenes_dir, features_dir, labels_dir, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": AffordanceMLP().state_dict()}, ckpt_dir / "best.pt")

    monkeypatch.setattr(Trainer, "fit", _fake_fit)

    inferencer_calls = []

    class _FakeInferencer:
        def __init__(self, *args, **kwargs):
            inferencer_calls.append(kwargs)

        def predict(self, scenes_dir, features_dir, labels_dir, ckpt_dir, predictions_dir):
            predictions_dir = Path(predictions_dir)
            predictions_dir.mkdir(parents=True, exist_ok=True)
            from affdet.schemas import Predictions
            for sj in sorted(Path(scenes_dir).glob("scene_*.json")):
                scene = Scene.model_validate_json(sj.read_text())
                (predictions_dir / sj.name).write_text(
                    Predictions(scene_id=scene.scene_id, tier1=[]).model_dump_json())

    import affdet.model.infer as infer_mod
    monkeypatch.setattr(infer_mod, "Inferencer", _FakeInferencer)

    run_experiment(cfg, root, mock=False)

    assert len(inferencer_calls) == 2  # test split + val split
    for kwargs in inferencer_calls:
        assert kwargs.get("all_pairs") is True


def test_runner_records_holdout_config_in_ledger(tmp_path):
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels),
                            epochs=1, holdout_mode="strict", holdout_frac=0.3)
    rec = run_experiment(cfg, tmp_path / "exp", mock=True)

    assert rec["config"]["holdout_mode"] == "strict"
    assert rec["config"]["holdout_frac"] == 0.3


def test_runner_auto_excludes_zero_support_predicates_under_holdout_mode(tmp_path):
    """Found live 2026-07-30: E1c/E1c' seed0 both stranded canPeel with ZERO
    test pairs (an unlucky mesh-holdout draw), and compute_metrics scored it
    F1=0.0 and averaged it into the macro like a real failure, dragging the
    reported macro down by a real amount. Under holdout_mode, runner.py must
    pass auto_exclude_zero_support=True so a stranded predicate is excluded
    from the macro (still reported individually), not silently punished.

    The mock fixture (_scenes_and_labels) only emits canContain labels, so
    every other predicate is naturally zero-support here -- a real case, not
    a contrived one."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels),
                            epochs=1, holdout_mode="strict", holdout_frac=0.3)
    rec = run_experiment(cfg, tmp_path / "exp", mock=True)

    # canContain is the only predicate with data; everything else has zero
    # support and must be excluded from the macro, leaving canContain's own
    # F1 as the macro (not diluted 1/6 by five phantom zeros).
    raw = rec["metrics"]["raw"]
    assert raw["macro_f1"] == pytest.approx(raw["per_predicate"]["canContain"]["f1"])


def test_runner_records_held_out_meshes_under_holdout_mode(tmp_path):
    """split_sizes under holdout_mode must describe the actual pair-level
    holdout (which meshes were drawn), not a scene-level split_map that has
    nothing to do with what was actually trained on -- H2 reproducibility."""
    scenes, labels = _scenes_and_labels(tmp_path)
    cfg = ExperimentConfig(name="t", scenes=str(scenes), labels=str(labels),
                            epochs=1, holdout_mode="strict", holdout_frac=0.3)
    rec = run_experiment(cfg, tmp_path / "exp", mock=True)

    holdout = rec.get("holdout")
    assert holdout is not None, "no holdout record for a holdout_mode run"
    assert holdout["test_meshes"], "no meshes recorded as held out for test"
    assert "split_sizes" not in rec or rec["split_sizes"] is None or \
        set(rec.get("split_sizes", {})) <= {"train", "val", "test"}
