import pytest
from affdet.schemas import Labels, LabelEntry, Predictions, PredictionEntry
from affdet.eval.metrics import compute_metrics, write_metrics_csv, calibrate_thresholds


def _mk_label(actor, recipient, predicate, verdict):
    return LabelEntry(actor=actor, recipient=recipient, predicate=predicate,
                      verdict=verdict, actor_part="edge" if verdict == 1 else None,
                      confidence=1.0, source="mock")


def _mk_pred(actor, recipient, predicate, verdict, score=0.5):
    return PredictionEntry(actor=actor, recipient=recipient, predicate=predicate,
                           verdict=verdict, actor_part="edge" if verdict == 1 else None,
                           score=score)


def test_metrics_perfect_predictions(tmp_path):
    labels_dir = tmp_path / "labels"
    preds_dir = tmp_path / "predictions"
    labels_dir.mkdir()
    preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())

    metrics = compute_metrics(preds_dir, labels_dir)
    assert metrics["per_predicate"]["canCut"]["f1"] == 1.0
    # macro averages over all 6 predicates; only canCut has data here so macro = 1/6
    assert metrics["macro_f1"] == pytest.approx(1.0 / 6, abs=1e-9)


def test_metrics_all_wrong(tmp_path):
    labels_dir = tmp_path / "labels"
    preds_dir = tmp_path / "predictions"
    labels_dir.mkdir()
    preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 0)],
    ).model_dump_json())

    metrics = compute_metrics(preds_dir, labels_dir)
    assert metrics["per_predicate"]["canCut"]["f1"] == 0.0


def test_calibrate_thresholds_finds_best_f1_threshold(tmp_path):
    """canCut ground truth is all-positive except one hard negative; raw
    scores are high for true positives, low for the one negative, but the
    default 0.5 cutoff would mislabel a borderline true positive sitting at
    0.4 -- a lower threshold recovers it with better F1."""
    labels_dir = tmp_path / "labels"
    preds_dir = tmp_path / "predictions"
    labels_dir.mkdir()
    preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[
            _mk_label("k_0", "a_0", "canCut", 1),
            _mk_label("k_0", "a_1", "canCut", 1),
            _mk_label("k_0", "a_2", "canCut", 1),
            _mk_label("k_0", "a_3", "canCut", 0),
        ],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[
            _mk_pred("k_0", "a_0", "canCut", 1, score=0.9),
            _mk_pred("k_0", "a_1", "canCut", 1, score=0.8),
            # This one scores 0.4 -- below the default 0.5 threshold, so it
            # would be a false negative at threshold=0.5, but a lower
            # threshold (e.g. 0.35) recovers it as a true positive.
            _mk_pred("k_0", "a_2", "canCut", 0, score=0.4),
            _mk_pred("k_0", "a_3", "canCut", 0, score=0.1),
        ],
    ).model_dump_json())

    thresholds = calibrate_thresholds(preds_dir, labels_dir)

    assert thresholds["canCut"] <= 0.4
    # At the calibrated threshold, all 4 instances should be correctly
    # classified (3 TP, 1 TN) -> F1 = 1.0 for canCut.
    metrics = compute_metrics(preds_dir, labels_dir, thresholds=thresholds)
    assert metrics["per_predicate"]["canCut"]["f1"] == pytest.approx(1.0)


def test_calibrate_thresholds_defaults_to_half_when_no_data(tmp_path):
    labels_dir = tmp_path / "labels"
    preds_dir = tmp_path / "predictions"
    labels_dir.mkdir()
    preds_dir.mkdir()

    thresholds = calibrate_thresholds(preds_dir, labels_dir)
    assert thresholds["canCut"] == 0.5


def test_compute_metrics_without_thresholds_uses_stored_verdict(tmp_path):
    """Backwards compatible: no thresholds arg -> behaves exactly as before,
    trusting each PredictionEntry's own already-thresholded verdict."""
    labels_dir = tmp_path / "labels"
    preds_dir = tmp_path / "predictions"
    labels_dir.mkdir()
    preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1, score=0.3)],
    ).model_dump_json())

    metrics = compute_metrics(preds_dir, labels_dir)
    assert metrics["per_predicate"]["canCut"]["f1"] == 1.0


def test_metrics_csv_written(tmp_path):
    labels_dir = tmp_path / "labels"
    preds_dir = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    labels_dir.mkdir()
    preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())

    write_metrics_csv(preds_dir, labels_dir, results_dir)
    csv_path = results_dir / "metrics.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "canCut" in content
    assert "precision" in content


# --- H1: write_metrics_csv must accept + apply calibrated thresholds --------

def test_write_metrics_csv_applies_thresholds(tmp_path):
    """A prediction scoring 0.3 with a stored verdict of 0 must count as a
    POSITIVE when a calibrated threshold of 0.2 is supplied -- i.e. the CSV
    reflects the calibrated cutoff, not the verdict baked in at inference."""
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 0, score=0.3)],
    ).model_dump_json())

    # raw (no thresholds): the 0.5-cutoff verdict=0 makes this a false negative
    raw = compute_metrics(preds_dir, labels_dir)
    assert raw["per_predicate"]["canCut"]["f1"] == 0.0

    csv_path = write_metrics_csv(preds_dir, labels_dir, results_dir,
                                 thresholds={"canCut": 0.2})
    rows = [r.split(",") for r in csv_path.read_text().splitlines() if r]
    cancut = next(r for r in rows if r[0] == "canCut")
    assert float(cancut[3]) == 1.0, "calibrated threshold was not applied"


def test_write_metrics_csv_records_thresholds_used(tmp_path):
    """The thresholds themselves must be persisted -- a thesis has to report
    the decision cutoff, and it is currently saved nowhere."""
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1, score=0.9)],
    ).model_dump_json())

    write_metrics_csv(preds_dir, labels_dir, results_dir,
                      thresholds={"canCut": 0.25})
    import json
    saved = json.loads((results_dir / "thresholds.json").read_text())
    assert saved["canCut"] == 0.25


# --- H6: canPierce has zero negatives corpus-wide, so its F1 is an artifact --

def test_macro_f1_excludes_degenerate_predicate(tmp_path):
    """canPierce has no negatives in the corpus, so it cannot be got wrong and
    its F1=1.0 silently inflates the macro average. compute_metrics must be
    able to exclude named predicates from the macro while still reporting
    them individually."""
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[
            _mk_label("k_0", "b_0", "canCut", 1),
            _mk_label("k_0", "b_0", "canPierce", 1),
        ],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[
            _mk_pred("k_0", "b_0", "canCut", 0),      # wrong -> f1 0.0
            _mk_pred("k_0", "b_0", "canPierce", 1),   # "right" -> f1 1.0
        ],
    ).model_dump_json())

    m = compute_metrics(preds_dir, labels_dir, macro_exclude=["canPierce"])
    # canPierce still reported individually...
    assert m["per_predicate"]["canPierce"]["f1"] == 1.0
    # ...but excluded from the macro, which now averages the other 5
    assert m["macro_f1"] == pytest.approx(0.0, abs=1e-9)


def test_zero_support_predicate_is_not_auto_excluded_by_default(tmp_path):
    """Existing, deliberate behaviour (test_metrics_perfect_predictions) must
    not move: with NO opt-in, a predicate absent from the test split scores
    F1=0.0 and is averaged into the macro like a real failure."""
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())

    m = compute_metrics(preds_dir, labels_dir)
    assert m["per_predicate"]["canPeel"]["f1"] == 0.0
    assert m["macro_f1"] == pytest.approx(1.0 / 6, abs=1e-9)


def test_auto_exclude_zero_support_drops_predicates_with_no_test_pairs(tmp_path):
    """A holdout draw can strand a low-instance predicate (canPeel: only 4
    actor meshes corpus-wide) with ZERO test pairs -- found live 2026-07-30,
    E1c/E1c' seed0 both scored canPeel F1=0.0 with tp=fp=fn=tn=0. That is not
    a wrong prediction, it is no evidence at all, and must not drag the macro
    down like a real failure -- same reasoning as canPierce's exclusion, but
    determined dynamically per run instead of a static name list."""
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())

    m = compute_metrics(preds_dir, labels_dir, auto_exclude_zero_support=True)
    # canPeel still reported individually (as 0.0, since there's nothing to
    # compute) but must not enter the macro average.
    assert m["per_predicate"]["canPeel"]["f1"] == 0.0
    assert m["macro_f1"] == pytest.approx(1.0, abs=1e-9)  # only canCut, which is perfect


def test_auto_exclude_zero_support_combines_with_macro_exclude(tmp_path):
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[
            _mk_label("k_0", "b_0", "canCut", 1),
            _mk_label("k_0", "b_0", "canPierce", 1),
        ],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[
            _mk_pred("k_0", "b_0", "canCut", 1),
            _mk_pred("k_0", "b_0", "canPierce", 1),
        ],
    ).model_dump_json())

    m = compute_metrics(preds_dir, labels_dir, macro_exclude=["canPierce"],
                        auto_exclude_zero_support=True)
    # canPierce excluded by name, canPeel/canContain/canSupport/canPourInto
    # excluded for zero support -- only canCut left, which is perfect.
    assert m["macro_f1"] == pytest.approx(1.0, abs=1e-9)
    assert "canPierce" in m["macro_excluded"]
    assert "canPeel" in m["macro_excluded"]


def test_write_metrics_csv_accepts_auto_exclude_zero_support(tmp_path):
    """write_metrics_csv (what runner.py actually calls) must expose the same
    opt-in, not just compute_metrics."""
    labels_dir = tmp_path / "labels"; preds_dir = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    labels_dir.mkdir(); preds_dir.mkdir()
    (labels_dir / "s1.json").write_text(Labels(
        scene_id="s1", tier1=[_mk_label("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())
    (preds_dir / "s1.json").write_text(Predictions(
        scene_id="s1", tier1=[_mk_pred("k_0", "b_0", "canCut", 1)],
    ).model_dump_json())

    csv_path = write_metrics_csv(preds_dir, labels_dir, results_dir,
                                 auto_exclude_zero_support=True)
    rows = [r.split(",") for r in csv_path.read_text().splitlines() if r]
    macro_row = next(r for r in rows if r[0] == "MACRO")
    assert float(macro_row[3]) == pytest.approx(1.0, abs=1e-9)
