from pathlib import Path
from affdet.schemas import Labels, LabelEntry, Predictions, PredictionEntry
from affdet.eval.metrics import compute_metrics


def _write(tmp_path):
    labs = tmp_path/"labs"; labs.mkdir()
    preds = tmp_path/"preds"; preds.mkdir()
    L = [LabelEntry(actor="k_1", recipient="a_1", predicate="canCut", verdict=1,
                    confidence=0.9, source="t", score=0.8),
         LabelEntry(actor="k_1", recipient="b_1", predicate="canCut", verdict=0,
                    confidence=0.1, source="t", score=0.2)]
    P = [PredictionEntry(actor="k_1", recipient="a_1", predicate="canCut",
                         verdict=1, score=0.7),
         PredictionEntry(actor="k_1", recipient="b_1", predicate="canCut",
                         verdict=0, score=0.3)]
    (labs/"scene_0001.json").write_text(Labels(scene_id="scene_0001", tier1=L).model_dump_json())
    (preds/"scene_0001.json").write_text(Predictions(scene_id="scene_0001", tier1=P).model_dump_json())
    return preds, labs


def test_score_fidelity_reports_mae(tmp_path):
    preds, labs = _write(tmp_path)
    m = compute_metrics(preds, labs)
    fid = m["score_fidelity"]["canCut"]
    # |0.8-0.7| and |0.2-0.3| -> MAE 0.1
    assert abs(fid["mae"] - 0.1) < 1e-6
    assert fid["n"] == 2


def test_binary_metrics_unchanged(tmp_path):
    preds, labs = _write(tmp_path)
    m = compute_metrics(preds, labs)
    assert "per_predicate" in m and "macro_f1" in m


def test_can_pour_into_excluded_from_score_fidelity(tmp_path):
    labs = tmp_path/"labs"; labs.mkdir()
    preds = tmp_path/"preds"; preds.mkdir()
    L = [LabelEntry(actor="bottle_1", recipient="cup_1", predicate="canPourInto",
                    verdict=1, confidence=0.9, source="t", score=1.0)]
    P = [PredictionEntry(actor="bottle_1", recipient="cup_1", predicate="canPourInto",
                         verdict=1, score=0.9)]
    (labs/"scene_0001.json").write_text(Labels(scene_id="scene_0001", tier1=L).model_dump_json())
    (preds/"scene_0001.json").write_text(Predictions(scene_id="scene_0001", tier1=P).model_dump_json())

    m = compute_metrics(preds, labs)

    assert "canPourInto" not in m["score_fidelity"]
