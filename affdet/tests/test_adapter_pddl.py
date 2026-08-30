from affdet.schemas import Predictions, PredictionEntry
from affdet.adapter.pddl_emitter import emit_pddl, predictions_to_pddl_strings


def test_emit_single_canCut_predicate():
    entry = PredictionEntry(
        actor="knife_0", recipient="banana_0", predicate="canCut",
        verdict=1, actor_part="edge", score=0.9,
    )
    s = predictions_to_pddl_strings([entry])
    assert "(canCut knife_0 edge banana_0)" in s


def test_emit_skips_negative_verdicts():
    entries = [
        PredictionEntry(actor="k_0", recipient="b_0", predicate="canCut",
                        verdict=0, actor_part=None, score=0.1),
    ]
    s = predictions_to_pddl_strings(entries)
    assert "(canCut" not in s


def test_emit_canContain_whole_object():
    entry = PredictionEntry(
        actor="bowl_0", recipient="apple_0", predicate="canContain",
        verdict=1, actor_part=None, score=0.85,
    )
    s = predictions_to_pddl_strings([entry])
    assert "(canContain bowl_0 apple_0)" in s


def test_emit_pddl_writes_file(tmp_path):
    preds = Predictions(scene_id="scene_0001", tier1=[
        PredictionEntry(actor="k_0", recipient="b_0", predicate="canCut",
                        verdict=1, actor_part="edge", score=0.9),
    ])
    preds_dir = tmp_path / "predictions"
    pddl_dir = tmp_path / "pddl"
    preds_dir.mkdir()
    pddl_dir.mkdir()
    (preds_dir / "scene_0001.json").write_text(preds.model_dump_json())
    emit_pddl(preds_dir, pddl_dir)
    out = (pddl_dir / "scene_0001.pddl").read_text()
    assert "(canCut k_0 edge b_0)" in out
    assert ":init" in out
