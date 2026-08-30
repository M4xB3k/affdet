"""H3/H4 — seed-family aggregation over the ledger, and LaTeX table export.

A thesis reports mean±std over seeds, not one arbitrary run, and its tables
must be generated from the ledger rather than transcribed by hand (this
project already shipped one wrong results table from stale hand-copied
numbers).
"""
import json

import pytest

from affdet.experiment.aggregate import (
    aggregate_seed_families, seed_family_key, write_results_table,
)


def _rec(name, seed, macro_raw, macro_cal, cancut_f1, labels_hash="abc123"):
    """Minimal ledger record shaped like the real one."""
    return {
        "run_id": f"{name}-s{seed}-hash-sha",
        "git_sha": "deadbeef",
        "labels_hash": labels_hash,
        "config": {"name": name, "seed": seed, "head": "mlp",
                   "backbone": "vitb14", "crop_padding": 0.0,
                   "bbox_source": "oracle", "scenes": "/s", "labels": "/l"},
        "metrics": {
            "raw": {"macro_f1": macro_raw,
                    "per_predicate": {"canCut": {"precision": 0.9, "recall": 0.9,
                                                 "f1": cancut_f1}}},
            "calibrated": {"macro_f1": macro_cal,
                           "thresholds": {"canCut": 0.25},
                           "per_predicate": {"canCut": {"precision": 0.95,
                                                        "recall": 0.95,
                                                        "f1": cancut_f1}}},
        },
    }


def test_seed_family_key_ignores_seed():
    """Two runs differing only in seed belong to the same family."""
    a = _rec("arm", 0, 0.90, 0.95, 0.8)
    b = _rec("arm", 1, 0.92, 0.96, 0.8)
    assert seed_family_key(a) == seed_family_key(b)


def test_seed_family_key_separates_different_arms():
    a = _rec("arm", 0, 0.90, 0.95, 0.8)
    b = _rec("arm", 0, 0.90, 0.95, 0.8)
    b["config"]["head"] = "attention"
    assert seed_family_key(a) != seed_family_key(b)


def test_seed_family_key_separates_different_label_corpora():
    """The v3/v3 collision: same arm name, different label corpus, must not
    be averaged together."""
    a = _rec("arm", 0, 0.90, 0.95, 0.8, labels_hash="aaa")
    b = _rec("arm", 1, 0.90, 0.95, 0.8, labels_hash="bbb")
    assert seed_family_key(a) != seed_family_key(b)


def test_seed_family_key_separates_holdout_mode():
    """Found live 2026-07-30: E1a (split_strategy=scene, holdout_mode=None)
    and E1c (holdout_mode=strict) share the same split_strategy default and
    labels_hash, so without holdout_mode/holdout_frac in _ARM_FIELDS they
    were silently averaged into one 6-seed family instead of two separate
    3-seed arms -- E1c's row vanished from the report entirely."""
    a = _rec("e1a-scene-s0", 0, 0.90, 0.95, 0.8)
    a["config"]["split_strategy"] = "scene"
    a["config"]["holdout_mode"] = None
    a["config"]["holdout_frac"] = 0.20
    b = _rec("e1c-strict-s0", 0, 0.75, 0.80, 0.6)
    b["config"]["split_strategy"] = "scene"
    b["config"]["holdout_mode"] = "strict"
    b["config"]["holdout_frac"] = 0.20
    assert seed_family_key(a) != seed_family_key(b)


def test_seed_family_key_separates_different_holdout_fracs():
    a = _rec("e1c-strict-20", 0, 0.75, 0.80, 0.6)
    a["config"]["holdout_mode"] = "strict"
    a["config"]["holdout_frac"] = 0.20
    b = _rec("e1c-strict-30", 0, 0.70, 0.78, 0.55)
    b["config"]["holdout_mode"] = "strict"
    b["config"]["holdout_frac"] = 0.30
    assert seed_family_key(a) != seed_family_key(b)


def test_seed_family_key_separates_all_pairs():
    """all_pairs=True (ungated eval) is not comparable to the gated ladder --
    a different denominator entirely -- so it must land in its own arm rather
    than silently merging with the gated run of the same name/config."""
    a = _rec("e1a-scene-s0", 0, 0.90, 0.95, 0.8)
    a["config"]["unbounded"] = True
    a["config"]["all_pairs"] = False
    b = _rec("e1a-scene-s0", 0, 0.60, 0.65, 0.5)
    b["config"]["unbounded"] = True
    b["config"]["all_pairs"] = True
    assert seed_family_key(a) != seed_family_key(b)


def test_aggregate_computes_mean_and_std():
    runs = [_rec("arm", 0, 0.90, 0.94, 0.80),
            _rec("arm", 1, 0.92, 0.96, 0.84),
            _rec("arm", 2, 0.94, 0.98, 0.88)]
    fams = aggregate_seed_families(runs)
    assert len(fams) == 1
    fam = fams[0]
    assert fam["n_seeds"] == 3
    assert fam["seeds"] == [0, 1, 2]
    assert fam["calibrated"]["macro_f1"]["mean"] == pytest.approx(0.96)
    assert fam["calibrated"]["macro_f1"]["std"] == pytest.approx(0.0163299, abs=1e-5)
    assert fam["raw"]["macro_f1"]["mean"] == pytest.approx(0.92)
    # per-predicate aggregates too, since the results table is per-predicate
    assert fam["calibrated"]["per_predicate"]["canCut"]["f1"]["mean"] == pytest.approx(0.84)


def test_aggregate_single_seed_has_zero_std():
    fams = aggregate_seed_families([_rec("arm", 0, 0.9, 0.95, 0.8)])
    assert fams[0]["n_seeds"] == 1
    assert fams[0]["calibrated"]["macro_f1"]["std"] == 0.0


def test_write_results_table_emits_latex(tmp_path):
    runs = [_rec("arm", 0, 0.90, 0.94, 0.80),
            _rec("arm", 1, 0.92, 0.96, 0.84)]
    out = tmp_path / "tables" / "results.tex"
    write_results_table(aggregate_seed_families(runs), out)
    tex = out.read_text()
    assert "\\begin{tabular}" in tex and "\\end{tabular}" in tex
    assert "booktabs" not in tex or "\\toprule" in tex
    assert "canCut" in tex
    # mean +/- std rendered
    assert "0.82" in tex or "0.820" in tex


def test_aggregate_computes_score_fidelity_mean_and_std():
    """E1d: score MAE/corr vs the VLM teacher, aggregated across seeds --
    same shape as per_predicate F1, just a different metric already computed
    per-run and previously dropped at the aggregation boundary."""
    runs = [_rec("arm", 0, 0.90, 0.94, 0.80), _rec("arm", 1, 0.92, 0.96, 0.84)]
    runs[0]["metrics"]["calibrated"]["score_fidelity"] = {
        "canCut": {"mae": 0.10, "corr": 0.90, "n": 100}}
    runs[1]["metrics"]["calibrated"]["score_fidelity"] = {
        "canCut": {"mae": 0.14, "corr": 0.86, "n": 120}}
    fam = aggregate_seed_families(runs)[0]
    fid = fam["calibrated"]["score_fidelity"]["canCut"]
    assert fid["mae"]["mean"] == pytest.approx(0.12)
    assert fid["corr"]["mean"] == pytest.approx(0.88)


def test_aggregate_score_fidelity_skips_none_corr():
    """canPierce has zero negatives, so corr is None (undefined, not zero) --
    that must not be averaged in as if it were a real correlation of 0."""
    runs = [_rec("arm", 0, 0.90, 0.94, 0.80)]
    runs[0]["metrics"]["calibrated"]["score_fidelity"] = {
        "canPierce": {"mae": 0.5, "corr": None, "n": 50}}
    fam = aggregate_seed_families(runs)[0]
    fid = fam["calibrated"]["score_fidelity"]["canPierce"]
    assert fid["mae"]["mean"] == pytest.approx(0.5)
    assert fid["corr"]["mean"] is None


def test_results_table_marks_degenerate_predicate(tmp_path):
    """canPierce must be visibly flagged as excluded from the macro, so a
    reader cannot mistake its artifactual 1.000 for performance."""
    r = _rec("arm", 0, 0.90, 0.94, 0.80)
    for block in ("raw", "calibrated"):
        r["metrics"][block]["per_predicate"]["canPierce"] = {
            "precision": 1.0, "recall": 1.0, "f1": 1.0}
        r["metrics"][block]["macro_excluded"] = ["canPierce"]
    out = tmp_path / "results.tex"
    write_results_table(aggregate_seed_families([r]), out)
    tex = out.read_text()
    assert "canPierce" in tex
    assert "excluded" in tex.lower()


def test_results_table_reports_score_fidelity_column(tmp_path):
    """E1d: score MAE vs the VLM teacher must appear in the results table --
    otherwise the fidelity numbers stay buried in the JSON and never reach
    the thesis, defeating the point of committing them to the ledger."""
    r = _rec("arm", 0, 0.90, 0.94, 0.80)
    r["metrics"]["calibrated"]["score_fidelity"] = {
        "canCut": {"mae": 0.12, "corr": 0.88, "n": 100}}
    out = tmp_path / "results.tex"
    write_results_table(aggregate_seed_families([r]), out)
    tex = out.read_text()
    assert "0.12" in tex
    assert "0.88" in tex
