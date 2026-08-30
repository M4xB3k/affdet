"""Exporting E1 ladder + E7 arm rows to the thesis tables.

The rule these tests defend is Chapter 6's: every number reaches the thesis
via ``\\input{tables/...}`` generated from the ledger, never transcribed.
The failure they exist to prevent is a table that silently averages a
superseded run, or one that emits a plausible number where there is no
evidence at all.
"""
import json

import pytest

from affdet.experiment.result_tables import (
    ladder_rows,
    write_e7_arms_tex,
    write_e7_significance_tex,
    write_per_predicate_tex,
)

PREDICATES = ("canContain", "canSupport", "canCut", "canPourInto",
              "canPeel", "canPierce")


def _block(f1, support=100, preds=PREDICATES):
    """One metrics block (raw or calibrated) over the given predicates."""
    return {
        "macro_f1": f1,
        "per_predicate": {
            p: {"precision": f1, "recall": f1, "f1": f1} for p in preds
        },
        "counts": {
            p: {"tp": support // 2, "fp": 0, "fn": 0, "tn": support // 2}
            for p in preds
        },
    }


def _run(name, f1=0.9, support=100, preds=PREDICATES):
    return {
        "kind": "affordance",
        "run_id": f"{name}-abc123",
        "git_sha": "deadbeef",
        "config": {"name": name, "seed": 0},
        "metrics": {"raw": _block(f1 - 0.02, support, preds),
                    "calibrated": _block(f1, support, preds)},
    }


@pytest.fixture
def snapshot(tmp_path):
    p = tmp_path / "snap.jsonl"
    rows = [
        _run("e1a-scene-s0-v4", 0.90),
        _run("e1a-scene-s1-v4", 0.94),
        # A superseded rung: the -v4fix rerun must retire its -v4 sibling.
        _run("e1c-strict-s0-v4", 0.50),
        _run("e1c-strict-s0-v4fix", 0.86),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# --------------------------------------------------------------------
# Loading: the supersede rule
# --------------------------------------------------------------------

def test_v4fix_run_supersedes_its_v4_sibling(snapshot):
    """A rung with any -v4fix run ignores its -v4 runs entirely.

    Averaging the two together is how this project once produced a wrong
    6-seed family; the fixed-holdout rerun replaces, never blends.
    """
    rungs = ladder_rows(snapshot)
    strict = rungs["e1c-strict"]
    assert len(strict) == 1
    assert strict[0]["config"]["name"] == "e1c-strict-s0-v4fix"


def test_unsuperseded_rung_keeps_all_its_seeds(snapshot):
    assert len(ladder_rows(snapshot)["e1a-scene"]) == 2


# --------------------------------------------------------------------
# Per-predicate table
# --------------------------------------------------------------------

def test_per_predicate_emits_both_macro_rows(snapshot, tmp_path):
    """Both macros ship: over all six, and over the five VLM-taught ones.

    canPourInto is supervised by a hand-authored rule rather than the
    teacher, so averaging it in mixes 'reproduces a teacher' with 'recovers
    a hand-authored rule'. Reporting both defuses that in the table itself.
    """
    out = write_per_predicate_tex(snapshot, tmp_path / "t.tex", rung="e1a-scene")
    tex = out.read_text()
    assert "Macro (6 pred.)" in tex
    assert "Macro (5 pred." in tex
    assert "canPourInto" in tex


def test_per_predicate_reports_raw_and_calibrated(snapshot, tmp_path):
    tex = write_per_predicate_tex(
        snapshot, tmp_path / "t.tex", rung="e1a-scene").read_text()
    assert "Raw" in tex and "Calibrated" in tex


def test_per_predicate_seed_averages_rather_than_taking_one(snapshot, tmp_path):
    """Two seeds at 0.90 and 0.94 must report their mean, not either endpoint.

    Checked on the calibrated columns specifically: the fixture offsets raw
    by -0.02, so 0.900 legitimately appears there and a document-wide
    substring check would pass for the wrong reason.
    """
    tex = write_per_predicate_tex(
        snapshot, tmp_path / "t.tex", rung="e1a-scene").read_text()
    row = [ln for ln in tex.splitlines() if "canContain" in ln][0]
    _, raw_p, raw_r, raw_f, cal_p, cal_r, cal_f = [
        c.strip() for c in row.split("&")]
    assert (cal_p, cal_r, cal_f.rstrip(" \\")) == ("0.920", "0.920", "0.920")
    assert raw_f.strip() == "0.900"  # mean of 0.88 and 0.92, not an endpoint


def test_zero_support_predicate_is_excluded_not_scored_zero(tmp_path):
    """No test pairs is 'no evidence', not a failure.

    canPeel on an instance split can land with zero support; scoring that
    as 0.0 would drag the macro down and read as a model failure.
    """
    p = tmp_path / "s.jsonl"
    run = _run("e1a-scene-s0-v4", 0.9)
    for block in ("raw", "calibrated"):
        run["metrics"][block]["counts"]["canPeel"] = {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        run["metrics"][block]["per_predicate"]["canPeel"]["f1"] = 0.0
    p.write_text(json.dumps(run) + "\n")

    tex = write_per_predicate_tex(p, tmp_path / "t.tex",
                                  rung="e1a-scene").read_text()
    peel = [ln for ln in tex.splitlines() if "canPeel" in ln][0]
    assert "--" in peel          # no evidence, printed as a dash
    assert "0.000" not in peel   # never as a zero score


def test_export_refuses_a_rung_with_no_runs(snapshot, tmp_path):
    """Better no table than a table of dashes that looks like a measurement."""
    with pytest.raises(ValueError, match="no runs"):
        write_per_predicate_tex(snapshot, tmp_path / "t.tex", rung="e1b-pair")


def test_per_predicate_is_a_float_with_a_label(snapshot, tmp_path):
    """It must be \\input-able as a table, not a bare tabular.

    e1_ladder_v4.tex emits a bare tabular, which is exactly why no chapter
    \\input s it — there is no caption or label to reference.
    """
    tex = write_per_predicate_tex(
        snapshot, tmp_path / "t.tex", rung="e1a-scene").read_text()
    assert "\\begin{table}" in tex
    assert "\\caption" in tex
    assert "\\label{tab:per-predicate}" in tex


def test_generated_files_warn_against_hand_editing(snapshot, tmp_path):
    tex = write_per_predicate_tex(
        snapshot, tmp_path / "t.tex", rung="e1a-scene").read_text()
    assert "GENERATED" in tex and "DO NOT EDIT" in tex.upper()


# --------------------------------------------------------------------
# E7 arms table
# --------------------------------------------------------------------

@pytest.fixture
def arms_json(tmp_path):
    def _metrics(acc):
        return {"n": 593, "accuracy": acc, "balanced_accuracy": acc - 0.05,
                "precision": 0.83, "recall": 0.87, "f1": 0.85,
                "mae": 2.09, "spearman": 0.39}
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({
        "human_judgements": 593,
        "excluded_predicates": None,
        "arms": {
            "gplus": {"n_passes": 3, "n_scored": 593,
                      "vlm_median": _metrics(0.788),
                      "vlm_mean": _metrics(0.803),
                      "student": _metrics(0.678)},
            "g": {"n_passes": 3, "n_scored": 577,
                  "vlm_median": _metrics(0.766),
                  "vlm_mean": _metrics(0.771),
                  "student": _metrics(0.679)},
            "e": {"n_passes": 3, "n_scored": 431,
                  "vlm_median": _metrics(0.763),
                  "vlm_mean": _metrics(0.780),
                  "student": _metrics(0.687)},
        },
    }))
    return p


def test_e7_arms_orders_the_ladder_gplus_then_g_then_e(arms_json, tmp_path):
    """Each row down restores one pipeline stage; the order is the argument."""
    tex = write_e7_arms_tex(arms_json, tmp_path / "a.tex").read_text()
    body = tex[tex.index("midrule"):]
    assert body.index("G+") < body.index("{G}") < body.index("{E}")


def test_e7_arms_pairs_each_vlm_row_with_its_student_row(arms_json, tmp_path):
    tex = write_e7_arms_tex(arms_json, tmp_path / "a.tex").read_text()
    assert tex.count("VLM") >= 3
    assert tex.count("student") >= 3


def test_e7_arms_uses_median_and_says_so(arms_json, tmp_path):
    """Median is the reported aggregate; the caption must not leave it implicit."""
    tex = write_e7_arms_tex(arms_json, tmp_path / "a.tex").read_text()
    assert "0.788" in tex     # the median value
    assert "0.803" not in tex  # not the mean
    assert "median" in tex.lower()


def test_e7_arms_caption_carries_the_familiar_mesh_caveat(arms_json, tmp_path):
    """A caption is read detached from the doc that holds the caveat.

    E7 is scene-disjoint but NOT instance-disjoint: novel scenes, familiar
    objects. Omitting that from the caption overstates the claim.
    """
    tex = write_e7_arms_tex(arms_json, tmp_path / "a.tex").read_text()
    low = tex.lower()
    assert "scene" in low and ("familiar" in low or "not instance" in low)


# --------------------------------------------------------------------
# Paired bootstrap table
# --------------------------------------------------------------------

def _sig(diff, lo, hi, sig):
    return {"student": 0.68, "vlm": 0.76, "diff": diff,
            "ci_low": lo, "ci_high": hi, "significant": sig}


@pytest.fixture
def sig_jsons(tmp_path):
    full = tmp_path / "sig.json"
    full.write_text(json.dumps({
        "arm": "e", "n_pairs": 431, "n_boot": 10000,
        "excluded_predicates": None,
        "metrics": {
            "accuracy": _sig(-0.0766, -0.1183, -0.0348, True),
            "balanced_accuracy": _sig(-0.0726, -0.1119, -0.0335, True),
            "f1": _sig(-0.0644, -0.1023, -0.0278, True),
            "mae": _sig(0.3856, 0.1483, 0.6245, True),
            "spearman": _sig(-0.0714, -0.1508, 0.0062, False),
        }}))
    nop = tmp_path / "sig_nop.json"
    nop.write_text(json.dumps({
        "arm": "e", "n_pairs": 300, "n_boot": 10000,
        "excluded_predicates": ["canPierce"],
        "metrics": {
            "accuracy": _sig(-0.02, -0.07, 0.03, False),
            "balanced_accuracy": _sig(-0.01, -0.06, 0.04, False),
            "f1": _sig(-0.02, -0.06, 0.03, False),
            "mae": _sig(0.10, -0.05, 0.26, False),
            "spearman": _sig(-0.01, -0.09, 0.07, False),
        }}))
    return full, nop


def test_significance_reports_both_inclusion_conditions(sig_jsons, tmp_path):
    """The headline claim rests on the contrast between these two columns."""
    full, nop = sig_jsons
    tex = write_e7_significance_tex(full, nop, tmp_path / "s.tex").read_text()
    assert "canPierce" in tex
    assert "431" in tex and "300" in tex


def test_significance_renders_intervals_not_bare_diffs(sig_jsons, tmp_path):
    full, nop = sig_jsons
    tex = write_e7_significance_tex(full, nop, tmp_path / "s.tex").read_text()
    assert "-0.118" in tex and "-0.035" in tex


def test_significance_marks_which_intervals_span_zero(sig_jsons, tmp_path):
    """A reader must be able to see significance without recomputing it."""
    full, nop = sig_jsons
    tex = write_e7_significance_tex(full, nop, tmp_path / "s.tex").read_text()
    spearman = [ln for ln in tex.splitlines() if "Spearman" in ln][0]
    accuracy = [ln for ln in tex.splitlines() if "Accuracy" in ln][0]
    assert spearman != accuracy
    assert "$^{*}$" in accuracy      # significant
    assert "$^{*}$" not in spearman  # spans zero


def test_significance_refuses_mismatched_arms(sig_jsons, tmp_path):
    """Comparing arm E against arm G would be a category error."""
    full, nop = sig_jsons
    bad = json.loads(nop.read_text())
    bad["arm"] = "g"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="same arm"):
        write_e7_significance_tex(full, p, tmp_path / "s.tex")


# --------------------------------------------------------------------
# E9 arms table (v5 student, balanced-accuracy-led)
# --------------------------------------------------------------------

def _e9_metrics(bacc, mae=2.3, n=340):
    return {"n": n, "acc": bacc, "bacc": bacc, "prec": 0.70,
            "rec": 0.77, "f1": 0.73, "mae": mae}


@pytest.fixture
def e9_ladder(tmp_path):
    p = tmp_path / "e9_ladder.json"
    d = {}
    for arm, vlm, v4, v5 in (("G+", 0.734, 0.668, 0.722),
                             ("G", 0.720, None, 0.716),
                             ("E", 0.758, None, 0.701)):
        d[f"BALANCED 50/50|{arm}|VLM"] = _e9_metrics(vlm)
        d[f"BALANCED 50/50|{arm}|student_v5_mean"] = _e9_metrics(v5)
        d[f"BALANCED 50/50|{arm}|student_v5_sd"] = _e9_metrics(0.004, mae=0.07, n=0.0)
        if v4 is not None:
            d[f"BALANCED 50/50|{arm}|student_v4"] = _e9_metrics(v4)
        d[f"NATURAL|{arm}|VLM"] = _e9_metrics(vlm, n=593)
        d[f"NATURAL|{arm}|student_v5_mean"] = _e9_metrics(v5, n=593)
    p.write_text(json.dumps(d))
    return p


def test_e9_table_leads_with_balanced_accuracy(e9_ladder, tmp_path):
    """The eval set is 71.3% positive, so accuracy and F1 partly measure the
    class prior. Balanced accuracy must be the first metric column."""
    from affdet.experiment.result_tables import write_e9_arms_tex
    tex = write_e9_arms_tex(e9_ladder, tmp_path / "t.tex").read_text()
    header = [ln for ln in tex.splitlines() if "Judge" in ln][0]
    assert header.index("B.acc") < header.index("F1")


def test_e9_table_carries_the_seed_spread(e9_ladder, tmp_path):
    """A student mean without its spread invites a reader to over-read a
    0.012 gap that sits near the seed noise."""
    from affdet.experiment.result_tables import write_e9_arms_tex
    tex = write_e9_arms_tex(e9_ladder, tmp_path / "t.tex").read_text()
    assert "pm" in tex


def test_e9_table_keeps_v4_for_contrast(e9_ladder, tmp_path):
    """The v4 row is what makes the fork fix legible as a result."""
    from affdet.experiment.result_tables import write_e9_arms_tex
    tex = write_e9_arms_tex(e9_ladder, tmp_path / "t.tex").read_text()
    assert "0.668" in tex


def test_e9_table_orders_arms_gplus_g_e(e9_ladder, tmp_path):
    from affdet.experiment.result_tables import write_e9_arms_tex
    tex = write_e9_arms_tex(e9_ladder, tmp_path / "t.tex").read_text()
    body = tex[tex.index("midrule"):]
    assert body.index("G+") < body.index("{G}") < body.index("{E}")


def test_e9_table_states_the_balanced_construction(e9_ladder, tmp_path):
    """The 50/50 set is constructed by discarding positives; a caption that
    omits that invites reading it as a truer ground truth."""
    from affdet.experiment.result_tables import write_e9_arms_tex
    tex = write_e9_arms_tex(e9_ladder, tmp_path / "t.tex").read_text()
    low = tex.lower()
    assert "resampl" in low or "constructed" in low or "discard" in low


# --------------------------------------------------------------------
# The v5 ladder: corpus versions must not be blended
# --------------------------------------------------------------------

def test_ladder_rows_can_filter_to_one_corpus_version(tmp_path):
    """v4 and v5 runs coexist in the ledger and measure different corpora.

    Averaging a v4 seed with a v5 seed reports a model that was never
    trained -- the same class of error as blending -v4 with -v4fix.
    """
    p = tmp_path / "s.jsonl"
    rows = [_run("e1b-pair-s0-v4", 0.90), _run("e1b-pair-s1-v4", 0.92),
            _run("e1b-pair-s0-v5", 0.80), _run("e1b-pair-s1-v5", 0.82)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    v5 = ladder_rows(p, corpus="v5")["e1b-pair"]
    assert {r["config"]["name"] for r in v5} == {
        "e1b-pair-s0-v5", "e1b-pair-s1-v5"}

    v4 = ladder_rows(p, corpus="v4")["e1b-pair"]
    assert {r["config"]["name"] for r in v4} == {
        "e1b-pair-s0-v4", "e1b-pair-s1-v4"}


def test_ladder_table_reports_all_four_rungs(tmp_path):
    """The drop across rungs is the result; a missing rung hides it."""
    from affdet.experiment.result_tables import write_ladder_tex
    p = tmp_path / "s.jsonl"
    rows = []
    for rung, f1 in (("e1a-scene", 0.94), ("e1b-pair", 0.93),
                     ("e1c-actor_only", 0.88), ("e1c-strict", 0.87)):
        for s in range(3):
            rows.append(_run(f"{rung}-s{s}-v5", f1))
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tex = write_ladder_tex(p, tmp_path / "t.tex", corpus="v5").read_text()
    for label in ("unseen scenes", "unseen pairings", "unseen actor",
                  "unseen both"):
        assert label in tex
    assert "\\label{tab:ladder}" in tex


def test_ladder_table_refuses_to_mix_corpora(tmp_path):
    from affdet.experiment.result_tables import write_ladder_tex
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(_run("e1a-scene-s0-v4", 0.9)) + "\n")
    with pytest.raises(ValueError, match="no runs"):
        write_ladder_tex(p, tmp_path / "t.tex", corpus="v5")


# --------------------------------------------------------------------
# Score fidelity: how closely the student reproduces the teacher's score
# --------------------------------------------------------------------

def _fidelity_run(name, mae=0.10, corr=0.90):
    r = _run(name)
    r["metrics"]["calibrated"]["score_fidelity"] = {
        p: {"mae": mae, "corr": corr, "n": 1000} for p in PREDICATES[:5]
    }
    return r


def test_fidelity_table_reports_mae_and_correlation(tmp_path):
    from affdet.experiment.result_tables import write_fidelity_tex
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(_fidelity_run(f"e1a-scene-s{s}-v5"))
                           for s in range(3)) + "\n")
    tex = write_fidelity_tex(p, tmp_path / "t.tex", corpus="v5").read_text()
    assert "0.100" in tex and "0.900" in tex
    assert "\\label{tab:fidelity}" in tex


def test_fidelity_caption_says_it_is_not_correctness(tmp_path):
    """A table read on its own must not be mistaken for accuracy: matching a
    teacher exactly is worth nothing if the teacher is wrong."""
    from affdet.experiment.result_tables import write_fidelity_tex
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(_fidelity_run("e1a-scene-s0-v5")) + "\n")
    tex = write_fidelity_tex(p, tmp_path / "t.tex", corpus="v5").read_text()
    low = tex.lower()
    assert "not" in low and ("correct" in low or "right" in low)


def test_per_predicate_can_restrict_to_one_corpus(tmp_path):
    """A snapshot holding both versions must not average them together."""
    p = tmp_path / "s.jsonl"
    rows = [_run("e1a-scene-s0-v4", 0.90), _run("e1a-scene-s0-v5", 0.80)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tex = write_per_predicate_tex(p, tmp_path / "t.tex",
                                  rung="e1a-scene", corpus="v5").read_text()
    row = [ln for ln in tex.splitlines() if "canContain" in ln][0]
    assert "0.800" in row and "0.900" not in row
