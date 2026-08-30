"""Exporting E5 ledger rows to the thesis table + committed results files."""
import json

import pytest

from affdet.experiment.component_tables import (
    load_component_runs,
    write_component_json,
    write_component_markdown,
    write_component_tex,
)


def _row(name, detector, zero_shot, f1, precision=0.9, recall=0.9,
         macro_f1=None, per_class=None, agnostic_f1=None):
    return {
        "kind": "component",
        "run_id": f"{name}-abc123-def4567",
        "config": {"name": name, "detector": detector,
                   "zero_shot": zero_shot, "iou_thresh": 0.5, "seed": 0},
        "n_scenes_scored": 100,
        "metrics": {
            "detection": {
                "precision": precision, "recall": recall, "f1": f1,
                "macro_f1": macro_f1 if macro_f1 is not None else f1,
                "tp": 90, "fp": 10, "fn": 10,
                "n_gold": 100, "n_pred": 100, "n_scenes": 100,
                "per_class": per_class or {
                    "apple": {"tp": 45, "fp": 5, "fn": 5, "precision": 0.9,
                              "recall": 0.9, "f1": 0.9},
                },
            },
            "detection_class_agnostic": {
                "precision": 0.95, "recall": 0.95,
                "f1": agnostic_f1 if agnostic_f1 is not None else 0.95,
                "macro_f1": 0.95, "tp": 95, "fp": 5, "fn": 5,
                "n_gold": 100, "n_pred": 100, "n_scenes": 100,
                "per_class": {},
            },
        },
    }


@pytest.fixture
def ledger(tmp_path):
    p = tmp_path / "runs.jsonl"
    rows = [
        {"kind": "affordance", "run_id": "e1a-x", "metrics": {}},
        _row("e5-yolo11s-ft", "yolo11s", False, 0.94),
        _row("e5-yoloworld-ft", "yolov8s-worldv2", False, 0.96),
        _row("e5-yoloworld-zs", "yolov8s-worldv2", True, 0.11),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_affordance_rows_are_not_loaded_as_component_rows(ledger):
    """One ledger holds both experiment kinds; mixing them would put an
    affordance run in the detector table."""
    runs = load_component_runs(ledger)
    assert len(runs) == 3
    assert all(r["kind"] == "component" for r in runs)


def test_latest_run_per_arm_wins(tmp_path):
    """Re-running an arm supersedes the old row rather than appearing twice
    -- the stale-rows-merging-into-one-family failure the E1 ladder hit."""
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        _row("e5-yoloworld-ft", "yolov8s-worldv2", False, 0.80),
        _row("e5-yoloworld-ft", "yolov8s-worldv2", False, 0.96),
    ]) + "\n")

    runs = load_component_runs(p)
    assert len(runs) == 1
    assert runs[0]["metrics"]["detection"]["f1"] == 0.96


def test_tex_is_generated_from_the_ledger(ledger, tmp_path):
    out = write_component_tex(load_component_runs(ledger),
                              tmp_path / "component_quality.tex")
    text = out.read_text()

    assert "DO NOT EDIT BY HAND" in text, "generated files carry the warning"
    assert "\\begin{table}" in text and "\\toprule" in text
    assert "\\label{tab:component-quality}" in text
    assert "0.96" in text


def test_tex_marks_zero_shot_arms_distinctly(ledger, tmp_path):
    """A zero-shot row sitting unmarked beside fine-tuned rows reads as a
    like-for-like comparison, which it is not."""
    out = write_component_tex(load_component_runs(ledger),
                              tmp_path / "t.tex")
    text = out.read_text()

    assert "zero-shot" in text.lower()
    assert "fine-tuned" in text.lower()


def test_tex_escapes_underscores_in_names(tmp_path):
    """Raw underscores are math-mode subscripts in LaTeX and break the
    build."""
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(
        _row("e5-x", "yolov8s-worldv2", False, 0.9, per_class={
            "red_wine_glass": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0,
                               "recall": 1.0, "f1": 1.0}})) + "\n")

    out = write_component_tex(load_component_runs(p), tmp_path / "t.tex",
                              per_class=True)

    assert "red\\_wine\\_glass" in out.read_text()


def test_markdown_records_numbers_and_reliability(ledger, tmp_path):
    """Results are committed as md/json first; the md carries the honest
    reading, following the precedent of the other data/eval docs."""
    out = write_component_markdown(load_component_runs(ledger),
                                   tmp_path / "E5.md")
    text = out.read_text()

    assert "Reliability" in text
    assert "e5-yoloworld-ft" in text
    assert "0.96" in text


def test_json_companion_is_machine_readable(ledger, tmp_path):
    out = write_component_json(load_component_runs(ledger), tmp_path / "e5.json")
    data = json.loads(out.read_text())

    assert len(data["arms"]) == 3
    names = [a["name"] for a in data["arms"]]
    assert "e5-yoloworld-ft" in names


def test_arms_are_ordered_finetuned_before_zero_shot(ledger, tmp_path):
    """The table should read as an argument: what we deploy, then what the
    alternatives score."""
    out = write_component_json(load_component_runs(ledger), tmp_path / "e5.json")
    arms = json.loads(out.read_text())["arms"]

    zs_index = [i for i, a in enumerate(arms) if a["zero_shot"]]
    ft_index = [i for i, a in enumerate(arms) if not a["zero_shot"]]
    assert max(ft_index) < min(zs_index)


def test_tex_caption_carries_the_threshold_caveat(tmp_path):
    """A LaTeX table is read detached from the markdown, so a caveat that
    changes how a column may be read has to travel with the table."""
    p = tmp_path / "runs.jsonl"
    a = _row("e5-Da-yoloworld-zs", "yolov8s-worldv2", True, 0.33)
    a["config"]["conf"] = 0.05
    b = _row("e5-Da-groundingdino-zs", "grounding-dino", True, 0.38)
    b["config"]["conf"] = 0.25
    p.write_text("\n".join(json.dumps(r) for r in (a, b)) + "\n")

    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    assert "0.25" in tex and "Confidence thresholds differ" in tex


def test_tex_caption_omits_the_caveat_when_thresholds_agree(tmp_path):
    p = tmp_path / "runs.jsonl"
    rows = []
    for n in ("e5-Da-yolo11s-ft", "e5-Da-yoloworld-ft"):
        r = _row(n, n, False, 0.9)
        r["config"]["conf"] = 0.05
        rows.append(r)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    assert "Confidence thresholds differ" not in tex


def test_no_doubled_rule_when_every_arm_is_zero_shot(tmp_path):
    """The fine-tuned/zero-shot separator must not fire on the first row:
    stacked against the header rule it renders as one thick line."""
    p = tmp_path / "runs.jsonl"
    rows = [_row("e5-Da-yoloworld-zs", "yolov8s-worldv2", True, 0.33),
            _row("e5-Da-groundingdino-zs", "grounding-dino", True, 0.38)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    assert "\\midrule\n    \\midrule" not in tex


def test_rule_divides_the_ladder_rungs(tmp_path):
    """The rule separates ladder RUNGS, not training modes. With two rungs
    the rung is the outer grouping, and a rule inside one rung would imply
    the rows above and below score different splits when they do not."""
    p = tmp_path / "runs.jsonl"
    rows = [_row("e5-Da-yolo11s-ft", "yolo11s", False, 0.94),
            _row("e5-Da-yoloworld-zs", "yolov8s-worldv2", True, 0.33),
            _row("e5-Db-yolo11s-ft", "yolo11s", False, 0.89),
            _row("e5-Db-yoloworld-zs", "yolov8s-worldv2", True, 0.31)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    body = tex.split("Split & Detector & Training")[1]
    assert body.count("\\midrule") == 2  # header rule + one rung boundary


def test_no_rule_inside_a_single_rung(tmp_path):
    p = tmp_path / "runs.jsonl"
    rows = [_row("e5-Da-yolo11s-ft", "yolo11s", False, 0.94),
            _row("e5-Da-yoloworld-zs", "yolov8s-worldv2", True, 0.33)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    body = tex.split("Split & Detector & Training")[1]
    assert body.count("\\midrule") == 1  # header rule only


def test_arms_can_be_restricted_to_a_named_set(tmp_path):
    """The ledger is append-only, so superseded arms stay in it forever. The
    "newest wins" rule keys on arm NAME, so it cannot retire a row whose name
    changed: the pre-ladder rows (e5-yolo11s-ft) were scored with detectors
    trained on the OLD 1,000-scene corpus and would otherwise sit in the
    thesis table next to the current e5-Da-/e5-Db- rows forever."""
    p = tmp_path / "runs.jsonl"
    rows = [_row("e5-yolo11s-ft", "yolo11s", False, 0.91),        # stale
            _row("e5-Da-yolo11s-ft", "yolo11s", False, 0.94),
            _row("e5-Db-yolo11s-ft", "yolo11s", False, 0.89)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    kept = load_component_runs(p, include=("e5-Da-", "e5-Db-"))

    assert {r["config"]["name"] for r in kept} == {
        "e5-Da-yolo11s-ft", "e5-Db-yolo11s-ft"}


def test_unrestricted_load_still_returns_every_arm(tmp_path):
    """The filter must be opt-in: the ledger audit path wants everything."""
    p = tmp_path / "runs.jsonl"
    rows = [_row("e5-yolo11s-ft", "yolo11s", False, 0.91),
            _row("e5-Da-yolo11s-ft", "yolo11s", False, 0.94)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert len(load_component_runs(p)) == 2


def test_markdown_warns_when_arms_used_different_conf_thresholds(tmp_path):
    """Grounding DINO's threshold also drives its text_threshold, so it is
    run at 0.25 while the YOLO arms run at 0.05. Precision is sensitive to
    that, so the doc must say the thresholds differ rather than let the
    "like-for-like" claim imply the columns are directly comparable."""
    p = tmp_path / "runs.jsonl"
    a = _row("e5-yoloworld-zs", "yolov8s-worldv2", True, 0.33)
    a["config"]["conf"] = 0.05
    b = _row("e5-groundingdino-zs", "grounding-dino", True, 0.38)
    b["config"]["conf"] = 0.25
    p.write_text("\n".join(json.dumps(r) for r in (a, b)) + "\n")

    text = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md").read_text()

    assert "0.05" in text and "0.25" in text
    assert "confidence threshold" in text.lower()


def test_markdown_omits_threshold_warning_when_arms_agree(tmp_path):
    """The warning must be driven by the ledger, not printed unconditionally,
    or it would cry wolf on a table that is in fact like-for-like."""
    p = tmp_path / "runs.jsonl"
    rows = []
    for name in ("e5-yolo11s-ft", "e5-yoloworld-ft"):
        r = _row(name, name, False, 0.9)
        r["config"]["conf"] = 0.05
        rows.append(r)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    text = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md").read_text()

    assert "did not all run at the same confidence threshold" not in text


def test_empty_ledger_does_not_produce_a_fake_table(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text("")

    with pytest.raises(ValueError, match="no component runs"):
        write_component_tex(load_component_runs(p), tmp_path / "t.tex")


def _row_with_subgroups(name, detector, zero_shot, f1,
                        novel_recall, seen_recall):
    r = _row(name, detector, zero_shot, f1)
    for key, aware in (("detection", True), ("detection_class_agnostic", False)):
        r["metrics"][key]["subgroups"] = {
            "novel": {"tp": 2872, "fp": 0, "fn": 1415, "n_gold": 4287,
                      "precision": 1.0, "recall": novel_recall,
                      "f1": novel_recall},
            "rest": {"tp": 10595, "fp": 0, "fn": 221, "n_gold": 10816,
                     "precision": 1.0, "recall": seen_recall,
                     "f1": seen_recall},
        }
    return r


def test_markdown_reports_the_novel_mesh_generalisation_gap(tmp_path):
    """The instance-disjoint arm's whole point is the novel-vs-seen split.
    Reporting only the pooled number hides a 31-point collapse in naming
    behind a 0.89 that looks fine."""
    p = tmp_path / "runs.jsonl"
    r = _row_with_subgroups("e5-Db-yoloworld-ft", "yolov8s-worldv2", False,
                            0.8927, 0.6699, 0.9796)
    p.write_text(json.dumps(r) + "\n")

    text = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md").read_text()

    assert "0.670" in text and "0.980" in text
    assert "novel" in text.lower()


def test_markdown_has_no_generalisation_section_without_subgroups(tmp_path):
    """The scene-disjoint arm has no novel-mesh question to answer."""
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(_row("e5-Da-yolo11s-ft", "yolo11s", False, 0.94))
                 + "\n")

    text = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md").read_text()

    assert "## Generalisation to unseen mesh instances" not in text


def test_tex_includes_the_generalisation_table_when_subgroups_exist(tmp_path):
    """The novel/seen split is the instance-disjoint arm's actual result; a
    thesis table that omits it reports the pooled number that hides it."""
    p = tmp_path / "runs.jsonl"
    r = _row_with_subgroups("e5-Db-yoloworld-ft", "yolov8s-worldv2", False,
                            0.8927, 0.6699, 0.9796)
    p.write_text(json.dumps(r) + "\n")

    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    assert "0.670" in tex and "0.980" in tex
    assert "novel" in tex.lower()
    assert "\\label{tab:component-generalisation}" in tex


def test_table_distinguishes_the_generalisation_arms(tmp_path):
    """Da and Db rows carry the same detector labels but different splits.
    Without an arm column the table shows 'YOLO-World-S fine-tuned' twice
    with different numbers and no stated reason -- which reads as an
    inconsistency rather than a ladder."""
    p = tmp_path / "runs.jsonl"
    a = _row("e5-Da-yoloworld-ft", "yolov8s-worldv2", False, 0.989)
    b = _row("e5-Db-yoloworld-ft", "yolov8s-worldv2", False, 0.893)
    p.write_text("\n".join(json.dumps(r) for r in (a, b)) + "\n")

    md = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md").read_text()
    tex = write_component_tex(load_component_runs(p), tmp_path / "t.tex").read_text()

    for text in (md, tex):
        assert "Scene-disjoint" in text or "scene-disjoint" in text
        assert "Instance-disjoint" in text or "instance-disjoint" in text


def test_header_does_not_claim_one_scene_count_for_differing_arms(tmp_path):
    """Da scores 1275 scenes and Db 3787; a single '(N scenes)' in the
    preamble silently attributes one arm's count to both."""
    p = tmp_path / "runs.jsonl"
    a = _row("e5-Da-yoloworld-ft", "yolov8s-worldv2", False, 0.989)
    a["n_scenes_scored"] = 1275
    b = _row("e5-Db-yoloworld-ft", "yolov8s-worldv2", False, 0.893)
    b["n_scenes_scored"] = 3787
    p.write_text("\n".join(json.dumps(r) for r in (a, b)) + "\n")

    md = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md").read_text()

    head = md.split("## Detection")[0]
    assert "1275 scenes)" not in head and "3787 scenes)" not in head


def _cls_row(name, probe, head, novel=None, seen=None, seed=0):
    def blk(p, h, n=1000):
        return {"probe": {"accuracy": p, "n": n},
                "classhead": {"accuracy": h, "n": n},
                "delta_accuracy": p - h, "n": n}
    r = {"kind": "classification",
         "run_id": f"{name}-cls-abc123-def4567",
         "config": {"name": name, "detector": "yolo11s", "zero_shot": False,
                    "iou_thresh": 0.5, "seed": seed,
                    "split_from_dataset": True},
         "n_eval_scenes": 3787, "n_train_scenes": 8966,
         "n_detector_misses": 268,
         "seed_affects_result": False,
         "metrics": {"classification": blk(probe, head)}}
    if novel is not None:
        r["metrics"]["classification"]["subgroups"] = {
            "novel": blk(*novel, n=4219), "rest": blk(*seen, n=10698)}
    return r


def test_markdown_reports_the_probe_vs_classhead_comparison(tmp_path):
    """The measurement that justifies re-classifying every crop. It was
    reaching the ledger and stopping there: the exporter only read detection
    rows, so the committed results and the thesis table did not contain it."""
    p = tmp_path / "runs.jsonl"
    rows = [_row("e5-Db-yolo11s-ft", "yolo11s", False, 0.916),
            _cls_row("e5-Db-yolo11s-ft", 0.9706, 0.9257,
                     novel=(0.9348, 0.7516), seen=(0.9847, 0.9943))]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    from affdet.experiment.component_tables import load_classification_runs
    text = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md",
        classification=load_classification_runs(p)).read_text()

    assert "0.935" in text and "0.752" in text     # the novel-mesh split
    assert "class head" in text.lower()


def test_markdown_omits_classification_when_absent(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(_row("e5-Db-yolo11s-ft", "yolo11s", False, 0.9))
                 + "\n")

    from affdet.experiment.component_tables import load_classification_runs
    text = write_component_markdown(
        load_component_runs(p), tmp_path / "E5.md",
        classification=load_classification_runs(p)).read_text()

    assert "class head" not in text.lower()


# --------------------------------------------------------------------
# Classification table (probe vs the detector's own class head)
# --------------------------------------------------------------------

def _cls_tex_row(name, n, probe, classhead, novel=None, seen=None):
    """One classification ledger row, optionally with the novel/seen split."""
    m = {"n": n,
         "probe": {"accuracy": probe},
         "classhead": {"accuracy": classhead},
         "delta_accuracy": probe - classhead}
    if novel is not None:
        m["subgroups"] = {
            "novel": {"n": novel[0], "probe": {"accuracy": novel[1]},
                      "classhead": {"accuracy": novel[2]}},
            "rest": {"n": seen[0], "probe": {"accuracy": seen[1]},
                     "classhead": {"accuracy": seen[2]}},
        }
    return {"kind": "classification", "run_id": f"{name}-abc123",
            "config": {"name": name, "detector": "yolo11s", "zero_shot": False},
            "metrics": {"classification": m}}


@pytest.fixture
def cls_runs():
    return [
        _cls_tex_row("e5-Da-yolo11s-ft", 4744, 0.988, 0.994),
        _cls_tex_row("e5-Db-yolo11s-ft", 14917, 0.971, 0.926,
                 novel=(4219, 0.935, 0.752), seen=(10698, 0.985, 0.994)),
    ]


def test_classification_tex_reports_both_classifiers(cls_runs, tmp_path):
    """An arm with no novel/seen split reports its pooled accuracies.

    Arms that DO carry the split report it instead of the pooled figure --
    see the next test -- since pooling is what hides the effect.
    """
    from affdet.experiment.component_tables import write_classification_tex
    tex = write_classification_tex(cls_runs, tmp_path / "c.tex").read_text()
    pooled = [ln for ln in tex.splitlines() if "e5-Da" in ln][0]
    assert "0.988" in pooled and "0.994" in pooled


def test_classification_tex_splits_novel_from_seen(cls_runs, tmp_path):
    """The whole argument lives in this split.

    Pooled, the probe looks marginally better. Split, the class head is ahead
    on seen meshes and collapses on novel ones -- which is what justifies
    discarding its labels.
    """
    from affdet.experiment.component_tables import write_classification_tex
    tex = write_classification_tex(cls_runs, tmp_path / "c.tex").read_text()
    assert "0.935" in tex and "0.752" in tex   # novel: probe holds, head falls
    assert "0.985" in tex and "0.994" in tex   # seen: head marginally ahead


def test_classification_tex_is_an_inputtable_float(cls_runs, tmp_path):
    from affdet.experiment.component_tables import write_classification_tex
    tex = write_classification_tex(cls_runs, tmp_path / "c.tex").read_text()
    assert "\\begin{table}" in tex
    assert "\\label{tab:classification}" in tex
    assert "GENERATED" in tex


def test_classification_tex_refuses_empty(tmp_path):
    from affdet.experiment.component_tables import write_classification_tex
    with pytest.raises(ValueError, match="no classification runs"):
        write_classification_tex([], tmp_path / "c.tex")
