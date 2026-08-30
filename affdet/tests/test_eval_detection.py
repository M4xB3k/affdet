"""Detection scoring (E5 component eval).

These tests pin the semantics that `affdet.detect.yolo_boxes.match_boxes`
deliberately does NOT provide: false positives (it drops unmatched
predictions, since the crop-source ablation has no gold obj_id to attach
them to) and class-awareness (it matches purely geometrically and keeps the
gold class). Detection P/R/F1 needs both, so this is a separate scorer
rather than a wrapper.
"""
import pytest

from affdet.eval.detection import score_detections, aggregate_detection_scores
from affdet.schemas import SceneObject


def _gold(obj_id, obj_class, bbox):
    return SceneObject(obj_id=obj_id, obj_class=obj_class, bbox=bbox)


def _pred(cls, bbox, conf=0.9):
    return {"bbox": bbox, "cls": cls, "conf": conf}


def test_perfect_detection_scores_one():
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 10])]

    r = score_detections(gold, preds)

    assert r["tp"] == 1
    assert r["fp"] == 0
    assert r["fn"] == 0
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0


def test_spurious_prediction_counts_as_false_positive():
    """The gap match_boxes structurally cannot express: a detector
    hallucinating an object that is not there must cost precision."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 10]),
             _pred("bowl", [50, 50, 60, 60])]

    r = score_detections(gold, preds)

    assert r["tp"] == 1
    assert r["fp"] == 1
    assert r["fn"] == 0
    assert r["precision"] == 0.5
    assert r["recall"] == 1.0


def test_missed_object_counts_as_false_negative():
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10]),
            _gold("bowl_0", "bowl", [50, 50, 60, 60])]
    preds = [_pred("knife", [0, 0, 10, 10])]

    r = score_detections(gold, preds)

    assert r["tp"] == 1
    assert r["fp"] == 0
    assert r["fn"] == 1
    assert r["precision"] == 1.0
    assert r["recall"] == 0.5


def test_iou_threshold_is_inclusive():
    """IoU >= 0.5 counts as a hit. match_boxes uses strict `>`, which would
    reject a box sitting exactly on the threshold; the thesis reports
    'IoU >= 0.5', so the boundary case must match the claim."""
    # Intersection 5x10=50, union 10x10 + 10x10 - 50 = 150 -> IoU = 1/3.
    # Pick boxes giving exactly 0.5: A=[0,0,10,10], B=[0,0,10,20] ->
    # inter=100, union=100+200-100=200 -> 0.5 exactly.
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 20])]

    r = score_detections(gold, preds, iou_thresh=0.5)

    assert r["tp"] == 1, "IoU exactly 0.5 must count as a match"


def test_below_threshold_box_is_both_fp_and_fn():
    """A badly-localised box is not a hit: the gold object is missed AND the
    prediction is spurious. Counting it as neither would silently flatter
    the detector."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [9, 9, 20, 20])]

    r = score_detections(gold, preds, iou_thresh=0.5)

    assert r["tp"] == 0
    assert r["fp"] == 1
    assert r["fn"] == 1


def test_right_box_wrong_class_is_not_a_true_positive():
    """Detection scoring is class-aware: a perfectly placed box with the
    wrong label has not detected the object."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("bowl", [0, 0, 10, 10])]

    r = score_detections(gold, preds, iou_thresh=0.5)

    assert r["tp"] == 0
    assert r["fn"] == 1, "the knife was not detected"
    assert r["fp"] == 1, "a bowl was reported that does not exist"


def test_class_agnostic_mode_ignores_labels():
    """Box quality alone -- the number that matters for the crop pipeline,
    which re-classifies with the DINOv2 probe and ignores detector labels."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("bowl", [0, 0, 10, 10])]

    r = score_detections(gold, preds, class_agnostic=True)

    assert r["tp"] == 1
    assert r["fp"] == 0
    assert r["fn"] == 0


def test_each_prediction_is_consumed_at_most_once():
    """Two gold objects cannot both claim the same box; the duplicate is a
    false positive, not a second true positive."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10]),
            _gold("knife_1", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 10])]

    r = score_detections(gold, preds)

    assert r["tp"] == 1
    assert r["fn"] == 1


def test_duplicate_predictions_cost_precision():
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 10]),
             _pred("knife", [0, 0, 10, 10])]

    r = score_detections(gold, preds)

    assert r["tp"] == 1
    assert r["fp"] == 1, "the second box detects nothing new"


def test_highest_iou_match_wins():
    """Greedy matching must pair each gold with its BEST box, not the first
    one that clears threshold -- otherwise a sloppy box can consume the gold
    and strand the tight one as a false positive."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 18]),   # IoU ~0.55
             _pred("knife", [0, 0, 10, 10])]   # IoU 1.0

    r = score_detections(gold, preds)

    assert r["tp"] == 1
    assert r["fp"] == 1
    assert r["matches"][0]["iou"] == pytest.approx(1.0)


def test_per_class_breakdown():
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10]),
            _gold("bowl_0", "bowl", [50, 50, 60, 60])]
    preds = [_pred("knife", [0, 0, 10, 10])]

    r = score_detections(gold, preds)

    assert r["per_class"]["knife"]["tp"] == 1
    assert r["per_class"]["knife"]["recall"] == 1.0
    assert r["per_class"]["bowl"]["fn"] == 1
    assert r["per_class"]["bowl"]["recall"] == 0.0


def test_false_positive_attributed_to_predicted_class():
    """An FP belongs to the class the detector CLAIMED -- that is the class
    whose precision it damaged."""
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]
    preds = [_pred("knife", [0, 0, 10, 10]),
             _pred("durian", [50, 50, 60, 60])]

    r = score_detections(gold, preds)

    assert r["per_class"]["durian"]["fp"] == 1
    assert r["per_class"]["durian"]["precision"] == 0.0


def test_empty_prediction_set_is_total_recall_failure():
    gold = [_gold("knife_0", "knife", [0, 0, 10, 10])]

    r = score_detections(gold, [])

    assert r["tp"] == 0
    assert r["fn"] == 1
    assert r["recall"] == 0.0
    assert r["precision"] == 0.0, "no predictions -> precision defined as 0"


def test_empty_gold_and_empty_preds_is_not_a_crash():
    r = score_detections([], [])

    assert r["tp"] == r["fp"] == r["fn"] == 0
    assert r["f1"] == 0.0


def test_aggregate_sums_counts_across_scenes():
    """Aggregation must pool TP/FP/FN and recompute P/R/F1 (micro), not
    average per-scene F1s -- a scene with one object would otherwise weigh
    as much as a scene with ten."""
    per_scene = [
        score_detections([_gold("knife_0", "knife", [0, 0, 10, 10])],
                         [_pred("knife", [0, 0, 10, 10])]),
        score_detections([_gold("bowl_0", "bowl", [0, 0, 10, 10])],
                         []),
    ]

    agg = aggregate_detection_scores(per_scene)

    assert agg["tp"] == 1
    assert agg["fn"] == 1
    assert agg["recall"] == 0.5
    assert agg["per_class"]["knife"]["tp"] == 1
    assert agg["per_class"]["bowl"]["fn"] == 1


def test_aggregate_macro_f1_averages_over_classes():
    """Macro = mean of per-class F1, so a rare class counts as much as a
    common one; reported alongside micro because they answer different
    questions."""
    per_scene = [
        score_detections(
            [_gold("knife_0", "knife", [0, 0, 10, 10]),
             _gold("bowl_0", "bowl", [50, 50, 60, 60])],
            [_pred("knife", [0, 0, 10, 10])],
        ),
    ]

    agg = aggregate_detection_scores(per_scene)

    # knife F1 = 1.0, bowl F1 = 0.0 -> macro 0.5, micro = 2*1/(2*1+0+1) = 0.667
    assert agg["macro_f1"] == pytest.approx(0.5)
    assert agg["f1"] == pytest.approx(2 / 3)


def test_subgroup_splits_scores_by_gold_object_membership():
    """The instance-disjoint arm reserves MESHES, but pulls whole SCENES into
    test -- and a scene holds 3-4 objects, most of them ordinary training
    meshes. Measured on the real split, only 28.6% of scored objects are
    novel meshes, so the pooled F1 is dominated by seen ones and does not
    mean what the arm's name claims. Scoring the two groups separately is
    what makes the arm answer its own question."""
    gold = [_gold("bowl_001", "bowl", [0, 0, 10, 10]),      # seen mesh
            _gold("bowl_999", "bowl", [20, 20, 30, 30])]    # novel mesh
    preds = [_pred("bowl", [0, 0, 10, 10]),                 # correct
             _pred("plate", [20, 20, 30, 30])]              # wrong name

    r = score_detections(gold, preds, subgroups={"novel": {"bowl_999"}})

    assert r["subgroups"]["novel"]["tp"] == 0
    assert r["subgroups"]["novel"]["fn"] == 1
    assert r["subgroups"]["rest"]["tp"] == 1
    assert r["subgroups"]["rest"]["f1"] == 1.0


def test_subgroup_absent_when_not_requested():
    """Opt-in: the D-a arm has no novel-mesh question to ask."""
    gold = [_gold("bowl_001", "bowl", [0, 0, 10, 10])]
    r = score_detections(gold, [_pred("bowl", [0, 0, 10, 10])])
    assert "subgroups" not in r


def test_aggregate_pools_subgroups_across_scenes():
    a = score_detections(
        [_gold("m_new", "bowl", [0, 0, 10, 10])],
        [_pred("bowl", [0, 0, 10, 10])],
        subgroups={"novel": {"m_new"}})
    b = score_detections(
        [_gold("m_new", "bowl", [0, 0, 10, 10])],
        [],
        subgroups={"novel": {"m_new"}})

    agg = aggregate_detection_scores([a, b])

    assert agg["subgroups"]["novel"]["tp"] == 1
    assert agg["subgroups"]["novel"]["fn"] == 1
    assert agg["subgroups"]["novel"]["recall"] == 0.5
