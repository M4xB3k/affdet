from affdet.eval.baseline_vlm_score import score_baseline_response


def _gt(positive_pairs, objects):
    """gt2_*.json shape: object ids ARE "<category>_<n>" (no separate
    category field) -- e.g. "knife_1", "cup_2"."""
    return {
        "objects": objects,
        "positive_pairs": positive_pairs,
    }


def test_exact_match_scores_perfect():
    gt = _gt(
        [{"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut"}],
        ["knife_1", "apple_1"],
    )
    response = {
        "objects": [{"id": "knife_1", "category": "knife"},
                    {"id": "apple_1", "category": "apple"}],
        "affordances": [
            {"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut", "verdict": 1},
            {"actor": "knife_1", "recipient": "apple_1", "predicate": "canPierce", "verdict": 0},
            {"actor": "apple_1", "recipient": "knife_1", "predicate": "canCut", "verdict": 0},
        ],
    }
    result = score_baseline_response(response, gt)
    assert result["matched_object_count"] == 2
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_missed_object_counts_as_false_negatives_for_its_positive_pairs():
    # VLM only found the knife, missed the apple entirely -> the ground-truth
    # positive pair (knife, apple, canCut) can never be matched -> FN.
    gt = _gt(
        [{"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut"}],
        ["knife_1", "apple_1"],
    )
    response = {
        "objects": [{"id": "knife_1", "category": "knife"}],
        "affordances": [],
    }
    result = score_baseline_response(response, gt)
    assert result["matched_object_count"] == 1
    assert result["missed_objects"] == 1
    assert result["fn"] == 1
    assert result["tp"] == 0


def test_false_positive_when_vlm_says_true_but_gt_says_false():
    gt = _gt([], ["knife_1", "apple_1"])
    response = {
        "objects": [{"id": "knife_1", "category": "knife"},
                    {"id": "apple_1", "category": "apple"}],
        "affordances": [
            {"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut", "verdict": 1},
        ],
    }
    result = score_baseline_response(response, gt)
    assert result["fp"] == 1
    assert result["tp"] == 0


def test_category_matching_uses_vlm_category_field_not_literal_id():
    # The VLM invents its own instance ids (e.g. "obj_2"), so matching against
    # gt2's "<category>_<n>" ids must go through the VLM's own "category"
    # field, not literal id string equality.
    gt = _gt(
        [{"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut"}],
        ["knife_1", "apple_1"],
    )
    response = {
        "objects": [{"id": "obj_2", "category": "apple"},
                    {"id": "obj_1", "category": "knife"}],
        "affordances": [
            {"actor": "obj_1", "recipient": "obj_2", "predicate": "canCut", "verdict": 1},
        ],
    }
    result = score_baseline_response(response, gt)
    assert result["tp"] == 1


def test_duplicate_category_instances_matched_positionally():
    # Two cups in gt (cup_1, cup_2) with DIFFERENT verdicts against the same
    # recipient -- matched to the VLM's two "cup" detections in list order.
    # This is a known simplification (real disambiguation would need a
    # distinguishing visual description), flagged as such in the module.
    gt = _gt(
        [{"actor": "cup_1", "recipient": "kiwifruit_1", "predicate": "canContain"}],
        ["cup_1", "cup_2", "kiwifruit_1"],
    )
    response = {
        "objects": [{"id": "cupA", "category": "cup"},
                    {"id": "cupB", "category": "cup"},
                    {"id": "kiwi", "category": "kiwifruit"}],
        "affordances": [
            {"actor": "cupA", "recipient": "kiwi", "predicate": "canContain", "verdict": 1},
            {"actor": "cupB", "recipient": "kiwi", "predicate": "canContain", "verdict": 0},
        ],
    }
    result = score_baseline_response(response, gt)
    assert result["tp"] == 1
    assert result["fp"] == 0
