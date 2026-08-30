"""Score a whole-photo VLM baseline response against gt2_*.json ground truth.

The VLM invents its own object ids (it has no access to our gt2 instance
numbering), so objects are matched to gt by CATEGORY, in list order, for
each category — e.g. gt's ["cup_1", "cup_2"] matches the VLM's first two
"cup"-category detections positionally. This is a known simplification for
duplicate-category scenes: it cannot recover which specific cup is which if
the VLM's own ordering doesn't line up with gt's (see gt2's
DSC01940 cup_1/cup_2 case, the one real instance of this in the 19 photos).
"""
from __future__ import annotations

from collections import defaultdict


def _category_of(gt_object_id: str) -> str:
    """"knife_1" -> "knife". Matches affdet.label.vlm._split_instance."""
    tail = gt_object_id.rsplit("_", 1)
    return tail[0] if len(tail) == 2 and tail[1].isdigit() else gt_object_id


def _match_gt_to_vlm_ids(gt_objects: list[str], vlm_objects: list[dict]) -> dict[str, str | None]:
    """Map each gt object id to a VLM object id (or None if unmatched),
    matching by category in list order."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for obj in vlm_objects:
        by_category[obj["category"]].append(obj["id"])

    mapping: dict[str, str | None] = {}
    consumed: dict[str, int] = defaultdict(int)
    for gt_id in gt_objects:
        cat = _category_of(gt_id)
        pool = by_category.get(cat, [])
        idx = consumed[cat]
        mapping[gt_id] = pool[idx] if idx < len(pool) else None
        consumed[cat] += 1
    return mapping


def score_baseline_response(response: dict, gt: dict) -> dict:
    """response: parsed VLM output ({"objects": [...], "affordances": [...]}).
    gt: one gt2_*.json ({"objects": [...], "positive_pairs": [...]})."""
    gt_objects: list[str] = gt["objects"]
    gt_to_vlm = _match_gt_to_vlm_ids(gt_objects, response["objects"])
    vlm_to_gt = {v: k for k, v in gt_to_vlm.items() if v is not None}

    matched_object_count = sum(1 for v in gt_to_vlm.values() if v is not None)
    missed_objects = len(gt_objects) - matched_object_count

    positive_keys = {
        (p["actor"], p["recipient"], p["predicate"]) for p in gt["positive_pairs"]
    }

    # Every ordered pair x predicate over gt objects is a label to score.
    # A pair is only SCOREABLE if both gt objects were matched to some VLM
    # detection; unmatched-object pairs can never be predicted true, so a
    # positive one is an automatic FN (the VLM had no chance), and a
    # negative one is skipped entirely (nothing to get wrong).
    predicted = {}
    for entry in response["affordances"]:
        actor_gt = vlm_to_gt.get(entry["actor"])
        recip_gt = vlm_to_gt.get(entry["recipient"])
        if actor_gt is None or recip_gt is None:
            continue
        predicted[(actor_gt, recip_gt, entry["predicate"])] = entry["verdict"]

    tp = fp = fn = tn = 0
    for actor in gt_objects:
        for recip in gt_objects:
            if actor == recip:
                continue
            for pred in ("canCut", "canPierce", "canPeel", "canContain",
                         "canSupport", "canPourInto"):
                key = (actor, recip, pred)
                is_positive = key in positive_keys
                actor_matched = gt_to_vlm.get(actor) is not None
                recip_matched = gt_to_vlm.get(recip) is not None
                if not (actor_matched and recip_matched):
                    if is_positive:
                        fn += 1
                    continue
                verdict = predicted.get(key, 0)
                if is_positive and verdict == 1:
                    tp += 1
                elif is_positive and verdict == 0:
                    fn += 1
                elif not is_positive and verdict == 1:
                    fp += 1
                else:
                    tn += 1

    return {
        "matched_object_count": matched_object_count,
        "missed_objects": missed_objects,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
