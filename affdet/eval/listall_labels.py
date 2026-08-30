"""Turn volunteered list-all output into a training label corpus.

The existing training corpus (`labels_omni_v4.jsonl`) has one row per
(actor, recipient, predicate) and carries no scene id -- labels are
instance-pair level. list-all output is scene-scoped, so the conversion has two
aggregation steps and their ORDER matters:

  1. within a scene, reconcile the passes (median of the passes that mentioned
     the pair);
  2. across scenes, reconcile the per-scene scores (median again).

Doing it in one flat pass would weight a scene by how often the VLM happened to
mention its pairs -- a pair named in 3 of 3 passes in one scene and 1 of 3 in
another would let the first scene outvote the second, which is an artefact of
VLM chattiness rather than evidence about the pair.

Structurally impossible pairs are filtered with `is_candidate_pair` on gold
classes. list-all volunteers ~50 % of them (see E7_LISTALL_RESULTS.md §4), and
training on pairs the gate will never present at inference would spend capacity
on questions the shipped pipeline does not ask.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from affdet.intersection_matrix import is_candidate_pair


def reconcile_scene_scores(passes: list[list[dict]]
                           ) -> dict[tuple[str, str, str, str], float]:
    """Step 1: median across passes, per (scene, actor, recipient, predicate).

    A pair mentioned in only one pass keeps that pass's score rather than being
    dropped: it is still something the VLM volunteered, and requiring a
    majority would discard the unstable tail that is ~24 % of volunteered
    triples.
    """
    acc: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for one_pass in passes:
        for rec in one_pass:
            sid = rec["scene_id"]
            for s in rec.get("scores", []):
                acc[(sid, s["actor"], s["recipient"],
                     s["predicate"])].append(float(s["score"]))
    return {k: statistics.median(v) for k, v in acc.items()}


def build_label_corpus(passes: list[list[dict]], cls_of: dict[str, str],
                       min_scenes: int = 1) -> list[dict]:
    """Step 2: aggregate across scenes into v4-shaped rows.

    `cls_of` maps obj_id to its effective class (post-curation recataloguing),
    which is what the gate must be checked against -- an object named
    `knife_011` whose effective class is `peeler` is a peeler for gating
    purposes.

    `min_scenes` drops pairs backed by fewer than that many scenes. The default
    keeps everything; raising it trades corpus size for per-label confidence.
    """
    if min_scenes < 1:
        raise ValueError(f"min_scenes must be >= 1, got {min_scenes}")

    per_scene = reconcile_scene_scores(passes)

    by_pair: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_, actor, recipient, predicate), score in per_scene.items():
        ac, rc = cls_of.get(actor), cls_of.get(recipient)
        if ac is None or rc is None:
            continue                       # cannot be gate-checked
        if not is_candidate_pair(ac, rc, predicate):
            continue                       # structurally impossible
        by_pair[(actor, recipient, predicate)].append(score)

    rows = []
    for (actor, recipient, predicate), scores in sorted(by_pair.items()):
        if len(scores) < min_scenes:
            continue
        rows.append({
            "actor": actor,
            "recipient": recipient,
            "predicate": predicate,
            # Kept on the 0-10 scale the v4 corpus uses; the distillation path
            # rescales downstream. Emitting 0-1 here would shrink every target.
            "score": statistics.median(scores),
            "n_scenes": len(scores),
            "rationale": None,
        })
    return rows
