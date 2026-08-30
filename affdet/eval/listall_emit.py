"""Emit list-all labels in the per-scene shape the trainer consumes.

The training corpus is a DIRECTORY of one JSON per scene (see
`labels_all_merged_v4/`), each holding a `tier1` list. Two details are easy to
get wrong and silent when wrong:

  * `score` is normalised to **raw/9**, not the 0-10 the VLM emits. A 2 becomes
    0.2222. Writing 0-10 here would leave every target ~10x too large and the
    model would look broken for reasons unrelated to the labels.
  * `verdict` is the binarised label at the same >= 5 cutoff the human labels
    use, kept alongside the graded score.

`source` marks the provenance so a ledger entry can tell which teacher produced
a given checkpoint.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

#: The VLM emits 0-10; the corpus normalises by 9 (a 9 is a full-confidence
#: positive, 10 is clamped). Matches the observed values in v4.
SCORE_DIVISOR = 9.0
#: A raw score >= 5 is a positive, matching manual_tool._HUMAN_SCORE_THRESHOLD.
POS_THRESHOLD = 5.0
SOURCE = "vlm_listall_e8"


def to_unit_score(raw: float) -> float:
    """Map a 0-10 VLM score onto the corpus's 0-1 scale (raw/9, clamped)."""
    return min(float(raw) / SCORE_DIVISOR, 1.0)


def emit_scene_labels(scores: dict[tuple[str, str, str, str], float]
                      ) -> dict[str, dict]:
    """Group {(scene, actor, recipient, predicate): raw_score} by scene.

    Returns {scene_id: scene_label_doc} in the v4 document shape.
    """
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for (sid, actor, recipient, predicate), raw in scores.items():
        unit = to_unit_score(raw)
        by_scene[sid].append({
            "actor": actor,
            "recipient": recipient,
            "predicate": predicate,
            "verdict": 1 if float(raw) >= POS_THRESHOLD else 0,
            "actor_part": None,
            "confidence": unit,
            "source": SOURCE,
            "score": unit,
            "rationale": None,
        })
    out = {}
    for sid, rows in by_scene.items():
        rows.sort(key=lambda r: (r["actor"], r["recipient"], r["predicate"]))
        out[sid] = {"scene_id": sid, "tier1": rows,
                    "tier2_placeholder": None}
    return out


def write_label_dir(scores: dict[tuple[str, str, str, str], float],
                    out_dir: Path | str) -> int:
    """Write one JSON per scene; returns the number of scenes written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = emit_scene_labels(scores)
    for sid, doc in docs.items():
        (out_dir / f"{sid}.json").write_text(json.dumps(doc, indent=1))
    return len(docs)
