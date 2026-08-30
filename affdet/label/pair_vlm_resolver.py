"""Resolve a scene (actor_instance, recipient_instance, predicate) to a soft
target using the pair-labeling VLM corpus (labels_omni.jsonl).

Distinct from vlm_resolver.py: that resolver was built for an older VLM
output schema keyed by {actor, food (a CLASS name), affordance}. The newer
pair-labeling run (affdet/scripts/label.py's OmniLabeler, gpt-5-mini/Azure,
8,249 pairs / 14,746 rows) scores specific (actor_instance, recipient_
instance) pairs, so a class-only key would silently discard the extra
per-instance signal already collected.

Lookup key here is (actor_instance, recipient_CATEGORY, PLACEMENT,
predicate), averaged across every labeled row whose recipient shares the
target's category+placement bucket. Placement (e.g. "whole" vs "piece" for
food, "opened" for containers) is read from curation.json's per-category
`placement` block. Validated 2026-07-13 against omni_scenes_v3_merged:
92.2% of VLM-relevant candidate triples resolve this way (see
check_coverage_tmp.py on the cluster); the residual gap was fully explained
(dropped/recategorized instances that shouldn't have been candidates at
all -- not a resolver defect).

canPourInto is NOT looked up here -- it stays rule-based
(resolved_instances.py's pourable_from/into flags), exactly as in
vlm_resolver.py: no amount of looking at an image tells you whether a
specific scan's lid is glued shut.

canPierce USED to be rule-based here too (hardcoded 1.0 for every knife/fork
actor except against corn). It is not anymore. As of 72d0ef6 the VLM judges
it directly -- piercing depends on tip sharpness, which is a visual property
a blunt-tipped novelty knife can fail -- so it resolves through the ordinary
score lookup like canCut. The old rule is why canPierce had zero negatives
corpus-wide and scored a degenerate F1 of 1.000: it could not be wrong.
Note this means canPierce needs real rows in the label corpus; corpora
predating 72d0ef6 (labels_omni_v3.jsonl and earlier) have none, and will now
correctly resolve to None ("no evidence") rather than a fabricated 1.0.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def _placement(inst_id: str, curation: dict) -> str | None:
    """obj_id -> its placement bucket ("whole"/"piece"/"opened"/...), or
    None if the category has no placement block or the number isn't listed."""
    parts = inst_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    cat, num = parts
    placement = curation.get(cat, {}).get("placement", {})
    for kind, nums in placement.items():
        if num in nums:
            return kind
    return None


@dataclass
class PairVlmResolver:
    _scores: dict[tuple[str, str, str, str], float]  # (actor_key, recip_class, placement, pred) -> mean score
    _resolved: dict[str, dict]
    _curation: dict

    @classmethod
    def from_dicts(cls, vlm_rows: list[dict], resolved: dict,
                   curation: dict) -> "PairVlmResolver":
        acc: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        for r in vlm_rows:
            recip_cls = r["recipient"].rsplit("_", 1)[0]
            plc = _placement(r["recipient"], curation)
            if plc is None:
                continue
            key = (r["actor"], recip_cls, plc, r["predicate"])
            acc[key].append(float(r["score"]))
        scores = {k: sum(v) / len(v) for k, v in acc.items()}
        return cls(_scores=scores, _resolved=resolved, _curation=curation)

    @classmethod
    def from_files(cls, vlm_jsonl: Path | str, resolved_json: Path | str,
                   curation_json: Path | str) -> "PairVlmResolver":
        rows = [json.loads(l) for l in Path(vlm_jsonl).read_text().splitlines() if l.strip()]
        resolved = json.loads(Path(resolved_json).read_text())
        curation = json.loads(Path(curation_json).read_text())
        return cls.from_dicts(rows, resolved, curation)

    def effective_class(self, obj_id: str, obj_class: str) -> str:
        rec = self._resolved.get(obj_id)
        return rec["effective_class"] if rec else obj_class

    def _actor_key(self, obj_id: str, obj_class: str) -> str:
        """Scene knife_011 -> peeler_011 (score rows keyed by effective class)."""
        eff = self.effective_class(obj_id, obj_class)
        if eff == obj_class:
            return obj_id
        num = obj_id.rsplit("_", 1)[-1]
        return f"{eff}_{num}"

    def soft_target(self, obj_id: str, obj_class: str,
                    recip_class: str, predicate: str,
                    recip_id: str) -> float | None:
        rec = self._resolved.get(obj_id)
        if rec and rec.get("dropped"):
            return None
        if predicate == "canPourInto":
            recip = self._resolved.get(recip_id)
            if rec is None or recip is None or recip.get("dropped"):
                return None
            if rec.get("pourable_from") is None or recip.get("pourable_into") is None:
                return None
            return 1.0 if (rec["pourable_from"] and recip["pourable_into"]) else 0.0
        plc = _placement(recip_id, self._curation)
        if plc is None:
            return None
        key = (self._actor_key(obj_id, obj_class), recip_class, plc, predicate)
        score = self._scores.get(key)
        if score is None:
            return None
        # The prompts advertise 0-10 while the scale is 9, so a compliant model
        # returning 10 would normalise above 1.0. No 10 occurs in the labelled
        # corpus; the clamp keeps a future prompt or model change from handing
        # the trainer a target outside the unit interval.
        return min(1.0, max(0.0, score / 9.0))
