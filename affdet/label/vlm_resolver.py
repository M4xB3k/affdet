"""Resolve a scene (actor_instance, recipient_class, predicate) to a soft target.

Reads the VLM score table (per-instance actor, class recipient, 0-9 score) and
the normalized resolved_instances table. Returns score/9 in [0,1], or None when
there is no teacher signal (unknown/dropped instance) so the deriver drops the
pair. canPourInto has no VLM rows: it is resolved from `pourable` instead.
canPierce also has no VLM rows: it's a hardcoded rule (true for any valid
knife/fork actor against any food recipient except corn), not a visual
per-instance judgment worth VLM cost.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VlmScoreResolver:
    _scores: dict[tuple[str, str, str], float]  # (actor_key, food_class, aff) -> mean score
    _resolved: dict[str, dict]

    @classmethod
    def from_dicts(cls, vlm_rows: list[dict], resolved: dict) -> "VlmScoreResolver":
        acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for r in vlm_rows:
            acc[(r["actor"], r["food"], r["affordance"])].append(float(r["score"]))
        scores = {k: sum(v) / len(v) for k, v in acc.items()}
        return cls(_scores=scores, _resolved=resolved)

    @classmethod
    def from_files(cls, vlm_jsonl: Path | str, resolved_json: Path | str) -> "VlmScoreResolver":
        rows = [json.loads(l) for l in Path(vlm_jsonl).read_text().splitlines() if l.strip()]
        resolved = json.loads(Path(resolved_json).read_text())
        return cls.from_dicts(rows, resolved)

    def effective_class(self, obj_id: str, obj_class: str) -> str:
        rec = self._resolved.get(obj_id)
        return rec["effective_class"] if rec else obj_class

    def _actor_key(self, obj_id: str, obj_class: str) -> str:
        """Scene knife_011 -> peeler_011 (score rows keyed by effective class)."""
        eff = self.effective_class(obj_id, obj_class)
        if eff == obj_class:
            return obj_id
        # replace the class prefix, keep the numeric suffix
        num = obj_id.rsplit("_", 1)[-1]
        return f"{eff}_{num}"

    def soft_target(self, obj_id: str, obj_class: str,
                    recip_class: str, predicate: str,
                    recip_id: str | None = None) -> float | None:
        rec = self._resolved.get(obj_id)
        if rec and rec.get("dropped"):
            return None
        if predicate == "canPourInto":
            # Relational: actor must pour FROM and recipient must receive INTO.
            # Both come from the class-rule table in resolved_instances (per
            # instance), not from a VLM. Missing/dropped either side -> no signal.
            recip = self._resolved.get(recip_id) if recip_id is not None else None
            if rec is None or recip is None or recip.get("dropped"):
                return None
            if rec.get("pourable_from") is None or recip.get("pourable_into") is None:
                return None
            return 1.0 if (rec["pourable_from"] and recip["pourable_into"]) else 0.0
        if predicate == "canPierce":
            # Hardcoded rule, no VLM lookup: true for any valid knife/fork
            # actor against any food except corn.
            return 0.0 if recip_class == "corn" else 1.0
        key = (self._actor_key(obj_id, obj_class), recip_class, predicate)
        score = self._scores.get(key)
        return None if score is None else score / 9.0
