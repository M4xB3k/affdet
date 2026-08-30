"""Normalize curation.json into a flat per-instance resolution table.

curation.json is the source of truth (read live by render/sampler/scale/label)
and is never hand-edited. This build step reads it plus the pan-artifact
exclusion and emits one uniform entry per instance, with every ambiguity
pre-resolved: effective class (peeler recategorization), dropped + reason,
and a single `pourable` bool for pour vessels (from the asymmetric open_closed
lists). Downstream (vlm_resolver) reads THIS file, not raw curation.json.
"""
from __future__ import annotations

import json
from pathlib import Path

# pan_001 / pan_003 were VLM-scored across 3 conflicting camera views
# (pan_00N_diag/front/top). Not a curation decision — a VLM-table artifact.
PAN_DROP: set[str] = {"pan_001", "pan_003"}

# --- canPourInto pour semantics (relational, 2026-07-05) ---
# canPourInto(actor, recipient) = actor.pourable_from AND recipient.pourable_into.
# The old single `pourable` flag (from curation open_closed) was buggy: it read a
# teapot's LID as "closed" and dropped ~all teapots/kettles, though a lidded
# teapot still pours from its spout. We therefore encode the user-dictated truth
# table directly as class rules here, NOT from curation open_closed.
#
# A rule is one of: "all", "none", or ("only", {nums...}).  from_rule = can pour
# OUT; into_rule = can be poured INTO. Any class not listed here is neither.
_POUR_RULES: dict[str, dict[str, object]] = {
    "bottle":         {"from": "none", "into": "none"},           # all sealed -- ruled out entirely, not asked to the VLM
    "thermos":        {"from": ("only", {"002", "004", "005"}), "into": "none"},  # only 2/4/5 verified pourable; rest closed lids
    "kettle":         {"from": "all",  "into": "none"},
    "teapot":         {"from": "all",  "into": ("only", {"014", "035", "036", "037", "038"})},
    "vase":           {"from": ("only", {"004", "012"}), "into": ("except", {"011"})},
    "cup":            {"from": "none", "into": ("except", {"002", "009", "033", "038", "039"})},
    "bowl":           {"from": "none", "into": "all"},
    "dish":           {"from": "none", "into": "all"},            # real dishes hold water
    "pan":            {"from": "none", "into": "all"},
    "red_wine_glass": {"from": "none", "into": "all"},
}
# Classes that participate in canPourInto at all (get from/into flags emitted).
_POUR_CLASSES = set(_POUR_RULES)


def _apply_pour_rule(rule, num: str) -> bool:
    if rule == "all":
        return True
    if rule == "none":
        return False
    kind, nums = rule
    if kind == "only":
        return num in nums
    if kind == "except":
        return num not in nums
    raise ValueError(f"bad pour rule: {rule!r}")


def _iid(category: str, num: str) -> str:
    return f"{category}_{num}"


def _normalize_recat(recat) -> dict[str, str]:
    """Both curation shapes -> {num: new_class}."""
    if isinstance(recat, dict):
        return dict(recat)
    if isinstance(recat, list):
        return {item["tile"]: item["to"] for item in recat}
    return {}


def build_resolved(curation: dict) -> dict[str, dict]:
    """curation dict -> {obj_id: {effective_class, dropped, reason, pourable}}."""
    resolved: dict[str, dict] = {}

    def _ensure(oid: str, eff_class: str) -> dict:
        return resolved.setdefault(oid, {
            "effective_class": eff_class,
            "dropped": False,
            "reason": None,
            "pourable_from": None,
            "pourable_into": None,
        })

    for category, entry in curation.items():
        if category.startswith("_") or not isinstance(entry, dict):
            continue

        norm_recat = _normalize_recat(entry.get("recategorize", {}))
        oc = entry.get("open_closed", {}) or {}
        open_ids = set(oc.get("open", []))
        closed_ids = set(oc.get("closed", []))
        drop_ids = set(entry.get("drop", []))
        # every instance number this category mentions
        nums = (set(norm_recat) | open_ids | closed_ids | drop_ids)

        for num in nums:
            eff = norm_recat.get(num, category)
            oid = _iid(category, num)
            rec = _ensure(oid, eff)
            if num in drop_ids:
                rec["dropped"] = True
                rec["reason"] = "curation_drop"
            # canPourInto flags come from the class-rule table, keyed on the
            # EFFECTIVE class (a recategorized china->vase gets vase's rules),
            # NOT from curation open_closed (which was lid-based and wrong).
            pour = _POUR_RULES.get(eff)
            if pour is not None:
                rec["pourable_from"] = _apply_pour_rule(pour["from"], num)
                rec["pourable_into"] = _apply_pour_rule(pour["into"], num)

    for oid in PAN_DROP:
        rec = resolved.setdefault(oid, {
            "effective_class": "pan", "dropped": False,
            "reason": None, "pourable_from": None, "pourable_into": None})
        rec["dropped"] = True
        rec["reason"] = "multiview_vlm_artifact"

    return resolved


def write_resolved(curation_path: Path | str, out_path: Path | str) -> int:
    curation = json.loads(Path(curation_path).read_text())
    resolved = build_resolved(curation)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(resolved, indent=2, sort_keys=True))
    return len(resolved)


def load_effective_class_map(resolved_json: Path | str) -> dict[str, str]:
    """obj_id -> effective_class, from a resolved_instances.json file."""
    data = json.loads(Path(resolved_json).read_text())
    return {oid: rec["effective_class"] for oid, rec in data.items()}


def load_dropped_set(resolved_json: Path | str) -> set[str]:
    """obj_ids marked dropped=True in a resolved_instances.json file --
    e.g. pan_001/pan_003 (multiview_vlm_artifact). This is THE single source
    of truth for "is this specific instance usable at all" -- any renderer
    or pair-selection step that reads raw curation.json's `drop` list
    directly instead of this file risks drifting from it (curation.json's
    own note can even give a different reason for the same drop, as it does
    for pan -- "broken geometry" there vs. multiview_vlm_artifact here; this
    file is the one that actually gates training/labeling)."""
    data = json.loads(Path(resolved_json).read_text())
    return {oid for oid, rec in data.items() if rec.get("dropped")}


def effective_class(obj_id: str, obj_class: str,
                    eff_map: dict[str, str] | None) -> str:
    """Effective class for gating; falls back to obj_class when unmapped/no map."""
    if eff_map is None:
        return obj_class
    return eff_map.get(obj_id, obj_class)
