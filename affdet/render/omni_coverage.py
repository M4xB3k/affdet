"""Coverage-aware selection for balanced affordance training data.

The base OmniSceneSampler cycles realistic templates at random, which produces
wildly imbalanced per-affordance / pos-neg / per-instance coverage (the VLM
labels themselves range from canContain's 3526 pairs down to canPeel's 68). This
module layers a coverage controller on top: it tracks what has been generated so
far and biases each next scene toward whatever is under-covered.

Two affordance mechanisms are handled distinctly:
  - VLM-labelled (canCut/canPierce/canPeel/canContain/canSupport): balanced by
    pulling positive (score>=POS_MIN) / hard-negative (score<=NEG_MAX) actor-food
    pairs from AffordancePairIndex.
  - Rule-based canPourInto (not labelled): balanced by injecting scenes —
    positive = pourable + open container, negative = pourable + closed container.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

POS_MIN = 6   # VLM score >= this is a positive
NEG_MAX = 3   # VLM score <= this is a hard negative (4-5 skipped as ambiguous)

# Affordance that is rule-based (composition-driven), not in the VLM label file.
POURINTO = "canPourInto"

BUCKETS = ("pos", "neg")


def normalize_actor_id(actor_id: str) -> str:
    """Reduce a labelled actor id to its real mesh instance id.

    Some labels carry a face/part suffix after the numeric instance index
    (e.g. 'pan_001_diag', 'pan_003_top', 'bowl_007_front') — a labeling
    artifact. Strip everything after '<cat>_<digits>' so the id maps to an
    actual mesh. Plain ids ('knife_001', 'red_wine_glass_007') pass through.
    """
    parts = actor_id.split("_")
    for i, tok in enumerate(parts):
        if tok.isdigit():
            return "_".join(parts[: i + 1])  # keep up to and incl. the number
    return actor_id


def _actor_category(actor_id: str) -> str:
    """'knife_001' -> 'knife'. Handles red_wine_glass_007 -> red_wine_glass and
    face-suffixed ids (pan_001_diag -> pan) via normalize_actor_id."""
    norm = normalize_actor_id(actor_id)
    parts = norm.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else norm


@dataclass
class AffordancePairIndex:
    """Index of labelled (actor_instance, food_category) pairs by affordance and
    pos/neg bucket, loaded from omni_vlm_raw.jsonl."""

    # affordance -> bucket -> list[(actor_instance_id, food_category)]
    pairs: dict[str, dict[str, list[tuple[str, str]]]] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AffordancePairIndex":
        pairs: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
            lambda: {"pos": [], "neg": []}
        )
        seen: set[tuple[str, str, str, str]] = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                score = r["score"]
                if score >= POS_MIN:
                    bucket = "pos"
                elif score <= NEG_MAX:
                    bucket = "neg"
                else:
                    continue  # skip ambiguous mid scores
                key = (r["affordance"], bucket, r["actor"], r["food"])
                if key in seen:  # the label file has duplicate rows
                    continue
                seen.add(key)
                pairs[r["affordance"]][bucket].append((r["actor"], r["food"]))
        return cls(pairs=dict(pairs))

    def affordances(self) -> list[str]:
        return sorted(self.pairs.keys())

    def available(self, affordance: str, bucket: str) -> int:
        return len(self.pairs.get(affordance, {}).get(bucket, []))

    def actor_category(self, actor_id: str) -> str:
        return _actor_category(actor_id)


@dataclass
class CoverageTracker:
    """Running coverage tallies + per-affordance/bucket targets.

    Targets: split a scene-pair budget equally across affordances, then 50/50
    pos/neg, each CLAMPED to what the label index can actually provide
    (canPeel has only 68 pairs, so its target can't exceed that)."""

    index: AffordancePairIndex
    n_scenes: int
    pairs_per_scene_est: float = 3.0  # rough avg labelled pairs a scene yields
    include_pourinto: bool = True

    pair_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    instance_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    target: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        affs = list(self.index.affordances())
        if self.include_pourinto and POURINTO not in affs:
            affs.append(POURINTO)
        budget = int(self.n_scenes * self.pairs_per_scene_est)
        per_aff = budget / max(1, len(affs))
        per_bucket = per_aff / 2.0
        for aff in affs:
            self.pair_counts[aff] = {"pos": 0, "neg": 0}
            self.target[aff] = {}
            for b in BUCKETS:
                if aff == POURINTO:
                    cap = int(per_bucket)  # composition-driven, no hard cap
                else:
                    cap = min(int(per_bucket), self.index.available(aff, b))
                self.target[aff][b] = cap

    # --- deficits (used as selection weights) -----------------------------
    def deficit(self, aff: str, bucket: str) -> int:
        return max(0, self.target[aff][bucket] - self.pair_counts[aff][bucket])

    def most_under_covered(self, skip: set[tuple[str, str]] | None = None) -> tuple[str, str] | None:
        """(affordance, bucket) with the largest remaining deficit, or None if
        every target is met. `skip` excludes targets the sampler has found it
        cannot currently satisfy (e.g. no meshes available), so the loop moves
        on instead of livelocking on an unsatisfiable target."""
        skip = skip or set()
        best, best_d = None, 0
        for aff, buckets in self.target.items():
            for b in BUCKETS:
                if (aff, b) in skip:
                    continue
                d = self.deficit(aff, b)
                if d > best_d:
                    best, best_d = (aff, b), d
        return best

    def weighted_target(self, rng, skip: set[tuple[str, str]] | None = None) -> tuple[str, str] | None:
        """Pick an (affordance, bucket) at random with probability proportional
        to its remaining deficit. Unlike strict argmax, this reaches rare
        affordances (canPeel) from the start even when total targets exceed the
        scene budget — each is chosen ~ deficit/sum_deficits of the time, so the
        realised distribution tracks the target ratios instead of draining the
        biggest buckets first and starving the rest."""
        skip = skip or set()
        choices, weights = [], []
        for aff in self.target:
            for b in BUCKETS:
                if (aff, b) in skip:
                    continue
                d = self.deficit(aff, b)
                if d > 0:
                    choices.append((aff, b))
                    weights.append(d)
        if not choices:
            return None
        total = float(sum(weights))
        # np.random.RandomState uses .rand(); stdlib random uses .random().
        draw = rng.rand() if hasattr(rng, "rand") else rng.random()
        r = draw * total
        acc = 0.0
        for choice, w in zip(choices, weights):
            acc += w
            if r <= acc:
                return choice
        return choices[-1]

    def instance_deficit(self, instance_id: str, floor: int) -> int:
        return max(0, floor - self.instance_counts.get(instance_id, 0))

    # --- updates ----------------------------------------------------------
    def record_pair(self, aff: str, bucket: str) -> None:
        if aff in self.pair_counts:
            self.pair_counts[aff][bucket] += 1

    def record_instance(self, instance_id: str) -> None:
        self.instance_counts[instance_id] += 1

    # --- reporting --------------------------------------------------------
    def report(self) -> str:
        lines = ["coverage report:", f"{'affordance':14s} {'pos':>8s} {'neg':>8s}"]
        for aff in sorted(self.pair_counts):
            pc = self.pair_counts[aff]
            tg = self.target[aff]
            lines.append(
                f"{aff:14s} {pc['pos']:>4d}/{tg['pos']:<3d} {pc['neg']:>4d}/{tg['neg']:<3d}"
            )
        counts = list(self.instance_counts.values())
        if counts:
            counts.sort()
            lines.append(
                f"instances: {len(counts)} seen, "
                f"min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}"
            )
        return "\n".join(lines)
