"""AffordanceDataset — flat in-memory dataset of (actor, recipient) pair feature vectors.

Loads all triplets (scene JSON, features NPZ, labels JSON) at construction.
Each item is (x, y, valid_mask) where:
  x          : float32 (1544,) = [f_A(768) | f_B(768) | geo(8)]
  y          : float32 (7,)    — label per TIER1_PREDICATES, 0 where not a candidate
  valid_mask : float32 (7,)    — 1 where pair is a candidate for that predicate

Impossible pairs (not a candidate for any predicate) are excluded entirely.
Split is by scene (not pair) to prevent leakage.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from affdet.schemas import Scene, Features, Labels
from affdet.intersection_matrix import TIER1_PREDICATES, is_candidate_pair
from affdet.label.resolved_instances import (
    load_effective_class_map, load_dropped_set, effective_class,
)

_N_PRED = len(TIER1_PREDICATES)  # 7
_PRED_IDX = {p: i for i, p in enumerate(TIER1_PREDICATES)}


def _find_scene_jsons(scenes_dir: Path) -> list[Path]:
    """Find scenes across all supported layouts:
      worker_N/scene_XXXX/scene_XXXX.json  (cluster, worker-sharded)
      scene_XXXX/scene_XXXX.json           (nested mock)
      scene_XXXX.json                      (flat mock)"""
    scene_jsons = sorted(scenes_dir.glob("worker_*/scene_*/scene_*.json"))
    if not scene_jsons:
        scene_jsons = sorted(scenes_dir.glob("scene_*/scene_*.json"))
    if not scene_jsons:
        scene_jsons = sorted(scenes_dir.glob("scene_*.json"))
    return scene_jsons


def scene_split_map(scenes_dir: Path | str, seed: int = 0) -> dict[str, str]:
    """Deterministic 70/15/15 train/val/test split by scene_id (the JSON
    stem), keyed on the same seeded shuffle AffordanceDataset uses internally
    -- the single source of truth both AffordanceDataset and Inferencer's
    split filter read from, so "test" always means the same held-out scenes
    in both training and evaluation."""
    scenes_dir = Path(scenes_dir)
    scene_jsons = _find_scene_jsons(scenes_dir)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(scene_jsons))
    n = len(scene_jsons)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    idx_to_split = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            idx_to_split[idx] = "train"
        elif i < n_train + n_val:
            idx_to_split[idx] = "val"
        else:
            idx_to_split[idx] = "test"
    return {
        scene_jsons[idx].stem: sp
        for idx, sp in idx_to_split.items()
    }


def _scene_obj_ids(scene_json: Path) -> list[str]:
    """Mesh-instance ids in a scene. `obj_id` (e.g. 'bowl_016') IS the concrete
    OmniObject3D instance identity -- there is no separate mesh field."""
    try:
        data = json.loads(scene_json.read_text())
    except Exception:
        return []
    return [o.get("obj_id") for o in data.get("objects", []) if o.get("obj_id")]


def _greedy_group_split(groups: dict[str, set[str]], seed: int) -> dict[str, str]:
    """Assign scenes to train/val/test so that no GROUP KEY (a mesh instance,
    or a mesh pair) ever spans two splits.

    `groups` maps scene_id -> the set of keys that scene contains. Scenes are
    linked into connected components by shared keys (union-find): if scene A
    and scene B both contain knife_003, they MUST land in the same split, or
    that mesh leaks. Components are then packed into 70/15/15 largest-first,
    which keeps the ratios close while making disjointness exact.

    Exact disjointness is the point -- an approximate split would silently
    reintroduce the leakage this exists to remove.
    """
    parent: dict[str, str] = {s: s for s in groups}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    key_owner: dict[str, str] = {}
    for scene_id, keys in groups.items():
        for k in keys:
            if k in key_owner:
                union(key_owner[k], scene_id)
            else:
                key_owner[k] = scene_id

    components: dict[str, list[str]] = {}
    for scene_id in groups:
        components.setdefault(find(scene_id), []).append(scene_id)

    rng = np.random.RandomState(seed)
    comps = [sorted(c) for c in components.values()]
    order = rng.permutation(len(comps))
    comps = [comps[i] for i in order]
    comps.sort(key=len, reverse=True)

    n = len(groups)
    quota = {"train": 0.70 * n, "val": 0.15 * n, "test": 0.15 * n}
    filled = {"train": 0, "val": 0, "test": 0}

    out: dict[str, str] = {}
    for comp in comps:
        # Whichever split is furthest below its quota takes the next component.
        target = max(quota, key=lambda s: quota[s] - filled[s])
        for scene_id in comp:
            out[scene_id] = target
        filled[target] += len(comp)
    return out


def instance_split_map(scenes_dir: Path | str, seed: int = 0) -> dict[str, str]:
    """Split so no MESH INSTANCE appears on both sides.

    Answers "does the model generalize to objects it has never seen?" -- the
    claim the thesis's instance-level principle (P2) actually needs, and the
    one the scene split cannot support.
    """
    scenes_dir = Path(scenes_dir)
    groups = {p.stem: set(_scene_obj_ids(p)) for p in _find_scene_jsons(scenes_dir)}
    return _greedy_group_split(groups, seed)


def pair_split_map(scenes_dir: Path | str, seed: int = 0) -> dict[str, str]:
    """Split so no unordered MESH PAIR co-occurs on both sides.

    Weaker than instance-disjoint (individual meshes may recur) but stronger
    than scene-level: it removes the 30.6% of test triples that are the exact
    (actor, recipient, predicate) question already asked during training.
    """
    scenes_dir = Path(scenes_dir)
    groups: dict[str, set[str]] = {}
    for p in _find_scene_jsons(scenes_dir):
        ids = sorted(set(_scene_obj_ids(p)))
        groups[p.stem] = {f"{a}|{b}" for i, a in enumerate(ids) for b in ids[i + 1:]}
    return _greedy_group_split(groups, seed)


@dataclass(frozen=True)
class InstanceHoldout:
    """Which mesh instances are held out, for a pair-level instance split.

    Scene-level assignment cannot express instance-disjointness on this corpus:
    `fork_003` alone appears in 1,478 of 12,753 scenes and there are only 3 fork
    meshes, so union-find over shared meshes chains everything into one
    component and `instance_split_map` puts every scene in train (empty test).

    The unit of evaluation is the PAIR, so hold out meshes and route each pair
    by which of its meshes are novel. One scene can then feed pairs to train and
    to test at once, which is correct -- nothing about a scene is being claimed.
    """
    test_meshes: frozenset[str]
    val_meshes:  frozenset[str]
    all_meshes:  frozenset[str]

    def to_dict(self) -> dict:
        """Ledger record -- a run is not reproducible without knowing WHICH
        meshes were drawn (H2)."""
        return {"test_meshes": sorted(self.test_meshes),
                "val_meshes":  sorted(self.val_meshes),
                "all_meshes":  sorted(self.all_meshes)}

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceHoldout":
        return cls(frozenset(d["test_meshes"]), frozenset(d["val_meshes"]),
                   frozenset(d["all_meshes"]))


def instance_holdout(scenes_dir: Path | str, seed: int = 0,
                     frac: float = 0.20,
                     effective_classes: dict[str, str] | None = None,
                     ) -> InstanceHoldout:
    """Draw a disjoint val/test mesh-instance holdout.

    `frac` is held out for test and again for val, stratified by category so a
    category with few instances (fork has 3, pan 4) still contributes instead of
    being lost to a lucky draw.

    Each seed resamples WHICH meshes are held out. That is deliberate: for the
    low-instance predicates draw variance dominates training-seed variance
    (canPourInto's strict test size ranges 106-163 across draws at 20%), so a
    variance estimate has to average over draws, not just over training seeds.

    `effective_classes` maps obj_id -> its recategorised class and, when given,
    becomes the stratification key in place of the id prefix. This matters
    because a predicate's actors can be a small RECATEGORISED subset of a large
    id-prefix category: canPeel's four actors (knife_011/014/018/021) are
    peelers sitting inside knife's 27 meshes, so prefix stratification draws
    k=5 of 27 and misses every peeler with probability
    comb(23,5)/comb(27,5) = 41.7% per seed. Only 19.8% of 3-seed runs then get
    all three seeds populated, and 7.2% lose canPeel outright. That is exactly
    what happened on the v4 ladder: seed 0 drew knives 002/003/024/025/030, so
    canPeel had zero test pairs and its reported F1 was a placeholder.

    Passing the map puts peelers in their own stratum, where `max(1, ...)`
    guarantees one in test and one in val on every seed. Omitting it preserves
    the old prefix behaviour exactly, so previously-committed runs stay
    reproducible.

    Even with a populated arm, callers reporting per-predicate results must
    still check for an empty arm rather than averaging a 0 in -- a stratum of
    one mesh can be split only so far.
    """
    scenes_dir = Path(scenes_dir)
    eff = effective_classes or {}
    by_cat: dict[str, set[str]] = {}
    for p in _find_scene_jsons(scenes_dir):
        for oid in _scene_obj_ids(p):
            key = eff.get(oid) or oid.rsplit("_", 1)[0]
            by_cat.setdefault(key, set()).add(oid)

    rng = np.random.RandomState(seed)
    test: set[str] = set()
    val:  set[str] = set()
    for _cat, meshes in sorted(by_cat.items()):
        ordered = sorted(meshes)
        rng.shuffle(ordered)
        n = len(ordered)
        # At least one mesh per side wherever the category can spare it, so
        # small categories are represented rather than silently dropped.
        k = max(1, int(round(n * frac))) if n >= 2 else 0
        test.update(ordered[:k])
        val.update(ordered[k:k + k] if n >= 2 * k + 1 else [])
    all_meshes = frozenset().union(*by_cat.values()) if by_cat else frozenset()
    return InstanceHoldout(frozenset(test), frozenset(val), all_meshes)


#: Pair-level instance-holdout modes (E1c and E1c' in the findings doc).
HOLDOUT_MODES = ("strict", "actor_only")


def pair_split_for(actor_id: str, recipient_id: str, holdout: InstanceHoldout,
                   mode: str = "strict") -> str | None:
    """Route one (actor, recipient) pair to train/val/test, or None to drop it.

    strict      (E1c)  both meshes novel -> the "unseen instance" claim
    actor_only  (E1c') actor mesh novel  -> the "unseen tool" claim

    Train is kept strictly clean in both modes: any pair touching a held-out
    mesh that does not qualify for val/test is DROPPED, never trained on.
    Letting it train would leak the very mesh the split exists to hold out.
    """
    if mode not in HOLDOUT_MODES:
        raise ValueError(
            f"unknown holdout mode {mode!r}; expected one of {HOLDOUT_MODES}")

    if mode == "strict":
        for split, held in (("test", holdout.test_meshes),
                            ("val", holdout.val_meshes)):
            if actor_id in held and recipient_id in held:
                return split
        touched = holdout.test_meshes | holdout.val_meshes
        if actor_id in touched or recipient_id in touched:
            return None          # partially novel -- neither clean train nor a
        return "train"           # valid strict-novel eval pair

    if actor_id in holdout.test_meshes:
        return "test"
    if actor_id in holdout.val_meshes:
        return "val"
    touched = holdout.test_meshes | holdout.val_meshes
    if recipient_id in touched:
        return None              # recipient novelty is not this arm's claim,
    return "train"               # but it must still not leak into train


#: Split strategies available to ExperimentConfig.split_strategy.
SPLIT_STRATEGIES = ("scene", "pair", "instance")


def build_split_map(scenes_dir: Path | str, strategy: str = "scene",
                    seed: int = 0) -> dict[str, str]:
    """Dispatch to a split strategy by name (the E1 generalization ladder)."""
    if strategy == "scene":
        return scene_split_map(scenes_dir, seed)
    if strategy == "pair":
        return pair_split_map(scenes_dir, seed)
    if strategy == "instance":
        return instance_split_map(scenes_dir, seed)
    raise ValueError(
        f"unknown split strategy {strategy!r}; expected one of {SPLIT_STRATEGIES}")


def _geo(obj_a, obj_b, W: float, H: float) -> list[float]:
    """8-d normalised bbox features for (actor, recipient)."""
    def _cx(b): return (b[0] + b[2]) / 2
    def _cy(b): return (b[1] + b[3]) / 2
    def _w(b):  return b[2] - b[0]
    def _h(b):  return b[3] - b[1]
    a, b = obj_a.bbox, obj_b.bbox
    return [
        _cx(a) / W, _cy(a) / H, _w(a) / W, _h(a) / H,
        _cx(b) / W, _cy(b) / H, _w(b) / W, _h(b) / H,
    ]


@dataclass
class AffordanceDataset(Dataset):
    scenes_dir:   Path | str
    features_dir: Path | str
    labels_dir:   Path | str
    split:        Literal["train", "val", "test"] | None = None
    split_strategy: Literal["scene", "pair", "instance"] = "scene"
    seed:         int = 0
    holdout_mode: str | None = None
    """When set ("strict" or "actor_only"), route each PAIR by mesh-instance
    novelty instead of assigning whole scenes (E1c / E1c'). Scene-level
    assignment cannot express instance-disjointness on this corpus -- see
    `InstanceHoldout`. Takes precedence over `split_strategy`."""
    holdout_frac: float = 0.20
    resolved_instances: Path | str | None = None
    unbounded:    bool = False
    """Keep every predicate in the loss mask for every ordered pair, instead
    of excluding structurally-impossible (actor, recipient, predicate)
    triples entirely. Gated-out predicates with no real label become hard
    negatives (y=0) rather than "never seen" — trains the model to actually
    reject nonsense pairs (e.g. knife-canCut-knife), which the default mode
    cannot do since it gets zero gradient there. Off by default: changes
    class balance a lot, so existing gated results stay reproducible."""
    unbounded_neg_ratio: int | None = None
    """Only meaningful when unbounded=True. Caps impossible-pair (hard
    negative) rows kept per scene to ratio x (candidate rows in that scene),
    subsampled with the dataset's own seeded RNG. None (default) keeps every
    impossible pair uncapped — a large hard-negative flood observed to hurt
    predicates with few structural candidates (e.g. canCut, canPierce) by
    diluting their positive signal despite pos_weight capping."""

    _xs:    torch.Tensor = field(init=False, repr=False)
    _ys:    torch.Tensor = field(init=False, repr=False)
    _masks: torch.Tensor = field(init=False, repr=False)
    _candidate_masks: torch.Tensor = field(init=False, repr=False)
    """The gate-only mask (is_candidate_pair), independent of `unbounded`.
    Equal to `_masks` when unbounded=False. Lets a Trainer keep evaluating F1
    on the original candidate pairs even when training on the wider
    unbounded mask, so eval stays comparable to non-unbounded runs."""
    _active_idx: list[int] = field(init=False, repr=False)
    _holdout: "InstanceHoldout | None" = field(init=False, repr=False, default=None)
    """The drawn mesh holdout when `holdout_mode` is set; None otherwise.
    Exposed so the runner can record it in the ledger (H2) -- a pair-level run
    is not reproducible without knowing which meshes were drawn."""

    def __post_init__(self):
        self.scenes_dir   = Path(self.scenes_dir)
        self.features_dir = Path(self.features_dir)
        self.labels_dir   = Path(self.labels_dir)
        eff_map = (load_effective_class_map(self.resolved_instances)
                   if self.resolved_instances is not None else None)
        dropped = (load_dropped_set(self.resolved_instances)
                   if self.resolved_instances is not None else set())
        xs, ys, masks, candidate_masks = [], [], [], []

        scene_jsons = _find_scene_jsons(self.scenes_dir)
        # A pair-level holdout keeps EVERY scene -- the filtering happens per
        # pair below, since one scene legitimately contributes to several
        # splits at once.
        if self.holdout_mode is not None:
            self._holdout = instance_holdout(self.scenes_dir, seed=self.seed,
                                             frac=self.holdout_frac,
                                             effective_classes=eff_map)
            split_map = None
        else:
            self._holdout = None
            split_map = build_split_map(self.scenes_dir, self.split_strategy,
                                        seed=self.seed)

        for scene_idx, scene_json in enumerate(scene_jsons):
            if (split_map is not None and self.split is not None
                    and split_map[scene_json.stem] != self.split):
                continue

            # Resolve features/labels WORKER-RELATIVE, not by stem alone — scene
            # IDs are unique only within a worker (worker_0 and worker_5 both
            # have scene_0001). `rel` is the worker dir (worker_N) for sharded
            # layouts, or "." for flat/nested-mock. Scenes are nested one level
            # deeper (scene_XXXX/scene_XXXX.json) than features/labels, so we
            # take the scene-dir's parent relative to scenes_dir.
            rel = scene_json.parent.relative_to(self.scenes_dir).parent
            feat_path  = self.features_dir / rel / f"{scene_json.stem}.npz"
            label_path = self.labels_dir   / rel / f"{scene_json.stem}.json"
            if not feat_path.exists() or not label_path.exists():
                continue

            scene = Scene.model_validate_json(scene_json.read_text())
            # A scene containing ANY curation-dropped instance is untrustworthy
            # as a whole (the render-time sampler placed it there despite the
            # drop list) -- exclude every pair in it, even pairs between two
            # otherwise-valid objects, rather than excise only the bad instance.
            if any(o.obj_id in dropped for o in scene.objects):
                continue

            feats  = Features.load(feat_path)
            labels = Labels.model_validate_json(label_path.read_text())
            W, H   = float(scene.image_size[0]), float(scene.image_size[1])

            feat_by_id = {oid: feats.features[i] for i, oid in enumerate(feats.obj_ids)}
            label_by_key: dict[tuple[str, str, str], object] = {
                (e.actor, e.recipient, e.predicate): e
                for e in labels.tier1
            }

            scene_xs, scene_ys, scene_masks, scene_cmasks = [], [], [], []

            for actor in scene.objects:
                a_eff = effective_class(actor.obj_id, actor.obj_class, eff_map)
                for recip in scene.objects:
                    if actor.obj_id == recip.obj_id:
                        continue
                    if self._holdout is not None and self.split is not None:
                        # None = the pair touches a held-out mesh without
                        # qualifying for this arm's eval set; dropping it is
                        # what keeps train clean of the "novel" meshes.
                        if pair_split_for(actor.obj_id, recip.obj_id,
                                          self._holdout,
                                          mode=self.holdout_mode) != self.split:
                            continue
                    valid_mask     = np.zeros(_N_PRED, dtype=np.float32)
                    candidate_mask = np.zeros(_N_PRED, dtype=np.float32)
                    y_vec          = np.zeros(_N_PRED, dtype=np.float32)
                    for pred in TIER1_PREDICATES:
                        pi = _PRED_IDX[pred]
                        candidate = is_candidate_pair(a_eff, recip.obj_class, pred)
                        if candidate:
                            candidate_mask[pi] = 1.0
                        elif not self.unbounded:
                            continue
                        valid_mask[pi] = 1.0
                        key = (actor.obj_id, recip.obj_id, pred)
                        entry = label_by_key.get(key)
                        if entry is None:
                            y_vec[pi] = 0.0  # no label -> negative (real or hard, unbounded)
                        elif entry.score is not None:
                            y_vec[pi] = float(entry.score)
                        else:
                            y_vec[pi] = float(entry.verdict)
                    if valid_mask.sum() == 0:
                        continue

                    f_a = feat_by_id[actor.obj_id].astype(np.float32)
                    f_b = feat_by_id[recip.obj_id].astype(np.float32)
                    geo = np.array(_geo(actor, recip, W, H), dtype=np.float32)
                    scene_xs.append(np.concatenate([f_a, f_b, geo]))
                    scene_ys.append(y_vec)
                    scene_masks.append(valid_mask)
                    scene_cmasks.append(candidate_mask)

            if self.unbounded and self.unbounded_neg_ratio is not None and scene_xs:
                is_candidate_row = [cm.sum() > 0 for cm in scene_cmasks]
                cand_idx = [i for i, c in enumerate(is_candidate_row) if c]
                impossible_idx = [i for i, c in enumerate(is_candidate_row) if not c]
                cap = self.unbounded_neg_ratio * len(cand_idx)
                if len(impossible_idx) > cap:
                    scene_rng = np.random.RandomState(self.seed + scene_idx)
                    impossible_idx = list(
                        scene_rng.choice(impossible_idx, size=cap, replace=False)
                    )
                keep = sorted(cand_idx + impossible_idx)
                scene_xs     = [scene_xs[i] for i in keep]
                scene_ys     = [scene_ys[i] for i in keep]
                scene_masks  = [scene_masks[i] for i in keep]
                scene_cmasks = [scene_cmasks[i] for i in keep]

            xs.extend(scene_xs); ys.extend(scene_ys)
            masks.extend(scene_masks); candidate_masks.extend(scene_cmasks)

        self._xs    = torch.from_numpy(np.stack(xs))    if xs else torch.zeros(0, 1544)
        self._ys    = torch.from_numpy(np.stack(ys))    if ys else torch.zeros(0, _N_PRED)
        self._masks = torch.from_numpy(np.stack(masks)) if masks else torch.zeros(0, _N_PRED)
        self._candidate_masks = (torch.from_numpy(np.stack(candidate_masks))
                                  if candidate_masks else torch.zeros(0, _N_PRED))
        self._active_idx = list(range(len(self._xs)))

    def __len__(self) -> int:
        return len(self._active_idx)

    def __getitem__(self, idx: int):
        i = self._active_idx[idx]
        return self._xs[i], self._ys[i], self._masks[i]

    def balance_masks(self, seed: int = 0) -> torch.Tensor:
        """Downsample the majority class PER PREDICATE to a true 50/50 split,
        by zeroing loss-mask entries. Returns the updated mask tensor.

        pos_weight (train.py::_compute_pos_weights) reweights the LOSS but
        leaves the DATA skewed (canSupport is ~89% positive in the training
        split). This instead makes the sample itself balanced.

        Masking rather than dropping rows is what keeps this per-PREDICATE: a
        pair usually carries several predicates with different balances, so
        dropping the row to fix one predicate would distort all the others.
        (That is exactly the flaw in resample() below, which works per-PAIR --
        it counts a pair positive if ANY predicate is positive, and since most
        predicates are majority-positive, nearly every pair reads as positive
        and almost nothing gets rebalanced.)

        A predicate with zero negatives (canPierce, corpus-wide) cannot be
        balanced at all and is masked out entirely rather than left at 100%
        positive -- a degenerate predicate contributes a meaningless F1 of
        1.000 and inflates the macro average.

        Train-time only: `_candidate_masks` (which drives evaluation) is left
        untouched, so eval stays comparable to unbalanced runs.
        """
        rng = np.random.RandomState(seed)
        n_pred = self._masks.shape[1] if len(self._masks) else 0

        for i in range(n_pred):
            valid = torch.nonzero(self._masks[:, i] > 0, as_tuple=True)[0]
            if len(valid) == 0:
                continue
            # Targets are soft VLM scores in [0,1]; class counting is binary,
            # so threshold at 0.5 -- the same boundary that derives `verdict`.
            is_pos = self._ys[valid, i] >= 0.5
            pos_idx = valid[is_pos]
            neg_idx = valid[~is_pos]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                # Degenerate: nothing to balance against. Disable entirely.
                self._masks[valid, i] = 0.0
                continue

            keep_n = min(len(pos_idx), len(neg_idx))
            for group in (pos_idx, neg_idx):
                if len(group) > keep_n:
                    drop = rng.choice(len(group), size=len(group) - keep_n,
                                      replace=False)
                    self._masks[group[drop], i] = 0.0

        return self._masks

    def resample(self, ratio: int = 5, seed: int = 0) -> list[int]:
        """Return a new active index list: all positives + ratio×n_pos negatives.

        Updates self._active_idx in-place and also returns it for testing.
        """
        rng = np.random.RandomState(seed)
        all_idx = list(range(len(self._xs)))
        pos_idx = [i for i in all_idx if self._ys[i].max() > 0]
        neg_idx = [i for i in all_idx if self._ys[i].max() == 0]
        n_neg = min(ratio * len(pos_idx), len(neg_idx))
        sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist() if n_neg else []
        self._active_idx = pos_idx + sampled_neg
        rng.shuffle(self._active_idx)
        return self._active_idx
