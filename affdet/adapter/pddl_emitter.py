"""PDDL emitter — rich prediction tuples → grounded PDDL problem files.

Phase 0 emits a minimal stub with only :init facts. Domain attachment and
recipient-face defaults are Phase 4 concerns.
"""
from __future__ import annotations
from pathlib import Path

from affdet.schemas import Predictions, PredictionEntry


_WHOLE_OBJECT_PREDICATES = {"canContain", "canFitOnto"}


def predicate_to_pddl(entry: PredictionEntry) -> str | None:
    """One positive PredictionEntry → PDDL ground atom. None for negatives."""
    if entry.verdict == 0:
        return None
    if entry.predicate in _WHOLE_OBJECT_PREDICATES:
        return f"({entry.predicate} {entry.actor} {entry.recipient})"
    if entry.actor_part is None:
        return None  # defensive: skip malformed positive entries
    return f"({entry.predicate} {entry.actor} {entry.actor_part} {entry.recipient})"


def predictions_to_pddl_strings(entries: list[PredictionEntry]) -> str:
    atoms = [predicate_to_pddl(e) for e in entries]
    return "\n".join(a for a in atoms if a is not None)


def _wrap_in_problem(scene_id: str, atoms: str, objects: list[str]) -> str:
    obj_decls = " ".join(objects)
    return f"""(define (problem {scene_id})
  (:domain affdet)
  (:objects {obj_decls})
  (:init
{atoms}
  )
  (:goal (and))
)
"""


def emit_pddl(predictions_dir: Path | str, pddl_dir: Path | str) -> None:
    predictions_dir = Path(predictions_dir)
    pddl_dir = Path(pddl_dir)
    pddl_dir.mkdir(parents=True, exist_ok=True)

    for preds_path in sorted(predictions_dir.glob("scene_*.json")):
        preds = Predictions.model_validate_json(preds_path.read_text())
        atoms = predictions_to_pddl_strings(preds.tier1)
        objs: set[str] = set()
        for e in preds.tier1:
            objs.add(e.actor)
            objs.add(e.recipient)
        problem = _wrap_in_problem(preds.scene_id, atoms, sorted(objs))
        (pddl_dir / f"{preds.scene_id}.pddl").write_text(problem)
