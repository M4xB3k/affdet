# Handoff: integrating affdet into UTAMP-2 (status 2026-10-14)

Context for continuing this work in a new session. Paste this file (or point
the session at it) to resume.

## Goal
Supervisors (Pouya Niaz, Alejandro Agostini) want the thesis work integrated
into UTAMP-2 (Niaz, Piater, Agostini, arXiv:2608.25641) for two affordances:
**can grasp** and **can support**, using their predicate definitions.
Next meeting: Oct 14, 14:30.

## UTAMP-2 facts (read from the paper PDF)
- Predicates (Table I): `isSupportive ?obj ?part`, `isSupportable ?obj ?part`,
  `isP2PSupportive ?part2 ?obj2 ?part1 ?obj1` (pairwise),
  `canGraspFrom ?obj ?mem ?hp ?hf1 ?hf2` (palm, finger1, finger2 faces of a member),
  `canLift ?obj ?mem` (mass), `canSlide ?obj1 ?obj2 ?p1 ?p2`.
- **Member** = rigidly attached sub-object (body / handle / edge) with its own
  bounding box parallel to the parent; authored by hand per object.
  **Part** = one of the 6 bbox faces (on, under, front, back, left, right) or `in`.
- Grasp types: wrap (body), edge, handle.
- Benchmark: 5 objects (salt shaker, glass mug, cutting board, frying pan,
  wooden tray), 127 stack configurations, goal = stack everything on the tray.
- UTAMP-2-GT 100% success; UTAMP-2-VLM (o4-mini attributes + rule parser) 90%,
  affordance P/R/F1 0.98 — failures from inconsistent VLM attributes dropping
  support facts. Sliding task: 1.0 / 0.81 / VLM 0.41.
- Size in UTAMP-2: VLM-estimated `bbox_size` primed by hardcoded few-shot dims
  in `llm_prompts_affd.py` (from old notes in `docs/taxonomy.md`; confirm with
  Pouya). Motion uses true bboxes from the CoppeliaSim scene.

## The .ttt scene (decoded; Huffman-compressed CoppeliaSim format)
Panda + gripper, table, workspace, `visionSensor`, `camera1`, `Gripper_Camera`,
the 5 objects with meshes, and member shapes:
cutting_board_body/handle; frying_pan_body/handle/edge_front/left/right;
glass_mug_body/handle/edge_front/left/right; salt_shaker_body;
wooden_tray_body/handle_front/handle_back/edge_left/edge_right;
plus grasp_* / place_* / support_* frames.

## Decisions reached in discussion
- Single-object affordances are the easy part; derive `isP2PSupportive` from
  unary labels + geometry, never ask a VLM pairwise questions.
- **DINO for appearance, numbers for size.** Model input = DINOv2 features of
  the object/member crop + metric box dims (object and members, metres).
  Size sources: training = set when scaling meshes; benchmark = read from the
  scene (`sim.getShapeBB`); real robot = depth → point cloud → gravity-aligned OBB.
- Labels: geometry computes metric parts (finger span ≤ 80 mm, flat faces,
  footprint fit, stable resting faces); VLM answers only non-metric questions
  (handle/rim/body, flat vs round, material, fragile, heavy-for-size) on
  face-labelled multi-view renders, asked 3×; human review of disagreements first.
- Keep rendering whole scenes (real robot sees scenes); labels are per object,
  reused across scenes (thesis trick).
- Scaling: use ObjaTHOR (allenai/objathor) GPT-4 annotations — height, max
  dimension, mass, materials, frontView (orientation), quality 0–10 — clamp to
  category ranges, review only outliers. `objathor.dataset.load_annotations()`.
- Avoid AffordanceNet ("support" = spatula blade there). Off-the-shelf grasp
  baseline: GraspGen (Panda model) binned to faces.

## Objects (Objaverse, via DeepMind objaverse_annotations type labels)
See `docs/objaverse/README.md`, `kitchen_candidates.csv` (5,090 rows),
`previews/*.jpg` (806 with thumbnails). Totals: shaker 183, mug 2,293,
cutting board 138, pan 200, tray 1,026, plate/bowl 1,250. Glass mugs only 18.
Plan: ~300 curated objects (~30–80 per category), ~24 views each, 10–20%
rendered in CoppeliaSim; estimated ~2.5–3 weeks full-time.

## Research report
`reports/UTAMP grasp and support perception.md` + notes in `research_notes/`.

## Open items
1. Run on a machine with Hugging Face access: Objaverse-LVIS counts, ObjaTHOR
   annotation overlap with `kitchen_candidates.csv`, per-object licences.
2. Ask Pouya: UTAMP-2 code, `llm_prompts_affd.py`, hand-made affordance table,
   127 configurations, whether hf1/hf2 order matters, canLift/canSlide scope.
3. Check GraspGen builds on the lab GPU.
4. ~~Fix GitHub access for this repo~~ — done 2026-10-14: the unpushed
   bundle was applied and pushed to `claude/dino-geometry-affordances-e32leg`.
