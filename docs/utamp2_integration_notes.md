# Handoff: integrating affdet into UTAMP-2 (status 2026-10-14, end of day)

Context for continuing this work in a new session. Point the session at this
file. Branch: `claude/dino-geometry-affordances-e32leg` (pushed, in sync).

## Goal
Supervisors (Pouya Niaz, Alejandro Agostini) want the thesis work integrated
into UTAMP-2 (Niaz, Piater, Agostini, arXiv:2608.25641) for two affordances:
**can grasp** and **can support**, using their predicate definitions.
Supervisor meeting was Oct 14, 14:30 — record its outcome here.

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

## Decisions
- Single-object affordances are the easy part; derive `isP2PSupportive` from
  unary labels + geometry, never ask a VLM pairwise questions.
- **DINO for appearance, numbers for size.** Model input = DINOv2 features of
  the object/member crop + metric box dims (object and members, metres).
  Size sources: training = set when scaling meshes; benchmark = read from the
  scene (`sim.getShapeBB`); real robot = depth → point cloud → gravity-aligned OBB.
- **Labels are per object and per face, computed from the mesh**, never per
  category. Two similar shakers can differ (domed vs flat cap → upside-down
  supportable or not; round vs faceted body → lying on side stable or not).
  Pipeline: upright the mesh → scale to target height → box faces →
  stable poses (trimesh `compute_stable_poses`), face flatness/area, gripper
  width ≤ 80 mm across opposite faces, optional GraspGen grasps binned to faces.
  VLM answers only non-metric questions (handle/rim/body, material, fragile)
  on face-labelled multi-view renders, asked 3×; human review of disagreements.
- Shape variety within a category is wanted: nuisance variety (material,
  colour, facets) teaches invariance; label-changing variety (cap shape) forces
  the model to use size/geometry. Watch for confounds (e.g. all glass = round cap).
- Keep rendering whole scenes (real robot sees scenes); labels are per object,
  reused across scenes (thesis trick).
- Avoid AffordanceNet ("support" = spatula blade there). Off-the-shelf grasp
  baseline: GraspGen (Panda model) binned to faces.
- **VLM for the non-metric questions: GPT-6-sol** (Azure, $2 / $10 per M tokens,
  short context). Whole labelling run costs < ~$15 on any candidate, so pick
  on quality. Not benchmarked yet — run a ~20-object pilot (3× each) against
  own labels, comparing GPT-6-sol with GPT-5-mini (the thesis teacher, 83.2 %
  vs human labels) and o4-mini (UTAMP-2's model); keep the most consistent.
  Note 4o-mini is not cheaper for images (bills ~30× more tokens per image).

## Objects

### Sources (all in `docs/objaverse/`, see its README)
- `kitchen_candidates.csv` — 5,090 Objaverse candidates from DeepMind
  objaverse_annotations type labels; `previews/*.jpg`.
- `objathor_kitchen.csv` — candidates joined with **ObjaTHOR 2024_08_16**
  (GPT-4o; 625,252 objects; downloaded from Cloudflare R2, not HF:
  `https://pub-2619544d52bd4f35927b08d301d2aba0.r2.dev/annotations.json.gz`,
  357 MB). 4,539 / 5,090 covered. Fields: height / max dim (cm), mass (kg),
  materials, quality, licence, THOR bbox (m). Built by `scripts/join_objathor.py`.
- Hugging Face is reachable from cloud sessions now (Objaverse-LVIS,
  `object-paths.json.gz`, GLBs at `huggingface.co/datasets/allenai/objaverse/resolve/main/<path>`).

### Findings about ObjaTHOR numbers
- Coarse (5 cm / 0.5 kg steps); bbox derives from the same scale (not an
  independent check). Ranges per role and outlier kinds: far off / borderline
  (within one step) / wrong object (category names something else).
- `max_dimension` is defined as the longest side (≥ height) but GPT-4o often
  wrote the **width** for tall objects (39 shaker-category, 72 mugs). Fix: if
  max_dim < height, treat it as width and use height as longest side.
- Name matching is unreliable both ways: "shaker" catches cocktail/protein
  shakers; real shakers get called "container"/"bottle". Look at thumbnails.
- Bad sizes don't cost objects (we rescale meshes); only wrong objects do.

### Counts after removing wrong objects (automatic, quality ≥ 7)
mug ~1,750 (35 glass) · plate/bowl ~780 · tray ~295 · frying pan ~86 ·
cutting board ~61 · salt shaker 19 → salt shaker needed hand curation.

### Curated salt-shaker set — `salt_shaker_set.csv`, 46 objects
Picked by eye from 259 candidates (our list + Objaverse-LVIS saltshaker /
shaker / pepper_mill / condiment + every ObjaTHOR object named or described as
a shaker/mill/jar): 21 shakers, 11 pepper mills, 7 spice jars, 5 two-shaker
models split into 7 shakers (`scripts/split_pairs.py`: groups mesh parts by
top-down footprint overlap > 20 %, keeps original parts/materials; `note`
column says which halves to keep). Columns include target height/mass for
scaling (ObjaTHOR value if plausible, else 9 / 18 / 10 cm). Picks are from
thumbnails only — meshes not yet opened; 5 have no licence in ObjaTHOR.

### Review page
"Kitchen Object Sizes": https://claude.ai/artifact/TJXEXFtwZazn2sEKfAmhFJ
(private; share from its menu). Per role: ranges, histograms, typical objects,
outliers with thumbnails; salt-shaker tab also has the curated set and the 122
name mismatches. Rebuild with `scripts/size_review/` (see its README).

## Research report
`reports/UTAMP grasp and support perception.md` + notes in `research_notes/`.
Demo track: geometric predicate generator over the .ttt members, diffed
against UTAMP-2-GT. Paper track: DINOv2 student for face-level predicates +
predicate-noise → plan-success study on the 127 configurations.

## Next steps
1. Write the per-face labeller (pipeline above) and run it on the 46 salt
   shakers: download GLBs, upright, scale, label, render a few for checking.
2. Curate the other categories the same way (by eye, ~30–80 each); mugs,
   plates, trays, pans have plenty; cutting boards ~61.
3. VLM pilot (20 objects × 3 models × 3 repeats) once there is an Azure endpoint.
4. Ask Pouya: UTAMP-2 code, `llm_prompts_affd.py`, hand-made affordance table
   (ground truth to test the labeller on — especially their salt shaker),
   127 configurations, whether hf1/hf2 order matters, whether upside-down
   counts as supportable, canLift/canSlide scope, what the benchmark salt
   shaker's cap looks like.
5. Check GraspGen builds on the lab GPU.
6. Licences: check the ~60 non-commercial objects and the 5 without a licence;
   confirm ObjaTHOR annotation licence before redistributing the joined CSV.
