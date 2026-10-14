# Handoff: integrating affdet into UTAMP-2 (status 2026-10-14, evening)

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

- **No weight in the output.** Mass enters only through `canLift` (object
  "significantly lighter than the payload" when grasped from that member; the
  pan lifts by its handle but not by an edge; the tray is too heavy, so it is
  slid). Support predicates are booleans too. Mass and COM therefore stay
  inside our labeller (canLift, pairwise stack stability for isP2PSupportive);
  the output stays member boxes + faces + predicates. The paper assumes
  placement at the centroid of a flat surface and grasps at the middle of a
  member (listed as limitations).
- **Benchmark objects as the paper describes them (ground truth hints):**
  - Salt shaker: cylindrical with a round head, so it supports nothing; stands
    only on its bottom; "can only be grasped stably from its top side". Our
    width rule also allows side grasps, so ask whether that is a deliberate
    restriction.
  - Glass mug: supports small objects (the shaker) but not large ones; stands
    on its top and bottom but not its round sides; wrap-grasped from top or
    bottom, or by the handle.
  - Cutting board: supports things on top; cannot be placed upside down;
    handle grasp only.
  - Frying pan: stands only on its bottom; handle grasp only (too large to
    wrap, too heavy to edge-grasp); supports only the cutting board.
  - Wooden tray: not liftable but slidable; graspable by handles and edges;
    supports everything; upright only.

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
- **VLM for the non-metric questions: GPT-6-sol** (Azure, $2 / $10 per M tokens).
  It is the only model the Azure key serves, so GPT-5-mini / o4-mini are not
  compared. Pilot done (see Results). Access: endpoint
  `https://personale.services.ai.azure.com/openai/v1/` (Responses API),
  deployment `gpt-6-sol`, key in env `AZURE_TOKEN` sent as the `api-key` header.
  The older scripts (`affdet/label/vlm.py` etc.) read
  `AZURE_OPENAI_ENDPOINT=https://personale.services.ai.azure.com` and
  `AZURE_OPENAI_API_KEY`; set those from the same values to reuse them.
- **Supportability is judged with the face down, allowing a settle of up to 20°
  onto a nearby stable facet**, using the convex hull and a hull-centroid COM.
  Thresholds: facets within 5° of flat count as contact; COM margin
  ≥ max(5 mm, ½ face width × sin 10°); tip angle ≥ 5°. The size-relative
  margin stops finely tessellated round bodies (an 84-gon of radius 4.5 cm)
  from passing, while a 0.4 mm cone on a cap still counts as flat. Low-poly
  bodies shaded smooth get a `review` flag rather than a changed label.
  Grasps: Franka Hand, 80 mm opening, fingers reaching 4 cm past the palm face.
  Grasps from every face are listed, including `under`.

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

## Results so far

### Per-face labeller (`affdet/geometry/`, `scripts/label_faces.py`)
Install with `pip install -e .[geometry]`; on a headless machine set
`PYOPENGL_PLATFORM=osmesa` (needs `apt install libosmesa6`) for check renders.
The 46 salt shakers are labelled:
- `docs/objaverse/salt_shaker_face_labels.csv`
- `docs/objaverse/salt_shaker_face_labels_overview.jpg`
- per-object JSON and 4-view check sheets in `data/face_labels/salt_shaker/`
  (gitignored; regenerate in ~1.5 min)

GLBs are in `data/objaverse/glbs/` (gitignored, 148 MB, from HF). Uprighting
needed only the two crank grinders (`up_axis=-x` column); one tilted scan is
levelled automatically. Counts: all 46 stand; 26 stand upside down; 19 lie on
a side; 8 hold something on top; 5 review flags. Tests:
`affdet/tests/test_face_labels.py` (primitives: round vs hexagonal vs domed,
wide 84-gon, near-flat cap, grasp widths). Details and caveats are in
`docs/objaverse/README.md`, which also covers four shared-mesh pairs that must
go in the same split.

### VLM pilot (`scripts/vlm_face_pilot.py`)
GPT-6-sol, 20 objects × 3, one face-labelled 4-view image each; $0.40.
- Handles are perfect: both cranks found, with the right face.
- Materials are the least consistent (14–15/20).
- On near-metric questions it agrees with geometry only 75–85 %, and the
  errors are size-blind (it says 2 cm tops can hold a coaster).
- It resolves the smooth-shaded review flags via `body_cross_section`.

This confirms "numbers from geometry, VLM for non-metric". Results are in
`docs/objaverse/vlm_pilot_gpt6sol.{md,csv}`; raw answers and images are in
`data/vlm_pilot/`.

## Research report
`reports/UTAMP grasp and support perception.md` + notes in `research_notes/`.
Demo track: geometric predicate generator over the .ttt members, diffed
against UTAMP-2-GT. Paper track: DINOv2 student for face-level predicates +
predicate-noise → plan-success study on the 127 configurations.

## Next steps
1. Apply the review rule: for flagged faces, drop side supportability when
   GPT-6-sol's majority `body_cross_section` is `round`. Then do a human pass
   over the leftovers and over the borderline `tip_deg` cases (soft labels?).
2. Members for the other categories: mugs, pans, trays and boards have handles
   and rims, so the labeller needs member boxes (body / handle / edge) before
   their grasp labels mean anything. Options: split by mesh part where artists
   modelled the handle separately, or ask the VLM for the handle face (reliable
   in the pilot) and cut the box there. The hull-centroid COM is wrong for
   hollow objects; use a shell-plus-content estimate.
3. Curate the other categories the same way (by eye, ~30–80 each); mugs,
   plates, trays and pans have plenty, cutting boards ~61.
4. Full VLM run on the non-metric questions (materials, handle, fragile,
   cross-section) for all curated objects, 3× each; under $15 in total.
5. Ask Pouya: UTAMP-2 code, `llm_prompts_affd.py`, the hand-made affordance
   table (ground truth to test the labeller on, especially their salt shaker),
   the 127 configurations, whether hf1/hf2 order matters, whether
   palm=`under` grasps should be listed, whether upside-down counts as
   supportable, their face naming (which axis is `front`/`left`), canLift and
   canSlide scope, and what the benchmark salt shaker's cap looks like.
6. Check GraspGen builds on the lab GPU; bin its grasps to faces and compare
   with the width rule.
7. Licences: check the ~60 non-commercial objects and the 5 without a licence;
   confirm the ObjaTHOR annotation licence before redistributing the joined CSV.
8. Supervisor meeting (Oct 14, 14:30): its outcome is still to be recorded here.
