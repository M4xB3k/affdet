# Spatially resolved affordance heatmaps from VLMs and foundation models

Research survey, 18 Oct 2026. Question: can VLMs (alone or with other foundation models) replace or cheapen the MuJoCo per-vertex maps for isP2PSupportive, canGrasp, canLift and canSlide, with a small set of simulated pairs used only to measure and tune?

Short answer: **VLMs give region-level semantics ("the rim", "the flat base", "the handle") reliably, and point-level output with roughly part-level accuracy. Nothing published shows VLM-only maps that match physics-derived support maps for two-object relations.** The strongest pipeline for us puts the VLM on top of 3D part segments of rendered meshes, with geometry doing the fine resolution, and uses physics for a sparse check rather than a dense sweep.

Note: the 14 Oct supervisor brief says labels "must come from physics, not from VLM guesses". On that brief, a VLM pipeline is a cheap prior or proposal generator, not a replacement.

---

## 1. Technique families

### 1.1 Pointing and keypoint output

| System | Output | Reported accuracy | Availability |
|---|---|---|---|
| **Molmo** (Deitke et al., 2024) [arXiv 2409.17146](https://arxiv.org/abs/2409.17146) | Points in image coordinates, trained on PixMo-Points | Pointing F1 75.2 (72B) and 74.3 (7B-D) on a 493-item hand-verified set. F1 drops to 58.1 when the crop count changes between training and test. | Apache-2.0 weights on [Hugging Face](https://huggingface.co/allenai/Molmo-7B-D-0924); runs locally |
| **RoboPoint** (Yuan et al., CoRL 2024) [arXiv 2406.10721](https://arxiv.org/abs/2406.10721) | 1 to ~50 points per query. Trained on synthetic data for "free space relative to object X". | Where2Place (share of points inside the GT mask): **46.8%** vs GPT-4o 29.1%, SpaceLLaVA 11.8%, Qwen-VL 10.5%. The authors note the number of points cannot be controlled. | Open (Vicuna-13B + CLIP ViT-L/14). Licence not stated in the paper [check project page](https://robo-point.github.io). |
| **Gemini Robotics-ER** 1.5 tech report [arXiv 2510.03342](https://arxiv.org/abs/2510.03342); current API is ER 2 preview ([docs](https://ai.google.dev/gemini-api/docs/robotics-overview)) | `[y, x]` points normalised to 0–1000, labelled JSON, plus boxes and trajectories | Report table: Point-Bench-Affordance ~70.9, RefSpatial ~48.5 (ER 1.5 with thinking) [column mapping garbled in extraction, check the PDF]. ER 1.6 is deprecated (31 Aug 2026), replaced by ER 2. | API only. ER 2 preview costs $1 per 1M input tokens and $5 per 1M output tokens until 31 Dec 2026, then $2 / $10 ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). |
| **MOKA** (RSS 2024) [arXiv 2403.03174](https://arxiv.org/abs/2403.03174), **PIVOT** [arXiv 2402.07872](https://arxiv.org/abs/2402.07872) | The VLM picks among marked candidate points (MOKA) or refines sampled points over iterations (PIVOT) | Task success rates only, with no map metric | Prompting recipes, so any VLM works |

**Resolution.** The coordinate format sets a floor of 1/1000 of the image. The practical floor is much coarser. At Where2Place's best (~47%), more than half of RoboPoint's points miss the target region. On a 512 px render of a 20 cm mug, a typical miss is still within one part (centimetres).

**Turning points into a heatmap.**
1. Sample N points per view at temperature > 0, over K views.
2. Raycast each point onto the mesh.
3. Splat a geodesic Gaussian (σ about 1–2 cm).
4. Divide by how often each vertex was visible across views.

This yields a density map, not a probability of success. Pointing models also suffer **mode collapse**: they return the same 1–3 "canonical" points (for example the mug handle centre), while physics gives a ring (the whole rim) or a large patch.

**Direct evidence for two-object relations.** O³Afford (CoRL 2025, [arXiv 2509.06233](https://arxiv.org/abs/2509.06233)) uses RoboPoint as a baseline for object-to-object point maps:

| Method | IoU |
|---|---|
| RoboPoint | 11.84 |
| O2O-Afford | 14.31 |
| IAGNet | 16.89 |
| O³Afford (trained, one-shot) | 26.19 |

O³Afford's labels are self-annotated. Even so, the result shows that off-the-shelf pointing is the weakest option for paired maps.

### 1.2 Set-of-Mark: pre-segment, then let the VLM choose

- **Set-of-Mark prompting** (Yang et al., 2023, [arXiv 2310.11441](https://arxiv.org/abs/2310.11441)) is the generic recipe: overlay numbered regions on the image and ask for region IDs.
- **ReKep** (CoRL 2024, [arXiv 2409.01652](https://arxiv.org/abs/2409.01652)): DINOv2 patch features inside SAM masks, k-means (k=5) gives keypoint candidates, and GPT-4o writes **relational constraints between keypoints on two objects**. This is the closest SoM precedent for inter-object relations, but it outputs keypoints, not maps.
- **CoPa** ([arXiv 2403.08248](https://arxiv.org/abs/2403.08248)): coarse-to-fine VLM grounding of the grasp part, then part-to-part geometric constraints. 63% success over 10 real tasks.
- **OmniManip** ([arXiv 2501.03841](https://arxiv.org/abs/2501.03841)): VLM-chosen object-centric interaction primitives.
- **UAD** (Tang et al., CVPR 2025, [arXiv 2506.09284](https://arxiv.org/abs/2506.09284)):
  - Renders objects from many views and fuses DINOv2 features in 3D.
  - Clusters them into regions and has GPT-4o name task–region pairs.
  - Distils the result into a FiLM-conditioned decoder on frozen DINOv2.
  - Our project summary (`tools/docs/papers/knowledge/tang_2025_uad.md`) records AUC ≥ 0.92 across generalisation splits [not re-verified against the paper]. Single-object only.

**3D part segmenters that provide the "marks" on meshes:**

| Segmenter | Method | Reported numbers | Notes |
|---|---|---|---|
| **PartField** (NVIDIA, ICCV 2025) [arXiv 2504.11451](https://arxiv.org/abs/2504.11451), [project](https://research.nvidia.com/labs/toronto-ai/partfield-release) | Feed-forward 3D feature field; hierarchical clustering gives parts at any granularity | "Up to 20% more accurate", often orders of magnitude faster than class-agnostic baselines | Fits our use best: deterministic, per-vertex, granularity is a knob, and features stay consistent across shapes |
| **SAMPart3D** [arXiv 2411.07184](https://arxiv.org/abs/2411.07184) | Scale-conditioned part features; a VLM names parts from multi-view renders | Third-party summary: 53.7% class-agnostic mIoU on PartObjaverse-Tiny [unverified] | Built on Objaverse-style assets |
| **Find3D** (ICCV 2025) [arXiv 2411.13550](https://arxiv.org/abs/2411.13550) | Text-queried part segmentation; data engine labels with SAM + Gemini | "260% mIoU improvement", 6–300× faster | Open-world queries ("rim", "base") |
| **PartSLIP** (CVPR 2023) [arXiv 2212.01558](https://arxiv.org/abs/2212.01558) | GLIP on multi-view renders, lifted to 3D | Strong zero-shot results on PartNet [numbers not checked] | Older; superseded by the three above |

**Resolution.** Segmentation fixes the resolution. With 8–30 parts per kitchen object, that is 1–5 cm patches. The VLM only has to make a discrete choice, which is the task it is best at. The main failure is a **granularity mismatch**: the physically relevant patch (the outer 3 mm lip of a plate, or an annulus on a pan base) is not a separate segment.

### 1.3 Models trained to output affordance masks

**2D models** (single object, human-action vocabulary):

| Model | Approach | Reported results |
|---|---|---|
| **AffordanceLLM** (CVPRW 2024) [arXiv 2401.06341](https://arxiv.org/abs/2401.06341) | LLaVA with a mask decoder | Evaluated on AGD20K |
| **LISA** (reasoning segmentation) [arXiv 2308.00692](https://arxiv.org/abs/2308.00692) | Base model for GLOVER and GLOVER++ | — |
| **GLOVER** [arXiv 2411.12286](https://arxiv.org/abs/2411.12286) | Fine-tuned for task-oriented grasping | 86.0% part identification, 76.3% grasping, 30 real scenes |
| **GLOVER++** [arXiv 2505.11865](https://arxiv.org/abs/2505.11865) | Trained on HOVA-500K; Gaussian heatmaps around actionable points | KLD 3.41, SIM 0.141 |
| **ManipVQA** [arXiv 2403.11289](https://arxiv.org/abs/2403.11289) | Affordance grounding and physical properties | — |
| **OVAL-Prompt** [arXiv 2404.11000](https://arxiv.org/abs/2404.11000) | VLPart detection, then an LLM picks the part, then a mask | "Competitive" with supervised models on UMD |

**Trajectory-level affordance:** RAM [arXiv 2407.04689](https://arxiv.org/abs/2407.04689) and AffordDP [arXiv 2412.03142](https://arxiv.org/abs/2412.03142) output a 3D contact point plus direction, not maps.

**3D models** (per-point on point clouds, single object):

| Model | Notes |
|---|---|
| **OpenAD** (IROS 2023) [arXiv 2303.02401](https://arxiv.org/abs/2303.02401) | ~100 ms inference |
| **LASO** (CVPR 2024) [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.html) | 19,751 point–question pairs; code at [github.com/yl3800/LASO](https://github.com/yl3800/LASO) |
| **3D-AffordanceLLM** (ICLR 2025) [arXiv 2502.20041](https://arxiv.org/abs/2502.20041) | ~8% mIoU gain |
| **SeqAfford** (CVPR 2025) [arXiv 2412.01550](https://arxiv.org/abs/2412.01550) | 180K instruction–point-cloud pairs |
| **Affogato / Espresso-3D** [arXiv 2506.12009](https://arxiv.org/abs/2506.12009) | 750K annotations on 150K 3D assets, auto-labelled with Gemma3 + Molmo + SAM. A follow-up reports **19.1 aIoU** on Affogato ([arXiv 2601.09211](https://arxiv.org/abs/2601.09211)). |

**Assessment.** All of these predict **single-object, action-word affordances** (grasp, pour, support as an object category). None conditions on a second object's geometry. Their absolute accuracy is modest: IoU or aIoU in the 20s is typical. They are usable as features or priors, not as isP2PSupportive labels.

### 1.4 Image generation and editing ("paint the heatmap", "show A placed on B")

- **Hand–object and human–object inpainting:**
  - Affordance Diffusion (Ye et al., CVPR 2023, [arXiv 2303.12538](https://arxiv.org/abs/2303.12538)).
  - ComA, "Beyond the Contact" (ECCV 2024, [arXiv 2401.12978](https://arxiv.org/abs/2401.12978)). It explicitly warns that naïve inpainting **changes the object itself**, so "the generated affordance is simply a hallucination from a different object". It needs an adaptive mask to prevent this.
  - InteractAnything (CVPR 2025, [arXiv 2505.24315](https://arxiv.org/abs/2505.24315)) reads contact off inpainted humans.
- **Physical plausibility of edits:** PICABench ([arXiv 2510.17681](https://arxiv.org/abs/2510.17681)) finds physical realism "remains a challenging problem for most mainstream models", across optics and mechanics.
- **I found no paper that validates generated "object placed on object" images, or painted heatmaps, against physical contact for object–object support.**

My verdict: unreliable for labels. The editor will cheerfully balance a shaker on a mug rim and redraw the mug so that it works.

Gemini 2.5 Flash Image cost $0.039 per image but shut down on 2 Oct 2026 ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). Use this family only for qualitative sanity checks.

### 1.5 Multi-view rendering, per-view prediction, back-projection

Precedents:
- **SATR** (ICCV 2023, [arXiv 2304.04909](https://arxiv.org/abs/2304.04909)): multi-view GLIP boxes are turned into mesh face labels with topology-aware reweighting.
- **PartSLIP** and **SAMPart3D**: multi-view 2D predictions lifted to 3D.
- **UAD**: multi-view DINOv2 features fused in 3D before clustering.

Recommended aggregation for a per-vertex score:

    s(x) = Σ_v w_v(x) · p_v(π_v(x))  /  Σ_v w_v(x)
    where w_v(x) = visible_v(x) · max(0, n(x) · d_v)

- `visible_v(x)` comes from a z-buffer test; `n(x) · d_v` is the cosine between the vertex normal and the view direction.
- Smooth geodesically, or average over PartField segments.
- **Views must include a view from below and grazing views.** Support contacts on A are usually its underside, which normal "robot" viewpoints never see.
- Prior work uses about 10–20 views. That count is a heuristic [not a verified optimum], and the pilot should test K ∈ {6, 12, 24}.

### 1.6 Object–object contact from foundation models

- **Simulation-trained:** O2O-Afford ([arXiv 2106.15087](https://arxiv.org/abs/2106.15087)), with placement F-score 91.4% and stacking 91.7%.
- **One-shot DINOv2 + point cloud:** O³Afford, which predicts both maps jointly.
- **"Where to place":**
  - RoboPoint / Where2Place: free-space points.
  - SpatialVLM ([arXiv 2401.12168](https://arxiv.org/abs/2401.12168)): metric spatial question answering.
  - **AnyPlace** (CoRL 2025, [arXiv 2502.04531](https://arxiv.org/abs/2502.04531)): Molmo proposes a coarse region, then a model trained on synthetic data predicts precise poses (76% real success and coverage). This is exactly the "VLM coarse, physics-trained model fine" split.
- **Stability judgements are weak in current VLMs:**
  - PAC Bench (NeurIPS 2025 D&B, [arXiv 2506.23725](https://arxiv.org/abs/2506.23725)): significant failures on complex affordances and physical constraints, including stackability.
  - [arXiv 2508.10770](https://arxiv.org/abs/2508.10770): mainstream VLMs "perform inadequately" on spatio-physical reasoning; fine-tuning helps but generalises poorly.
- **I found no VLM/LLM-based method that outputs a paired contact heatmap (where on A touches where on B).** This is an open gap. O³Afford is the closest, and it is a trained DINOv2 model, not a VLM.

---

## 2. (a) Ranked shortlist for our case

### 1. 3D-part Set-of-Mark with true-scale paired renders, plus geometric refinement

**Pipeline:**
1. Run PartField on A and B, keeping two granularities (about 8 and about 25 parts).
2. Render A and B **together in one frame at true relative scale**, with a ruler or grid backdrop. This fixes the zoomed-tile scale loss behind our F1 = 0.785 on support. Use 6–12 views including a view from below, with numbered parts on both objects.
3. Ask: "Which parts of A could rest on which parts of B? Give (A-part, B-part, score 0–10)."
4. Scatter the scores to the parts' vertices.
5. Refine *within* each part with cheap geometry: upward- or downward-facing normals, local flatness, and whether the A-patch support polygon contains A's projected centre of mass.

**For canGrasp:** pick parts, then filter by local thickness against the 7 cm GRIPPER_WIDTH.

| Aspect | Expectation |
|---|---|
| Resolution | Part level (1–5 cm), sharpened by the geometric terms |
| Main failures | Segments that do not isolate thin rims or lips; overconfident "canonical use" answers (cup on saucer: yes; board on mug rim: the VLM may say no, the sim says yes) |

### 2. Multi-sample pointing ensemble, back-projected

**Pipeline:**
1. Prompt Gemini Robotics-ER 2 (API) and Molmo-7B (local, free, many samples) on the same paired renders: "point to where A touches B when A rests on B", for both A and B.
2. Back-project, splat and visibility-normalise as in §1.5.
3. Use self-consistency across samples and models as a confidence signal.

| Aspect | Expectation |
|---|---|
| Resolution | Sub-part in principle (~1 cm); in practice clustered at 1–3 canonical spots |
| Main failures | Mode collapse (a ring becomes a dot); hidden undersides; view-dependence; poor recall on large flat support regions |

The published object-to-object evidence (RoboPoint IoU 11.8 in O³Afford) argues against relying on this alone.

### 3. UAD/O³Afford-style distillation, conditioned on the partner

**Pipeline:**
1. Fuse multi-view DINOv2 (and PartField) features per vertex.
2. Take labels from pipeline 1 (cheap, many pairs) plus the existing simulated pairs (expensive, few).
3. Train a small pair-conditioned per-vertex head (cross-attention between A's and B's vertex features, as in O³Afford).

**Expected resolution:** per-vertex.

**Main failure:** it inherits the VLM's systematic biases unless the simulated subset dominates the loss on physics-critical cases (rims, narrow bases). Our E3 finding is a direct warning: student errors were 3.4× enriched in the VLM's error set.

**Cross-cutting recommendation: VLM proposes, physics verifies.** Use pipeline 1 or 2 to propose the top-k candidate contact regions and poses. Run MuJoCo only there, perhaps 50–200 drops instead of about 2,000 per pair, as in AnyPlace's coarse-then-fine split. This keeps labels physics-exact, as the supervisor requires, and cuts cost by about 10× [estimate].

---

## 3. (b) Evaluating against the simulator maps

Let `y(x)` be the per-vertex sim score (survival rate) and `s(x)` the VLM score.

**Ranking and threshold-free metrics:**
- **Area-weighted AUC-ROC** of `s` against `y ≥ τ`, for τ ∈ {0.5, 0.9}. Weight vertices by their Voronoi area so that dense mesh regions do not dominate.
- **Area-weighted Spearman ρ** between `s` and `y`.

**Region metrics:**
- **Top-k% IoU**: IoU between the top-k% area of `s` and of `y`, for k = 5, 10 and 20.
- **SIM and KLD** after normalising both maps to distributions. These are the standard AGD20K/GLOVER metrics, which makes results comparable with the literature.

**Point metrics** (pointing pipeline):
- **Hit rate**: the share of points landing on vertices with `y ≥ τ`, as in Where2Place.
- **Geodesic error** from the argmax to the nearest positive vertex.

**Pair-specific checks:**
- **Partner sensitivity**: correlation between A's map with partner B₁ and A's map with partner B₂. The sim says the mug-board and mug-shaker maps differ. A VLM whose maps barely change across partners is not doing pairwise reasoning.
- **Joint consistency**: the share of top (A-part, B-part) pairs whose patches actually co-occur in successful sim placements.
- **Predicate-level F1** after thresholding the maximum, to link back to the existing E-series.

**Downstream check:** for each pair, take the top-1 VLM placement and run about 20 noisy sim drops. Report success rate against the sim's own top-1. This is the metric that matters for TAMP.

---

## 4. (c) Pilot (~40 pairs)

**Pair selection:** stratify 40 pairs from the 276/702 existing sim pair maps:
- 15 "easy flat" pairs (board on tray, plate on board).
- 15 "physics-critical" pairs (things on mug rims, shaker on pan edge, plate on bowl rim, narrow bases).
- 10 negatives or concavities (shaker into mug).

Add canGrasp maps for the roughly 15 distinct objects involved.

**Arms:**

| Arm | What runs | Calls |
|---|---|---|
| P1: SoM on PartField parts | Gemini ER 2 and gpt-5-mini; 8 views × 2 granularities × 3 samples | ~48 calls per pair → ~2,000 calls |
| P2: pointing | Gemini ER 2 (8 views × 5 samples) + Molmo-7B locally (8 × 20 samples) | ~1,600 API calls; Molmo costs GPU time only |
| P0: geometry-only baseline | Upward normals, flatness and the centre-of-mass test, no VLM | None |

The P0 baseline is essential. If the VLM does not beat it, the VLM adds nothing.

**Cost.** Each call is about 3–5k input tokens (4 images) plus about 1–2k output or thinking tokens:

| Model | Per call | ~3,600 calls |
|---|---|---|
| Gemini ER 2 ($1 / $5 per 1M tokens) | ~$0.01–0.015 | **~$40–55** |
| gpt-5-mini (scaled from our $4.73 / 8,249 pairs) | — | **~$3–10** [estimate; depends on image tokens and thinking] |

Rendering and PartField run locally, with negligible cost. Effort is about 2–3 days to build and 1 day to evaluate.

**Decision rule:** continue only if P1 or P2 beats P0 on area-weighted AUC and top-10% IoU on the *physics-critical* stratum, not just the easy one.

---

## 5. (d) Honest assessment

**Where VLMs will do fine:**
- canGrasp at part level (handles, rims, the body of a bottle).
- Large flat-on-flat support (board on tray).
- Which face is the bottom.
- Naming and choosing among regions.

This matches the evidence: GLOVER reaches 86% part identification, and AnyPlace and Molmo give useful coarse regions.

**Where they will most likely fail against physics-derived support maps:**
1. **Thin rims and lips.** A board on a mug rim is stable, a shaker is not. The difference is millimetres of overlap and the partner's footprint. Pointing collapses the rim ring to a dot; part segments often merge the rim with the wall.
2. **Balance on narrow features.** Centre of mass inside the support polygon, tipping margins, a pan's long handle shifting its centre of mass. Current VLMs are documented as weak on stability (PAC Bench; arXiv 2508.10770).
3. **Scale.** This is the cause of our F1 = 0.785. Same-frame true-scale renders help, but VLM metric estimates remain coarse.
4. **Concavities and "falls in" outcomes.** The geometry looks supportive, but the object drops into the cavity.
5. **Hidden surfaces.** The undersides that carry contact must be rendered explicitly.
6. **Calibration.** VLM 0–10 scores are not probabilities. Only rank metrics are meaningful without per-predicate calibration on sim data.
7. **Pairwise conditioning.** No published VLM method outputs paired contact maps. Expect maps for A that barely change with B.

**Bottom line:**
- VLM-only maps will plausibly reach part-level agreement (AUC well above chance) on easy support and grasp.
- I do not expect them to match physics on rims, narrow bases or concave partners, which is precisely where TAMP needs exact labels.
- The defensible design is: VLM plus part segmentation proposes regions; geometric priors sharpen them; MuJoCo verifies a sparse set; a DINOv2 per-vertex model is distilled from the verified labels.
- This keeps the supervisor's "labels from physics" requirement and uses the VLM where it is strong, to cut simulation cost.
