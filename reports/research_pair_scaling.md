# Scaling pair labels beyond 22 objects

Question: how to label isP2PSupportive (A rests on B, heatmaps on A and B) for about 320 curated objects without paying for every ordered pair. Analysis scripts and raw output: `/tmp/claude-1001/-home-mascarpone-max-uibk-uibk-courses-Semester-6-affdet/189f72ac-8b11-494c-8525-de5068c48f27/scratchpad/pair_scaling.py`, `pair_scaling2.py`, `pair_scaling3.py` and `*_out.txt` in the same folder (scratchpad, not committed; copy them if you want to keep them).

## 1. Short answer

- **What the data say.** The pair statistics are mostly a property of the **base**. The top contributes a smaller part, and that part is predicted by a **unary** property of the top (which of its faces stand on a table) plus a few **probe** simulations. A top's category barely matters: 35 shakers spread as widely on a given base as 16 tops from 6 categories do.
- **What does not transfer.** The B-side **heatmap** cannot be copied from a neighbouring top (IoU 0.5 to 0.7). Scalar summaries are predictable; maps are not. Maps for pairs that are not simulated have to come from a learned model that sees both geometries, not from lookup.
- **Recommended plan.** Use a linear-cost plan: unary face tests for every object, K≈4 to 6 probe bases per top and about 4 probe tops per base, plus about 3n random pairs as training signal for the model. Hold out a set of objects with dense rows and columns for validation. That is about 3,950 pairs for n=320 instead of 102,080, roughly 3.9 %.

## 2. How the count grows

- **Ordered pairs.** With n objects there are n(n-1) ordered pairs (A on B and B on A differ). This is quadratic, not factorial: 22 objects give 462, 100 give 9,900, 320 give 102,080, 1,000 give 999,000.
- **Roles.** Restricting roles to |T| plausible tops times |B| plausible bases is still quadratic if both grow with n. Dropping shakers as bases at n=320 saves only about 15 %.
- **Multi-object scenes.** An ordered k-object stack has n!/(n-k)! choices, which is about n^k: polynomial for fixed k (cubic for 3-stacks, 32 million at n=320).
  - It becomes factorial only if k grows with n (one ordered tower of all n objects is n!).
  - Labelling every subset is 2^n.
  - A scene with m objects contains m(m-1) pairwise relations, so scenes do not reduce the number of relations to learn. They only package several of them into one render, which helps a VLM but not a simulator, where each placement trial still tests one pair.
- **Cost per pair.** Each pair label is itself a map over faces × placement cells. That is why one pair costs about 1,950 simulations at 32 cells (measured mean 1,954, median 1,518; about 0.46 core-hours) against 630 at 16 cells (scale-out mean, median 408, about 0.16 core-hours).
- **Full matrix at n=320.** About 47,000 core-hours at the 32-cell rate, or about 23,000 with the recent speed-ups.
- **VLM cost.** At the past rate ($4.73 for 8,249 pairs) the full 320-object matrix would cost about **$58**. For a VLM, quadratic cost is not the problem; label quality is. The VLM gives no margins or physical heatmaps, and the scale-blindness found earlier for canSupport applies here directly.

## 3. What others do (methods scan)

| Approach | Source | What it does | Relevance here |
|---|---|---|---|
| **No enumeration, sparse random trials** | O2O-Afford (Mo et al., CoRL 2021), https://arxiv.org/abs/2106.15087 | Picks a random scene and a random acting object (random size and orientation), samples categories equally, then simulates **one** random interaction point per trial (appendix D). Points outside the "possible" region (e.g. normals not facing up, for placement) are labelled failed without simulation. They collect "hundreds of thousands" of trials per task and require at least 20,000 positives; random sampling gives 7.4 % (pushing) to 41.2 % (placement) positives. 1,785 ShapeNet models. | They never build dense per-pair maps. The network learns the map from many sparse pairs. |
| **Adaptive sampling** | Where2Act (Mo et al., ICCV 2021), https://arxiv.org/abs/2101.02692 | Single object. Offline random sampling gives only 1 % positives for pulling. Online, they sample the next trial from the softmax of the network's own predicted scores and mix 50 % random with 50 % prediction-biased. Gripper orientations in the wrong hemisphere are marked negative without a trial. | Active sampling inside a pair (cells) and across pairs. Keep a random half to avoid tunnel vision. |
| **ID-based matrix completion with active learning** | Szedmak, Ugur, Piater (IROS 2014), https://iis.uibk.ac.at/public/papers/Szedmak-2014-IROS.pdf | 83 × 83 stack interactions, completed with ID-based "knowledge propagation" (recommender-style). From 1 % of the matrix it is significantly better than shape-feature classifiers; active learning overtakes random after about 6 % of the matrix. The paper names new objects as the open problem. A local summary quotes "85 % at 1 %"; that number is read off a figure and is **[unverified]**. | Same lab as the thesis. Directly comparable. |
| **Unary first** | Ugur, Szedmak, Piater (ICDL-EpiRob 2014), https://iis.uibk.ac.at/public/emre/papers/ICDL2014-Bootstrapping.pdf | Single-object affordance predictions are used as inputs to the paired-object predictor; this learns faster than raw shape features. The next pair is chosen by diversity in that affordance space. | Our strongest empirical finding (section 4) is the same idea. |
| **Two-tower / pairwise kernels** | Stock et al., Neural Computation 2018, https://arxiv.org/abs/1803.01575 | Kronecker and two-step kernel ridge regression (per-object representations with a pairwise head) unify matrix filtering and zero-shot pair prediction. They define Setting A (both objects seen, pair new), B and C (one object new) and D (both new). D is "the most difficult". | Report every result per setting. |
| **Pair-aware evaluation** | Park & Marcotte, Nature Methods 2012, https://pmc.ncbi.nlm.nih.gov/articles/PMC3531800 | Pair-input predictors look much better on test pairs whose members appear in training. | This is the same trap as the instance leakage found earlier in the project. Split by object, not by pair. |
| **Low-rank completion** | Soft-Impute (Mazumder, Hastie, Tibshirani, JMLR 2010), https://www.jmlr.org/papers/v11/mazumder10a.html | Nuclear-norm completion of a sparse matrix. | Section 4 tests it; it adds little beyond additive effects. |
| **Probe objects / gauge set** | Eigentaste (Goldberg et al., 2001), https://www2.eecs.berkeley.edu/Pubs/TechRpts/2000/3869.html; CUR decompositions (Mahoney & Drineas, PNAS 2009), https://pmc.ncbi.nlm.nih.gov/articles/PMC2630100; active-learning survey (Elahi, Ricci, Rubens, Comput. Sci. Rev. 2016), doi 10.1016/j.cosrev.2016.05.002 | Eigentaste has every new user rate a fixed "gauge set" of items. CUR picks actual columns (by leverage) to stand for the whole matrix. The survey separates non-personalised elicitation (the same items for everyone) from personalised elicitation. Survey details come from secondary sources **[partly unverified]**. | A fixed gauge set is exactly the K-probe idea. |
| **Correspondence transfer within a category** | R-NDF (Simeonov et al., CoRL 2022), https://arxiv.org/abs/2211.09786; DenseMatcher (Zhu et al., ICLR 2025), https://arxiv.org/abs/2412.05268 | R-NDF transfers object-object relations from 5 to 10 demonstrations to new instances via category-level descriptor fields. DenseMatcher transfers labels through 3D semantic correspondence, also across categories. | Suited to the **A-side** contact patch. Section 4 shows the B-side map is too top-specific to copy. |
| **Learned placement that generalises without enumerating pairs** | Jiang et al., IJRR 2012, https://arxiv.org/abs/1202.1694 | Learned placement on 98 objects and 40 areas: 98 % success for known objects, 82 % for new ones (abstract). How they sampled pairs: **[unverified]**. | Evidence that a geometry model can generalise without the full pair set. |

**Geometric pre-filters.** Both Mo papers prune at the **cell/orientation** level, not the pair level. Our pipeline already does the analogue for faces: a face that fails on a flat table is skipped.

## 4. Empirical check on our data

### Data

- The 16 × 18 matrix: 276 pairs at 32 cells, 2 to 3 instances per category.
- The scale-out: 35 shakers × 18 bases, 630 pairs at 16 cells.
- Targets: `b_supported_frac` (the share of B's grid that carries A) and `best_b_margin_mm`.

### Metric

On the 276 matrix the base mean alone reaches R² 0.75 (frac) and 0.85 (margin) on held-out pairs, so raw R² flatters every method. I therefore report **skill = 1 - SSE/SSE(base mean)** on identical held-out cells: the share of the remaining, top-specific variance that a method explains. 95 % intervals come from resampling held-out objects (1,000 draws).

Whole-object hold-outs are strict: when an object is held out as a top, its column as a base is removed too.

### Results

| Test | Result |
|---|---|
| Variance explained in-sample (frac / margin) | Base category 0.76 / 0.80. **Top category 0.04 / 0.01.** Top-category × base-category cells 0.86 / 0.87. Additive instance effects 0.87 / 0.89 |
| Within vs between category | SD of frac per base: across **35 shakers 0.146**, across **16 tops from 6 categories 0.146** |
| Held-out random pairs (Setting A), frac | Additive top+base skill 0.33. Adding rank 1 to 2 gives 0.36 (absolute R² 0.84) |
| Held-out random pairs (Setting A), margin | Additive 0.07. **Rank 1 gives 0.34** |
| Held-out random pairs (Setting A), shakers | Frac: additive 0.64, rank 2 0.69. Margin: 0.46, rank 2 0.52 |
| New top, mixed categories, frac | Same-category mean 0.15 [-0.16, 0.37]. 4 random probe bases 0.23 to 0.26. 4 greedy probes 0.33 to 0.38 [0.23, 0.51]. 6 greedy probes 0.34 to 0.40 |
| New top, mixed categories, margin | Same-category mean 0.24 [-0.03, 0.56]. Probes −0.17 to 0.19, most intervals contain 0 |
| New top, bounding-box features (cold) | 276 frac: ridge R² 0.66, kNN 0.74, against 0.75 for the base mean, so **no gain**. Shakers: kNN skill about 0.13 |
| New shaker, frac | Category mean 0 (by construction). **Face pattern from the table test 0.50 to 0.56 [0.36, 0.77].** Random probes: K=1 0.43, K=4 0.58. Greedy probes: K=4 0.66, K=6 0.69 to 0.71. **Face pattern + 1 probe 0.64**; + 4 probes 0.68 |
| New shaker, margin | Face pattern 0.04 to 0.16. Probes K=4 0.37 to 0.50 (wide intervals) |
| New base, frac | **Same-category mean 0.86 [0.75, 0.93].** Probe tops: K=1 0.60, K=3 0.77 to 0.81, K=6 0.81 to 0.86 |
| Copying the B-side map (b_any IoU), mixed tops | Random other top 0.45. Consensus of other tops 0.50. Nearest by extents 0.52. Same category 0.52. Oracle (nearest by true frac) 0.53 |
| Copying the B-side map (b_any IoU), shakers | Random 0.53. Consensus 0.62. Nearest by extents 0.69. Face pattern then extents 0.73. Nearest on 3 probe bases 0.73 |
| A-side faces | 945 of 1,656 face slots removed by the unary table test. Of the remaining 711, **8 (1.1 %)** fail on a particular base |

### Reading the results

1. **The base carries most of the label.** A new base is predicted well from its category (skill 0.86) or from 3 to 4 probe tops (about 0.8). Caveat: only 2 to 3 bases per category, all fairly typical; a strange tray or a pan with a high rim would not be covered.
2. **Top category is the wrong unit.** Top category explains 4 % of the variance, and one category (shakers) spans the full range. The pair labels split shakers by **which faces stand**. For example, on a flat board frac is 0.852 for every upright-only shaker and 0.994 for those that can also lie down. That is a unary property costing a handful of table simulations per object. The 2 to 3 per category sampling under-represents within-category variety, as suspected, but in the sense of face behaviour, not size.
3. **Probes help, with diminishing returns.**
   - Skill climbs fast up to K≈3 to 4 and then flattens.
   - Probes chosen greedily on the training tops beat random probes by about 0.05 to 0.10.
   - For shakers, unary prior + 1 probe (0.64) is about as good as the ceiling of additive and low-rank models with **all** other entries of the row known (0.64 to 0.69).
   - About a third of the top-specific variance (frac) is therefore true pair interaction that none of these models capture. For mixed tops it is about two thirds.
4. **Margin is harder than frac.** Additive effects explain nearly nothing of the top-specific margin across mixed categories, and probes are not reliably better than the base mean. Rank-1 structure helps when the row is otherwise known (0.07 to 0.34), so there is an interaction (a top that is robust on flat boards is not necessarily robust on rims), but a few probes cannot pin it down.
5. **Maps do not transfer.** Even an oracle neighbour gives IoU 0.53 on mixed tops. Within shakers it reaches 0.73. Heatmaps for pairs that are not simulated must come from a model conditioned on both shapes.
6. **Pair-level geometric pruning has little to cut.** Only 0.7 % (matrix) and 1.6 % (shakers) of pairs have frac = 0. Almost every top fits somewhere on almost every base in this pool, so a "trivially infeasible" filter would remove almost nothing. "Trivially feasible" (flat board, upright face) is real: the label then depends only on the face pattern. Bounding boxes added nothing in the cold-start test.

### Honest limits

- 16 tops and 18 bases is small; the intervals above are wide and most comparisons at K≥4 overlap.
- The greedy-probe rows score fewer, easier cells; the skill metric corrects for this, raw RMSE does not.
- The shaker results are one category with mostly upright, convex objects.
- The scale-out ran at 16 cells, the matrix at 32, so the two frac scales differ slightly.
- All targets are scalar summaries; nothing here validates predicted heatmaps.
- The A-side "99 % unary" result is partly built in, because faces that fail on a table are never tried on concave bases.

## 5. Proposed labelling plan for about 320 objects

Costs use the measured 16-cell rate (about 0.16 core-hours per pair, roughly half with the recent speed-ups if they carry over) for probes and random pairs, and the 32-cell rate (0.46) for validation.

| Rank | Step | Pairs at n≈320 | Core-hours (16-cell rate) | Benefit |
|---|---|---|---|---|
| 1 | **Unary pass for every object**: per-face table test (already exists: `face_supports`), plus the grasp and slide maps | 0 pairs, about 6 face tests per object | < 10 (estimate) | Fixes the A-side face set (99 % of slots). Best single predictor of a new top (skill 0.5 to 0.56). Linear in n |
| 2 | **Probe bases per top** (K=4 to 6, greedy from the 18): at least one flat board, one tray, one rim (mug), one bowl, one pan/wok; re-select with the CUR/greedy procedure once more bases exist | about 300 × 5 = 1,500 | about 240 | Skill 0.6 to 0.7 on frac for shakers, about 0.35 for mixed tops |
| 3 | **Probe tops per base** (K≈4: an upright-only shaker, a shaker that can lie down, a mug, a pan) | about 250 × 4 = 1,000 | about 160 | Skill about 0.8 for a new base |
| 4 | **Random pairs for the model** (about 3n), stratified by base type × top face pattern. Half of later batches chosen by disagreement between the factor model and the learned model (Where2Act-style 50/50) | about 1,000 | about 160 | The pair-interaction residual that probes cannot reach, and the training signal for heatmaps |
| 5 | **Validation, held out by object (Setting D)**: 10 to 12 objects (tops and bases, spread over categories and face patterns), each with dense rows and columns against about 40 partners at 32 cells | about 450 | about 200 | The only honest test of everything above |
|  | Total | about 3,950 (3.9 % of 102,080) | about 760 (vs about 47,000 for the full matrix at 0.46) | |

### How the pieces fit

- **The model.** Train the thesis model as a pairwise network on per-object geometry embeddings (the two-tower form, as in Stock et al.). Supervise it with the sparse pairs from steps 2 to 4, with the probe statistics and unary face verdicts as auxiliary inputs or targets. The matrix factor model (base effect + unary prior + probe offset) is a **cheap baseline and an active-learning critic**, not the final labeller.
- **Cell budget.** For heatmap supervision, consider O2O-style sparse cells (e.g. 100 to 200 random cells per pair over 5 to 10 times more pairs) instead of dense 600 to 2,000-cell maps on few pairs. Our result that maps are pair-specific argues for **pair diversity over per-pair density**. This is a hypothesis; test it on the validation set.

### What this loses

- Simulated (exact) heatmaps for about 96 % of pairs; they become predictions.
- About a third (shakers) to two thirds (mixed tops) of the top-specific frac variance is not captured by the factor model. The learned model has to recover it from geometry, and the validation set measures whether it does.
- Margins across categories are poorly predicted by any cheap scheme.
- New base or top categories outside the probe set's span (bottles, boxes, plates with deep wells) need new probes. Add one probe per genuinely new support type.

### Validation and decision rules

- Report results separately for Setting A, B/C and D (Park & Marcotte; Stock et al.), never pooled.
- Report skill against the base-mean baseline, not raw R².
- Stop adding probes when the validation skill from K to K+1 rises by less than about 0.03.
- Check the 16-cell vs 32-cell agreement on the validation pairs before trusting 16-cell maps as heatmap targets.

### VLM

- **Use:** cheap semantic role filters ("would anyone put a pan on a shaker"), and possibly a coarse yes/no on rim bases.
- **Before any use:** score it against the 906 simulated pairs we already have (276 + 630, cost under $1).
- **Not for:** margins or heatmaps.

## 6. Ranked summary

| Rank | Idea | Verdict |
|---|---|---|
| 1 | Unary face test + probes | Measured, cheap, linear |
| 2 | Probe tops for new bases | Bases are category-consistent, so this is cheap |
| 3 | Sparse random pairs + learned two-tower model, with object-held-out validation | Needed anyway for heatmaps |
| 4 | Active pair selection | Helps only once a model exists (Szedmak: after about 6 % of the matrix) |
| 5 | Low-rank completion beyond additive effects | Small gain for frac. Useful for margin only when the row is well observed |
| 6 | Category-level pairing for tops | Does not work in our data (top category explains 4 %) |
| 7 | Bounding-box pre-filters or regressors | No measurable gain, and almost nothing infeasible to prune |
| 8 | Correspondence transfer of B-side maps | IoU 0.5 to 0.7 is not good enough. A-side transfer is unnecessary because the face test is cheap |
