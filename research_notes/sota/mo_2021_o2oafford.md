# O2O-Afford: Annotation-Free Large-Scale Object-Object Affordance Learning
**Authors**: Kaichun Mo, Yuzhe Qin, Fanbo Xiang, Hao Su, Leonidas Guibas
**Year / Venue**: 2021, CoRL 2021 (arXiv v2, 25 Oct 2021)
**Cite key**: mo_2021_o2oafford
**Source read**: https://arxiv.org/abs/2106.15087 (abstract page); https://arxiv.org/pdf/2106.15087 (full PDF including appendices A to G, text extracted locally and read in full)

## In one paragraph
O2O-Afford learns where an "acting" object can be successfully put, fitted or pushed relative to a scene, using physics simulation instead of human labels. For a given task (placement, fitting, pushing, stacking), the network sees a partial scan of the scene and a complete point cloud of the acting object, and outputs a per-point success heatmap over the scene. The labels come from SAPIEN trials: drop the acting object at a point and check whether it stays stable. The core idea is the "object-kernel point convolution": the acting object point cloud is moved over a set of seed points in the scene, scene features are interpolated at the object's points, and a PointNet turns this into a score for that seed. This is the closest published relative of our pair model and of our support test.

## Input and output
- Scene S: partial scan with n = 10,000 points, taken by FPS from a back-projected depth image (n x 3, xyz only).
- Acting object O: complete point cloud with m = 1,000 points (m x 3), plus a centre c, a 1-DoF yaw q in [0, 2 pi) about the up axis, and an isotropic scale. Both clouds are in the camera frame (z up) and are zero-centred before the networks.
- Output: a per-point affordance heatmap A in [0, 1] over the n scene points (the formulation line writes the dimension as m, which is a typo for n as far as I can tell). One network is trained per task. A single acting-object pose and size is conditioned on at a time.
- Success threshold for binary decisions: 0.5.

## Architecture
- Two separate segmentation-style PointNet++ encoders (no shared weights), single-scale grouping, ReLU and BatchNorm:
  - Enc_scene: 4 set-abstraction layers at 1024, 256, 64, 16 points (MLP sizes [3,32,32,64], [64,64,64,128], [128,128,128,256], [256,256,256,512]); 4 feature-propagation layers; final linear layer gives a 128-d feature per scene point (F_S in R^(n x 128)).
  - Enc_object: 3 set-abstraction layers at 512, 128 and 1 point (the last one is the global feature); 3 propagation layers; linear layers give a 128-d feature per object point (F_O in R^(m x 128)) and a 128-d global feature F_g.
- Object-kernel point convolution:
  1. Pick k = 1,000 seed points on the scene by furthest point sampling.
  2. For each seed p_i, translate the object cloud so it sits at p_i. For every object point o_j, interpolate the scene feature at o_j + p_i from its t = 3 nearest scene points with inverse-distance weights.
  3. Concatenate the interpolated scene feature (128) with the object feature F_O (128) per object point, giving m x 256.
  4. A PointNet with MLP [256, 128, 128, 128] and max-pool over the m object points gives a 128-d seed feature F_SO|p_i.
- Decoder Dec_critic: concatenate F_SO|p_i (128), the local scene feature at the seed (128) and the global object feature F_g (128) = 384, then an MLP [384, 128, 1] (Leaky ReLU, no BatchNorm). The main text says sigmoid, the appendix says "no activation function" on the last layer. The loss is BCE, so a sigmoid is implied.
- The k seed scores are interpolated back to all n points by inverse-distance weighting.
- All feature dimensions f1 = f2 = f3 = f_g = 128. Parameter count: not verified in the paper text.

## Simulation and labels
- Simulator: SAPIEN (PhysX), 500 fps, solver iterations 20, gravity 9.81, static and dynamic friction 4.0, restitution 0.01. All objects have uniform density (stated as a limitation).
- Data: 1,785 ShapeNet/PartNet models in 18 categories. 12 training categories (867 train shapes, 281 test shapes) and 6 test-only categories (637 shapes). "Heavy" classes (cabinet, microwave, table, refrigerator, safe, washing machine) serve as static scene objects. "Item" classes (basket, bottle, bowl, box, can, pot, mug, trash can, bucket, dispenser, jar, kettle) are acting objects.
- Each trial: choose a point p on the scene; start the acting object with its centre at c = p + r(O), a task-specific offset; run a straight hard-coded trajectory shorter than 0.1 unit (gravity direction for placement, fitting, stacking; forward for pushing).
- Placement: scene is a table with 0 to 15 small items. The acting object starts 0.01 above p (offset s_z/2 + 0.01). Success if (1) no collision at start, (2) it stays on the countertop, (3) it lands without a large change of orientation. Points whose normal is not near "up" are marked failures without simulating.
- Fitting adds: the drawer or door must close with the object inside. Pushing needs a large enough motion, direction within 30 degrees of forward, and no toppling. Stacking: the acting object has no large pose change when the scene object is dropped on it.
- The stacking description is terse: the text says "only points on the ground are applicable" and the scene object "is initialized at the final stably stacking pose". Which object the heatmap lives on in stacking is not clear from the text I read (I did not read the figure itself).
- Sampling: random positions among the "possible" region, equal sampling per category, equal positives and negatives per batch. Hundreds of thousands of trials per task, at least 20,000 positives per task. Positive rate under random sampling: 7.4 % for pushing up to 41.2 % for placement.

## Evaluation and headline results
- Metrics: F-score and average precision (AP) with threshold 0.5. How the test positions are drawn for these metrics is not described beyond the training procedure (not verified).
- Baselines: B-PosNor (position and ground-truth normal replace the scene feature), B-Bbox (6-d bounding box extents replace the acting-object geometry), B-3Branch (two PointNet++ branches plus seed position, no object-kernel correlation).
- Numbers (before slash: unseen shapes of training categories; after: unseen categories):
  - placement: F 81.4 / 90.0, AP 91.1 / 95.2 (B-Bbox 80.9 / 90.6 and 90.5 / 94.5);
  - fitting: F 73.6 / 80.3, AP 80.1 / 86.3 (B-Bbox 69.5 / 79.5);
  - pushing: F 35.5 / 40.3, AP 46.9 / 43.1;
  - stacking: F 89.6 / 87.5, AP 91.7 / 90.8 (B-3Branch 87.3 / 84.8).
- Ablation (remove the object-kernel feature): placement is unchanged or slightly better in F (82.2 / 91.3), but stacking drops (F 82.6 / 80.4) and fitting drops (68.0 / 78.3).
- Honest reading: for placement, a bounding box of the acting object plus scene geometry is almost as good as the full model. The pair feature matters mostly for stacking and fitting.
- Real world: qualitative predictions on iPad Pro scans, Replica, RBO and Google Scanned Objects, with YCB or Google objects as acting objects. No robot execution.

## How it differs from our setup
- Simulator and physics: SAPIEN with objects dropped 1 cm onto a surface, versus our MuJoCo drop of object A on object B. Both define support-like success as "stays stable". Their placement success also needs "no collision at start". To be confirmed for our drop test: do we reject initial interpenetration between A and B?
- Label definition: binary stability test per trial. We have the same for support, plus a second "level" label for grasp.
- Dense versus sampled: random positions with tuned positive rates and balanced batches, versus our exhaustive grid per object pair. Our setup gives a complete ground-truth heatmap per pair, theirs gives a sparse set of labelled points.
- Scale of pairs: they pair a random scene with a random acting object, from 1,785 shapes, a few hundred thousand trials. We simulate about 20,000 ordered object pairs with a full grid each, which is far denser per pair.
- Backbone and pair mechanism: two PointNet++ encoders and a geometric kernel that physically moves the acting cloud over the scene. We use one small EdgeConv encoder per object plus 2 cross-attention blocks in both directions, so our model can output heatmaps on both objects and has no explicit translation of one cloud over the other.
- Pose and size of the acting object: they feed one pose and one scale (yaw, scale) per example, and show the heatmap changes with size and orientation (their Fig. 6). Our pair variant must say clearly which pose of A is assumed in the support test.
- Input: partial scene scan (10,000 points) versus our complete clouds with normals (and separately one RGB-D view or TripoSR).
- Evaluation: F-score and AP at 0.5 versus our per-point AUC and best-point success.
- Worth copying: the bounding-box baseline (B-Bbox). If a model that only sees the extents of A and the geometry of B scores close to ours, that tells us how much our cross-attention really adds. Also copy: a held-out-category split, and the size and orientation sensitivity plot.

## Likely supervisor questions
1. Why cross-attention and not their object-kernel convolution? The kernel assumes you can slide a rigid copy of A onto B, which fixes the pose of A. Cross-attention is permutation-invariant and gives heatmaps on both objects, but it is a design choice we should justify with an ablation. We should also run their B-Bbox style baseline.
2. Is your support label the same as their placement or stacking? It is close to placement: drop A on a point of B and check A rests stably. Their threshold is "no big orientation change"; ours is "rests stably". We should state our numeric stability thresholds next to theirs.
3. Can your numbers be compared with their F-scores and AP? No. They report F-score and AP at threshold 0.5 on a sampled set of points, and F-score depends on the positive rate. We report per-point AUC on dense grids with held-out objects. The two sets of numbers cannot be compared directly.

## Not verified
- Total parameter count.
- How the evaluation points for F-score and AP are chosen at test time.
- Which object the stacking heatmap is defined on (I read the text, not the figure).
- Whether the final layer has a sigmoid (main text says yes, appendix says no activation).
- The number of simulated trials per task beyond "hundreds of thousands" and "at least 20,000 positives".

## Verification (second reader, 2026-11-16)
Checked against the extracted full text (arXiv 2106.15087): Table 1 (placement F 81.4 / 90.0 and AP 91.1 / 95.2, B-Bbox
80.9 / 90.6 and 90.5 / 94.5; fitting F 73.6 / 80.3, AP 80.1 / 86.3, B-Bbox 69.5 / 79.5; pushing F 35.5 / 40.3, AP 46.9 / 43.1;
stacking F 89.6 / 87.5, AP 91.7 / 90.8, B-3Branch 87.3 / 84.8), Table 2 ablation (82.2 / 91.3, 68.0 / 78.3, 82.6 / 80.4),
SAPIEN settings (500 fps, 20 solver iterations, 9.81, static friction 4.0), data split (1,785 shapes, 867 / 281 / 637),
positive rates 7.4 % (pushing) to 41.2 % (placement), scene 10,000 and object 1,000 points. All match. The table text is
scrambled by pdftotext, so rows were matched by order; the column headers of Table 1 were not re-read from the figure.

## BibTeX
```bibtex
@inproceedings{mo2021o2oafford,
  title     = {O2O-Afford: Annotation-Free Large-Scale Object-Object Affordance Learning},
  author    = {Mo, Kaichun and Qin, Yuzhe and Xiang, Fanbo and Su, Hao and Guibas, Leonidas},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2021},
  eprint    = {2106.15087},
  archivePrefix = {arXiv}
}
```
