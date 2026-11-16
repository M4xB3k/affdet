# AdaAfford: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-shot Interactions
**Authors**: Yian Wang, Ruihai Wu, Kaichun Mo, Jiaqi Ke, Qingnan Fan, Leonidas Guibas, Hao Dong
**Year / Venue**: 2022, ECCV 2022 (arXiv v6, 4 May 2023)
**Cite key**: wang_2022_adaafford
**Source read**: https://arxiv.org/abs/2112.00246 (abstract page); https://arxiv.org/pdf/2112.00246 (main paper PDF, text extracted locally and read in full; the supplementary material is not in this PDF and was not read)

## In one paragraph
A per-point affordance map predicted from one static view must average over things the camera cannot see: where the door hinge is, how stiff a joint is, how heavy a part is. AdaAfford adds a few test-time interactions (one to four) that reveal this hidden information. A module chooses where to interact next, observes whether the part moved, and a second module turns the set of interaction results into a latent code that adapts the affordance map from a "prior" to an instance-specific "posterior". Both modules are trained in the SAPIEN simulator. The paper's point for us is the limit of single-view affordance: some of the label variance is not predictable from geometry at all.

## Input and output
- Input: a single-frame partial point cloud O in R^(N x 3) from one RGB-D view (the paper does not give N in the main text), plus the set of interactions I = {(O_i, p_i, R_i, m_i)} made so far: point, gripper orientation, and observed part motion.
- Prior: the Where2Act actionability map A in [0, 1]^N and its action proposals are the starting point (zero interactions).
- Output: a posterior actionability map A_I in [0, 1]^N, an updated action score per proposed orientation, and (from the second module) the next point and orientation to try. The latent code z is 128-d.

## Architecture
- Shared input encoder: a PointNet++ segmentation network gives a 128-d feature per point (f_O in R^(N x 128)); MLPs encode actions and part motion into a 128-d f_a; the concatenation is 256-d. Encoders are not shared between modules.
- Adaptive Affordance Prediction (AAP) module:
  - Adaptive Information Encoder: encodes each interaction, then an MLP gives a 128-d latent per interaction, and another MLP gives an attention weight; z is the weighted average of the latents.
  - Adaptive Affordance Network D_AAP: MLP on the point feature and z, outputs the posterior actionability a_p|z.
  - Adaptive Critic Network C_AAP: scores an action u = (p, R) given z.
- Adaptive Interaction Proposal (AIP) module:
  - AIP Affordance Network D_AIP: scores how informative interacting at point p would be. The next point is the arg-max.
  - AIP Critic C_AIP: for that point, 100 orientations from the Where2Act proposal network are scored, the best is chosen.
  - Stopping: when a budget is reached or the maximum AIP score is under a threshold (for example 0.05).
- Losses: BCE for C_AAP against the binary outcome (part motion above tau = 0.01). L1 loss for D_AAP against the mean of the top-5 critic scores over 100 sampled actions at that point. For AIP, the target of C_AIP is the drop in the AAP loss when an interaction is added to the set (greedy "information gain"), and D_AIP regresses the top-5 mean of C_AIP.
- Training alternates between the two modules until both converge. Parameter count and learning rate: not verified in the paper text (likely in the supplementary).

## Simulation and labels
- Simulator: SAPIEN with PhysX, same setup as Where2Act: Panda flying gripper, RGB-D camera 5 units from the object, clean segmentation assumed.
- Data: PartNet-Mobility, 972 articulated objects in 15 categories (10 training categories, 5 test categories), as in Where2Act.
- New relative to Where2Act: the dynamics are randomised. Three parameters vary: friction of the target part joint, mass of the target part, and surface friction coefficient of the part. Two special tasks have deliberate kinematic ambiguity: pulling a closed cabinet door with no handle (hinge side unknown) and pushing a faucet switch (only clockwise, only counter-clockwise, or both).
- Interaction and success: as in Where2Act, a short scripted push or pull at (p, R); success if part motion exceeds 0.01.
- Number of simulated trials: not verified in the paper text.

## Evaluation and headline results
- Metrics: F-score of the critic, and sample-success-rate (10 trials per test shape; pick a random point from the top-100 actionability points, sample 100 actions, execute the best).
- Baselines: Where2Act (no interaction), Where2Act-interaction (4 FPS-chosen interactions as extra input), Where2Act-adaptation (heuristic local adjustment), ours-random, ours-fps, ablation without iterative training.
- Numbers (interaction budgets 1 / 2 / 4): pushing, all categories, training-category shapes, F-score 72.78 / 73.12 / 75.18 versus Where2Act 56.44, sample-success 33.82 / 33.23 / 35.23 versus 20.85. Pulling closed doors: F-score 57.83 / 65.60 / 79.65 versus Where2Act 48.44. Pushing faucet: 77.42 / 83.06 / 83.83 versus 64.92.
- Real robot: yes, a Franka Panda with a parallel gripper pulls open a real cabinet door after two interaction trials (qualitative, in a figure and video). No success rate is given in the main paper.

## How it differs from our setup
- Hidden information: AdaAfford's main message is that a static view cannot tell friction, mass or hinge position. Our objects are rigid single bodies. If our MuJoCo friction and density are the same for all objects (to be confirmed and stated in the thesis), the grasp labels depend only on geometry and pose. If we vary friction or mass in future, we would meet the same ambiguity.
- Interactive adaptation: not part of our design. Our model is a pure prior, like Where2Act.
- Simulator and labels: same differences as for Where2Act (SAPIEN flying gripper and short scripted moves, versus MuJoCo full grasp and lift).
- Dense versus sampled: they only observe a handful of interactions at test time. Our labels cover every grid point, so test-time adaptation is not needed to evaluate the prior.
- Backbone: PointNet++ 128-d plus MLPs, versus our small EdgeConv encoder with global max-pool.
- Single object versus pair: single articulated object.
- Evaluation: F-score and sample-success-rate, which depend on random trials and positive rates. We use AUC and best-point success.
- Worth copying: the idea of reporting the "Where2Act-style prior" alone as a baseline and separating geometry-explained from physics-hidden variance. Also a useful way to explain label noise: points that fail in one random simulation but succeed in another should be called out as "hidden-parameter" noise, which our fixed-parameter grid does not have.

## Likely supervisor questions
1. Could part of the ranking error of your heatmap (the part not explained by AUC) be unobservable physics? In principle yes (friction, mass), but if our simulator parameters are the same for every object (to be confirmed), geometry plus pose determines the label. The error is then more likely limited capacity or unseen shapes. We can test this by checking whether errors cluster at label boundaries.
2. Does your prior need test-time interaction like AdaAfford? No. The model is a one-shot predictor from a single cloud. Interaction would only help if the real object differs from simulation in unobserved ways.
3. How does this paper relate to your sim-to-real argument? It shows that even in simulation a single view limits accuracy when dynamics vary. Real objects vary far more, so a simulation-trained prior should be used to propose candidates, and we should not claim it predicts the real outcome.

## Not verified
- The number of input points N, parameter count, learning rate, batch size and number of simulated trials (supplementary material not read).
- The numeric ranges used for friction and mass randomisation.
- A real-robot success rate (only qualitative figure and video in the main paper).
- In Table 1 the extraction of the Where2Act-interaction and Where2Act-adaptation rows was garbled, so I only quote the Where2Act and ours-final rows.

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (none). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@inproceedings{wang2022adaafford,
  title     = {{AdaAfford}: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-shot Interactions},
  author    = {Wang, Yian and Wu, Ruihai and Mo, Kaichun and Ke, Jiaqi and Fan, Qingnan and Guibas, Leonidas and Dong, Hao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  eprint    = {2112.00246},
  archivePrefix = {arXiv}
}
```
