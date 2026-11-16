# VAT-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D ARTiculated Objects
**Authors**: Ruihai Wu, Yan Zhao, Kaichun Mo, Zizheng Guo, Yian Wang, Tianhao Wu, Qingnan Fan, Xuelin Chen, Leonidas Guibas, Hao Dong
**Year / Venue**: 2022, ICLR 2022 (arXiv v2, 1 Apr 2022)
**Cite key**: wu_2022_vatmart
**Source read**: https://arxiv.org/abs/2106.14440 (abstract page); https://arxiv.org/pdf/2106.14440 (full PDF including appendices A to H, text extracted locally; I read the main text and appendices A to F in full, appendix G and H only partly)

## In one paragraph
Where2Act predicts where to poke an object for one short move. VAT-Mart extends this to task-aware, multi-step manipulation: for example "pull this drawer open by 0.5". For every point on an articulated part the network predicts how actionable it is for that task, a distribution of whole 6-DoF gripper trajectories, and a success score for each trajectory. The training trajectories do not come from random trials. A reinforcement learning policy (TD3) explores the simulator and finds successful trajectories, and the perception network learns from them. The perception network in turn gives a "curiosity" reward that pushes the policy to find diverse solutions. At test time only the perception network is used.

## Input and output
- Input: a partial point cloud S_O of the articulated object, an interaction type T (pushing or pulling), and a task value theta (a joint angle in radians for doors, a length relative to shape scale for drawers). The contact point p (3-d) is also an input to the heads.
- Output per point p on an articulated part:
  - actionability a in [0, 1] (is there any trajectory of type T at p that accomplishes theta);
  - a distribution of trajectories, each a sequence of up to 5 six-DoF waypoints (3-d hand centre plus orientation, stored in a residual form; 30 numbers after flattening and zero-padding);
  - a success score r in [0, 1] per trajectory.
- Point count of the input cloud: not verified in the paper text. The environment follows Where2Act's camera settings (RGB-D 448 x 448, viewpoints sampled in front of the object).

## Architecture
- Input encoders: segmentation PointNet++ (single-scale grouping) gives a 128-d per-point feature f_s. Small MLPs encode the contact point (3 to 32, f_p), the trajectory (30 to 128 to 128 to 128, f_tau, 128-d) and the task (1 to 32, f_theta).
- Actionability head D_a: concatenate f_s, f_p, f_theta (192-d), then a 5-layer MLP 192-128-128-128-128-1.
- Trajectory proposal head: a conditional VAE. The encoder is an MLP 320-128-128-256 giving mean and standard deviation (128-d each). The decoder is an MLP 320-512-256-30. The conditions are f_s, f_p, f_theta.
- Trajectory scoring head D_s: concatenate f_s, f_p, f_tau, f_theta (320-d) and a 2-layer MLP 320-128-1.
- Losses: BCE for D_s. KL plus L1 (translation) plus 6D-rotation loss for the cVAE. For D_a: sample 100 trajectories from the cVAE, score them with D_s, and regress D_a to the mean of the top-5 scores with an L1 loss. So the actionability target is a prediction of the model's own scoring head, not a direct simulator outcome.
- Training: Adam, learning rate 0.001, batch 32 for the perception side. RL side: TD3 with MLPs (policy 33-512-512-512-512 with mean and variance heads, Q-network 39-512-512-512-1), buffer 2048, batch 512, learning rate 0.0001, Hindsight Experience Replay.
- Parameter count: not verified in the paper text. Training about 3 days per experiment on one V100. Inference 15 ms for the actionability map.

## Simulation and labels
- Simulator: SAPIEN with the same environment settings as Where2Act, Panda flying gripper with a velocity-based PID controller.
- Data: PartNet-Mobility, 562 shapes in 7 categories, door and drawer experiments. Split (Table D.3, garbled in the text extraction, so read with care): 328 training shapes, 94 test shapes from training categories and 140 shapes from unseen categories. For the door experiment the training categories are cabinet (270), door (17), microwave (9), refrigerator (32) and the test categories are safe (29), table (95), washing machine (16). For the drawer experiment the training category is cabinet and the test category is table.
- Label source: the RL policy. State is a 33-d vector (part pose change, target, gripper pose, finger positions, contact point, joint axis and location, and similar). Action is a residual gripper pose. Reward: 500 on success (part pose within 15 % relative tolerance of the target), a step-wise progress term of 300 times the reduction in error, a distance penalty to stay near the contact point, plus a curiosity reward of minus 500 times the success score from the perception network. At most 5 waypoint steps per trial.
- Perception training data: 5,000 successful RL trajectories and 5,000 negatives. Negatives are made by shifting the target task of a successful trajectory by a random offset (0.1 theta0 up to 45 degrees or 0.45 length), plus failed first grasps for pulling.
- The RL policy uses ground-truth simulator state. It is discarded at test time.

## Evaluation and headline results
- Metrics for the scoring module: accuracy, precision, recall, F-score on 350 positive and 350 negative trajectories per experiment, collected by running the learned RL policy on test shapes. A "coverage" score measures how well proposals cover ground-truth trajectories.
- Example (test shapes of training categories / unseen categories): door pushing accuracy 82.24 / 72.44 and F-score 82.76 / 72.54; drawer pushing F-score 81.65 / 70.52; door pulling F-score 76.06 / 72.01.
- Downstream task success rate (within 15 % tolerance), ours versus baselines (train-category shapes / unseen categories): pushing door 55.14 / 36.49 versus Where2Act-multi-step 32.67 / 32.27, heuristic 25.74 / 25.23, naive RL 1.43 / 3.72; pulling door 12.04 / 14.33 (Where2Act 6.02 / 3.51); pulling drawer 43.53 / 31.14 (Where2Act 8.31 / 9.71). The hand-written heuristic (given ground-truth handle masks and joint parameters) beats ours on drawer pushing in unseen categories (57.99 versus 38.28); ours wins on training-category shapes (56.02 versus 40.25). This matches the paper's own sentence that it wins all comparisons except one drawer-pushing entry.
- Real world: qualitative predictions on RBO, Google Scanned Objects and a cabinet scanned with a ZED Mini. A real-robot experiment is shown in a figure and the supplementary video; I could not see the video or measure success rates.

## How it differs from our setup
- Label generation: an RL policy explores and produces the labels. We enumerate a grid of grasps and drop positions exhaustively with no learning in the loop. Our labels are fully reproducible and unbiased by a policy, theirs depend on what the policy discovered.
- Label type: their actionability target is the mean of the top-5 critic scores (a soft number from a learned scorer). Ours are directly simulated binary outcomes per point.
- Task conditioning: they condition on a numeric task (open by 30 degrees). We have separate labels per predicate (hold, level, support). A task input could be a way to unify predicates later but is not needed now.
- Trajectories versus positions: they output whole trajectories. We predict only a position (and optionally yaw) and rely on a fixed top-down grasp and drop.
- Simulator: SAPIEN floating gripper versus MuJoCo parallel pads with a force-controlled squeeze, lift and hold.
- Backbone: PointNet++ 128-d plus MLP heads versus our small EdgeConv network (about 0.2 M parameters).
- Single object versus pair: single articulated object only.
- Evaluation: they never report per-point AUC on dense ground truth. They test trajectories sampled by the RL policy.
- Worth copying: the coverage-style metric for multi-modal outputs (if we ever predict several yaws per point), and the idea that perception, not the policy, is deployed. Also the "negative by perturbing a successful example" trick, though we do not need it.

## Likely supervisor questions
1. Why not use RL to find grasps as VAT-Mart does? Because our action space (top-down pads, grid of positions, 6 yaws) is small enough to enumerate. Exhaustive labels are cheaper to trust than RL discoveries, and they give a ground truth for evaluation.
2. Their actionability is a prediction of a learned scorer's top-5 mean. Is yours? No. Our per-point label is a simulated outcome, so our heatmap is supervised by physics directly, with no second model in the loop.
3. They show a real-robot run. Do you? Our camera route is tested with one perfect RGB-D view and a TripoSR reconstruction, not with a robot execution. We should say plainly that the labels come from simulation, and that the claim is about a heatmap model trained on simulated ground truth.

## Not verified
- Number of input points.
- Total parameter count.
- Contents of appendices G and H in full (metric definitions and baseline details); I relied on the main text for the metrics.
- Real-robot success rates (only a figure and a video are mentioned).

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (none). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@inproceedings{wu2022vatmart,
  title     = {{VAT-Mart}: Learning Visual Action Trajectory Proposals for Manipulating 3D {ARTiculated} Objects},
  author    = {Wu, Ruihai and Zhao, Yan and Mo, Kaichun and Guo, Zizheng and Wang, Yian and Wu, Tianhao and Fan, Qingnan and Chen, Xuelin and Guibas, Leonidas and Dong, Hao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  eprint    = {2106.14440},
  archivePrefix = {arXiv}
}
```
