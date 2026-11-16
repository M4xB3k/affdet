# 6-DOF GraspNet: Variational Grasp Generation for Object Manipulation
**Authors**: Arsalan Mousavian, Clemens Eppner, Dieter Fox
**Year / Venue**: 2019 / ICCV 2019 (arXiv:1905.10520, v2 17 Aug 2019)
**Cite key**: mousavian_2019_6dofgraspnet
**Source read**: https://arxiv.org/abs/1905.10520 (abstract page) and https://arxiv.org/pdf/1905.10520 (full text, read via local text extraction). The arXiv id is correct.

## In one paragraph
6-DOF GraspNet treats grasp generation as sampling from a learned distribution of successful grasps, because good grasps of a mug form several separate modes (rim, handle, bottom). A variational autoencoder (VAE) takes the point cloud of one object and a latent code and produces a 6-DoF grasp pose. A second network, the grasp evaluator, predicts whether a given grasp would succeed. The evaluator is differentiable, so its gradient is used to nudge failed grasps towards success (iterative refinement). Both networks are trained only in simulation and applied to a real robot without extra steps. The real robot reaches 88% success on 17 objects, against 47% for GPD in the same protocol.

## Input and output
- Input: the partial point cloud X of a single object from a depth camera (the object is cut out with plane fitting and clustering). Grasps are expressed in a frame whose origin is the centre of mass of the observed point cloud, with axes parallel to the camera axes. The exact number of points is not stated in the text I read.
- Output: a grasp g = (R, T) in SE(3). R is a unit quaternion (output of a linear layer followed by L2 normalisation), T is a 3-vector without normalisation. So 6 degrees of freedom (the gripper is a parallel-jaw model whose closing is fixed).
- Latent space: dimension 2, chosen by an ablation (dimensions 1, 2, 3, 4 compared). Prior N(0, I).
- Evaluator input: the object point cloud plus a point cloud of the gripper placed at the candidate pose, merged into one cloud with a binary feature that says object versus gripper.

## Architecture
- Both the generator (encoder and decoder) and the evaluator are based on PointNet++.
- Three set-abstraction layers sample 128, 32 and all points, with ball radii 2 cm, 4 cm and infinity. Each layer uses 3 fully connected layers. Channel sizes per layer: [64, 64, 128], [128, 128, 256], [256, 256, 512].
- These are followed by two fully connected layers with 1024 units.
- VAE: encoder maps (point cloud, grasp) to a latent z. The decoder takes each point feature concatenated with z. During inference the encoder is removed and z is sampled from N(0, I).
- Losses: VAE loss = reconstruction + alpha x KL (alpha = 0.01). The reconstruction loss is the mean L1 distance between predefined points transformed by the true and predicted grasp (so rotation and translation are combined in one term). The evaluator uses binary cross-entropy (softmax output).
- Refinement: gradient of the success score with respect to the grasp pose, through the transformed gripper points. Update step limited so that translation moves at most 1 cm. The AUC curve plateaus after about 10 refinement iterations.
- Training: Adam, learning rate 0.0001. Evaluator batches: 30% positives, 30% negatives, 40% hard negatives (positive grasps perturbed by +/-0.6 rad per axis and +/-3 cm in translation). Generator batches: a render of the object from a random view plus 64 grasps drawn with stratified sampling.
- Speed (Titan XP, batch of 200 latent values): forward pass of VAE plus evaluator 0.04 s, each refinement iteration 0.3 s.

## How grasp labels are made
- Source: physics simulation with the NVIDIA FleX simulator. The gripper is free-floating and the object is free-floating, with no gravity (as in a prior work they cite).
- Candidate sampling: random points on the mesh surface, gripper z axis aligned with the surface normal, standoff distance uniform between zero and the finger length, and random rotation about z. Only grasps with no collision and a non-empty closing volume are simulated.
- Success label: after closing the fingers the gripper executes a predefined shaking motion. A grasp is positive if the object is kept between both fingers. Surface friction and object density are constant (values not stated in the text I read).
- Size: 206 objects from six categories of ShapeNet (randomly generated boxes and cylinders, plus bowls, bottles and mugs). 10,816,720 candidate grasps were sampled. 7,074,038 (65.4%) passed the non-empty closing-volume test and were simulated. 2,104,894 were successful (19.4% of all candidates sampled).
- Hard negatives for the evaluator: pre-generated negatives and perturbed positives (gripper in collision or far from the object).

## Evaluation and headline results
- Metrics: success rate (fraction of generated grasps that succeed in simulation) and coverage rate (fraction of positive training-set grasps lying within 2 cm of a generated grasp, by translation). Curves of success against coverage, summarised by AUC.
- Ablations: latent dimension 2 beats 1, 3 and 4; refinement improves both success and coverage; the learned VAE sampler beats a geometric sampler (surface normals plus random standoff and rotation) in both success and coverage.
- Real robot (Franka Panda, RealSense D415 on the hand, 51 trials per method, success = lift 10 cm without dropping): overall success 88% for 6-DOF GraspNet against 47% for GPD. Per-category average 90% against 52%. By category: box 83% vs 50%, cylinder 89% vs 78%, bowl 100% vs 78%, mug 86% vs 6%.

## How it differs from our setup
- Grasp definition: success is "object stays between the fingers after closing and shaking, with no gravity". This is a clamp-and-hold-under-disturbance test. Lifting against gravity and tilt are not tested. Our label adds a real lift and an explicit tilt (level) criterion.
- Pose space: full 6-DoF poses around one object. Ours is top-down only with a grid of positions and yaw, producing a heatmap.
- Model: generative (VAE) plus evaluator, in the grasp frame. Ours is a discriminative per-point classifier: no sampling, no refinement. Their evaluator is the closest relative to a "score a candidate" network; our heatmap is the dense version of that for one approach direction.
- Evaluator trick worth copying: represent the gripper as a point set in the same cloud as the object, with an object-or-gripper flag. For our top-down pad grasp this suggests feeding the yaw and pad footprint into the network rather than only predicting a pooled per-point score.
- Hard negatives: they build hard negatives by perturbing positives. We have natural hard negatives (grid points near a held region that fail), which could be weighted.
- Single object: they also handle one object at a time, so the setting is comparable to ours (no clutter), unlike Contact-GraspNet and AnyGrasp.

## Likely supervisor questions
1. Why not a generative model over grasps like 6-DOF GraspNet? Answer: our output is a dense map over the object's own points for a fixed top-down approach, so a classifier is the direct fit. A generative model solves the multi-modality of 6-DoF poses, which we do not have.
2. Their label uses shaking without gravity; yours lifts under gravity. Is that a problem? Answer: it is stricter and closer to what the planner needs. ACRONYM (same group) measured 91% precision and 93% recall of gravity-free shaking against gravity labels, so the two are close, but ours adds the level criterion.
3. Do you have a coverage metric? Answer: not in the same sense. Our analogue is per-point AUC plus recall at a fixed precision over the held and level labels.

## Not verified
- Number of input points to the networks, the batch size and the training epochs.
- The friction coefficient, object density and finger force used in the FleX labelling ("constant", not stated in the text I read).
- Which exact simulated success rate the paper reports against GPD (only the real-robot comparison was read in full; the simulation comparison is in figures).

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (ACRONYM's 93 % recall). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@inproceedings{mousavian2019graspnet,
  title     = {6-DOF GraspNet: Variational Grasp Generation for Object Manipulation},
  author    = {Mousavian, Arsalan and Eppner, Clemens and Fox, Dieter},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019},
  note      = {arXiv:1905.10520}
}
```
