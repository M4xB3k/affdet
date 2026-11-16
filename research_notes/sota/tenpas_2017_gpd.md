# Grasp Pose Detection in Point Clouds
**Authors**: Andreas ten Pas, Marcus Gualtieri, Kate Saenko, Robert Platt
**Year / Venue**: 2017 / arXiv:1706.09911 (submitted 29 June 2017). The task brief says IJRR 2017. The GPD paper's own header shows only a template placeholder ("Journal Title", SAGE format). Secondary confirmation: the reference list of 6-DOF GraspNet (arXiv:1905.10520) cites it as The International Journal of Robotics Research, 36(13-14):1455-1473, 2017. This is evidence from a citing paper, not from GPD itself.
**Cite key**: tenpas_2017_gpd
**Source read**: https://arxiv.org/abs/1706.09911 (abstract page) and https://arxiv.org/pdf/1706.09911 (full text, read via local text extraction). The arXiv id is correct.

## In one paragraph
GPD finds 6-DoF hand poses from which a two-finger hand would form a good grasp, working directly on a noisy, partial point cloud and needing no object model at test time. The method first samples several thousand hand poses around the visible surface, then turns the point cloud inside each hand's closing region into a multi-channel image, and finally scores each image with a small CNN. The labels for training come from a force-closure style test against the full mesh. The paper lists several design improvements (a new sampling scheme, a richer 15-channel descriptor that includes occlusion information, and training with simulated data as pretraining) and measures each one. On a Baxter robot in dense clutter it reaches 93% grasp success.

## Input and output
- Input: a "viewpoint cloud" (point cloud plus the camera viewpoint for each point), from one or more depth sensors, and a region of interest. Colour is ignored.
- Candidate generation: sample N points (several thousand candidates in total) from the cloud, compute a local Darboux frame at each point from the eigenvectors of the sum of surface normal outer products in a ball, then search a grid over translations along the frame's y axis and rotations about its z axis (|Y| = 10, |Phi| = 8 in the main algorithm), pushing the hand forward until contact. A candidate is kept if the hand body does not collide with the cloud and the closing region contains at least one point.
- Grasp representation: a full 6-DoF hand pose h in SE(3) for a two-finger hand. The pose fixes the contact locations when the fingers close.
- Output per candidate: a binary score (grasp or not). The final system ranks candidates and selects one with heuristics (reachability, approach direction and so on).
- CNN input: three projections of the closing-region voxel grid, each with five channels (average occupied height map, average unobserved height map, three normal channels), 15 channels in total. The voxel grid size M is not given in the text I read.

## Architecture
- A four-layer CNN with the structure of LeNet (as in Caffe): two convolution plus pooling layers, one inner-product layer with a ReLU, and one inner-product layer with softmax. Kernel sizes and strides are those of Caffe's LeNet example (not restated in the paper).
- Training: stochastic gradient descent, learning rate 0.00025, implemented in Caffe.
- Ablated inputs: 15-channel (full), 12-channel (without the unobserved-region channels), and two 3-channel versions from earlier work. The 3-channel version is used for the robot experiments because it is faster.

## How grasp labels are made
- Source: analytic test on a full mesh. No physics simulation.
- The label is "force closure" in the special case of a frictionless antipodal grasp: the two contact wrenches would balance. This is stricter than force closure for any non-zero Coulomb friction. Because meshes are noisy, each vertex is allowed a 1 mm position perturbation and the grasp counts if any perturbation within the bound gives an equilibrium. No lift, no shaking, no friction value.
- Data: BigBird objects with real Asus Xtion Pro point clouds. 55 of 125 objects were used (29 box-like, 16 cylinder-like, 10 other; complete mesh and graspable by a parallel jaw that opens at most 10 cm). 20 registered clouds per object (two views 53 degrees apart). Up to 50k labelled exemplars per object, 1.5 million in total with equal positives and negatives. About 300k are used in any one training session.
- The paper also trains on simulated clouds of 400 CAD models from 16 categories (3DNet, 207k exemplars). Tested on the real BigBird test set, this gives only about 83% accuracy against about 90% for training on real clouds, so they use simulation only for pretraining.

## Evaluation and headline results
- Metric: recall at high precision (for example recall at 99% precision), not accuracy, since a false positive costs a failed grasp.
- Classification (train/test split over views, all 55 objects): 15-channel representation about 90% accuracy; at 99% precision the recall is 89% for the 15-channel and 82% for the 12-channel representation as quoted in the text. Trained on other objects of the SAME category (category known in advance): about 93.5% accuracy and 35% recall-at-99%-precision for cylinder-like objects, about 93% and 60% for box-like objects. Trained on ALL other objects (the general novel-object case): about 92% and 27% for cylinders, about 90% and 56% for boxes.
- Robot (Baxter, dense clutter benchmark, 10 objects per round, 27 household objects): 93% grasp success with an active point cloud (20 failures in 288 attempts, 89% of objects cleared), 84% with a passive point cloud (22 failures in 138 attempts), 75% without the grasp selection heuristics (38 failures in 155 attempts).
- Sampling comparison: the proposed sampling gave at least three times the number of true positive candidates compared with an x-axis rotation scheme (122 vs 22 for a box, 259 vs 77 for a cylinder).

## How it differs from our setup
- Grasp definition: GPD's label is purely geometric (frictionless antipodal on the mesh). It has no clamp force, no lift, no tilt. The text itself says force closure may not always predict real success. Our label is a physical lift test; so for the three-way separation, GPD's label is the "geometry only" rung.
- Pose space: 6-DoF hand poses on all visible surfaces, in clutter. Ours is top-down only on a table, single object.
- Model class: GPD classifies one local grasp image at a time with a tiny CNN. The ACRONYM authors report that GPD's failures mostly come from occlusion boundaries of the point cloud, because it only sees points between the fingers (this is in the ACRONYM paper, not in GPD). Ours is a global per-point network with a max-pool context.
- Output: GPD gives a list of grasp poses with scores. We give a heatmap over object points.
- Metric idea worth copying: recall at fixed high precision. For our heatmap, reporting recall at 90 or 95% precision of the "held" and "level" labels is more useful for planning than a single AUC.
- Worth copying: they ablate the representation and report the cost of simulated versus real input clouds. Our "perfect RGB-D 0.86 versus single image 0.82" belongs in the same kind of table.

## Likely supervisor questions
1. GPD is one of the first learning-based detectors; is your model comparable? Answer: only loosely. GPD classifies local hand poses in 6-DoF. We predict a per-point top-down heatmap, so we compare against baselines of the same output type (analytic rule, height rule, width rule).
2. GPD's label ignores friction, lifting and tilt. Does yours? Answer: ours is a simulated clamp-lift-hold with a separate level label. GPD's analytic label corresponds to our analytic baseline, which is the strongest baseline (0.92 on held) and still weaker on level.
3. Why not just use recall at high precision? Answer: it is a good complement to AUC for planning use, and we can report it for held and level.

## Not verified
- The venue (IJRR 2017) in GPD's own text: the fetched PDF shows only a placeholder journal title. It is confirmed only through the reference list of 6-DOF GraspNet.
- The voxel grid resolution M and the CNN kernel sizes and channel counts (the paper refers to the LeNet solver in Caffe without listing them).
- The exact grid sizes of the candidate search used for generating the training data (|Y| = 10, |Phi| = 8 are for the main algorithm; |Y| = 80 is for the sampling comparison).

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (our 0.86, 0.82, 0.92; the page range is from the BibTeX of another paper's reference list). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@article{tenpas2017gpd,
  title   = {Grasp Pose Detection in Point Clouds},
  author  = {ten Pas, Andreas and Gualtieri, Marcus and Saenko, Kate and Platt, Robert},
  journal = {The International Journal of Robotics Research},
  volume  = {36},
  number  = {13-14},
  pages   = {1455--1473},
  year    = {2017},
  note    = {arXiv:1706.09911; journal details from the reference list of arXiv:1905.10520}
}
```
