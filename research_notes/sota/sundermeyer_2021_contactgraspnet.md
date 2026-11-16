# Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes
**Authors**: Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, Dieter Fox
**Year / Venue**: 2021 / ICRA 2021 (arXiv:2103.14127, submitted 25 March 2021)
**Cite key**: sundermeyer_2021_contactgraspnet
**Source read**: https://arxiv.org/abs/2103.14127 (abstract page) and https://arxiv.org/pdf/2103.14127 (full text, read via local text extraction). The arXiv id is correct.

## In one paragraph
Contact-GraspNet predicts grasps for unseen objects in clutter from a single depth image, in one forward pass. Its key idea is to treat the points of the observed cloud as potential grasp contacts. For most two-finger grasps at least one contact is visible, so each observed point can carry a grasp: the network predicts, per point, a contact success score, the approach and baseline directions and the grasp width. This anchors the grasp to observed geometry and cuts the 6-DoF problem down to 4 DoF per point. The labels come from the ACRONYM simulated grasp dataset, which are mapped onto the points of rendered scene clouds. On a Franka robot in structured clutter it reaches 90.2% success, against 80.4% and 62.7% for earlier methods.

## Input and output
- Input: a depth image converted to a point cloud of a whole scene (or a cube-shaped local region around a target object). n = 20000 random points, shape 20000 x 3, centred at the mean of the cloud at inference. The input has no normals or colour.
- Network output: for m = 2048 farthest points of the input, four per-point heads: a contact success score s (1 value), two 3D vectors z1 and z2 that become the baseline b and approach a, and a grasp width as 10 equidistant bins.
- Grasp representation: contact point c (observed point), approach vector a, baseline vector b (orthonormal to a through in-network Gram-Schmidt), width w and a constant distance d from baseline to gripper base. The grasp translation is t = c + (w/2) b + d a (Eq. 1 as extracted). Full pose is 6-DoF, but since the contact fixes 3 DoF, only 3 rotation DoF plus width are predicted per point ("reduced to 4-DoF").

## Architecture
- A PointNet++ style asymmetric U-shaped network built from set-abstraction and feature-propagation layers. Each set abstraction layer uses 3 parallel (multi-scale) branches with ball-query radii [0.02, 0.04, 0.08], [0.04, 0.08, 0.16] and [0.08, 0.16, 0.32] m.
- Four heads, each with two 1D-conv layers. Channel widths of the backbone layers are not given in the text I read.
- Loss: l = alpha l_bce,k + beta l_add-s + gamma l_width with alpha = 1, beta = 10, gamma = 1.
  - l_bce,k: binary cross-entropy of the contact score over all output points, but only the top k = 512 errors are back-propagated (counters class imbalance).
  - l_add-s: an average-distance loss on five gripper points, transformed by predicted and true grasps, taking the minimum over ground-truth grasps (handles multi-modality) and weighted by the predicted contact confidence.
  - l_width: weighted multi-label binary cross-entropy over the 10 width bins, weighted inversely to bin size because narrow grasps dominate.
- Optimiser: Adam, learning rate 0.001 decayed to 0.0001, batch size 3, 144,000 iterations, about 40 hours on one V100. 10,000 training scenes with 8 to 12 ShapeNet objects each.
- Runtime: 0.28 s for a full scene, about 0.19 s for a local region.

## How grasp labels are made
- Source: the ACRONYM dataset (8,872 ShapeNet meshes, 17.7 million simulated grasps). ACRONYM labels come from physics simulation: closing and shaking in the FleX simulator (see the ACRONYM note; this paper says "under varying friction", the ACRONYM paper says friction 1.0).
- Making per-point labels (the part that matters for us): objects are placed at random stable poses in scenes. Grasps that make the gripper collide are removed. A virtual camera renders a point cloud. A recorded point p_i gets label s_i = 1 if the mesh contact point of any non-colliding ground-truth grasp lies within r = 5 mm of it, otherwise 0. Positive points take the closest grasp's rotation, translation and width as regression targets. So the dense per-point "can a good grasp contact here" map is built from 6-DoF grasp labels by a distance rule.
- Success in the simulator evaluation: the open gripper does not collide, and the object is still in the gripper after grasping and a shaking motion.

## Evaluation and headline results
- Real robot (Franka Panda, RealSense L515, 9 cluttered scenes with 51 unseen objects, a maximum of two attempts per object): Table I as extracted gives 90.20% success and 84.31% on the first attempt with 59 attempts for Contact-GraspNet, against 80.39%, 68.63% and 67 attempts for 6-DOF GraspNet with CollisionNet. A further row shows 62.7% success for 6-DOF GraspNet [11] (other columns empty).
- Simulator success-coverage curves: ablations show that the binned weighted width loss is essential (without it widths collapse to narrow grasps), that l_add-s improves success at high confidence, and that a large diverse dataset like ACRONYM is needed for out-of-category generalisation (training on the small earlier dataset with 110 objects from 5 categories is not enough).
- Failure cases: thick objects needing nearly the maximum width, and small objects with low-confidence contacts.

## How it differs from our setup
- Closest conceptual match to us. Contact-GraspNet also predicts a per-point score on the observed point cloud ("a grasp can contact here"). Our heatmap "a top-down pinch centred here succeeds for some yaw" is the same kind of per-point label. The difference: they additionally regress the full grasp pose per point, we only predict the success map for a fixed approach direction.
- Label construction: they derive the point label from sparse 6-DoF grasps within a 5 mm radius. We simulate a full grid directly (33 x 21 points, step 1.3 cm, 6 yaw angles), so every point has a defined label for each channel. This is cleaner, but only for one approach direction.
- Label meaning: their positives are "closes and survives shaking in a gravity-free simulation, no collision". Our "held" adds lift and gravity; "level" adds tilt.
- Imbalance: they back-propagate only the top 512 errors. Our per-point BCE could use a similar hard-example weighting if the positive fraction is low.
- Backbone: PointNet++ with multi-scale grouping on 20,000 points, versus our small EdgeConv (k = 16, 64-64-128, global max-pool, about 0.2 M parameters). They use no normals; we use normals.
- Setting: clutter and whole scene, versus our single object.
- Worth copying: separate heads/channels for separate criteria (their heads for contact, direction and width). We already have 4 output channels; a clean split into clamp, lift, level would follow their per-point multi-head idea. Also the confidence-weighted loss, which couples the contact score to pose accuracy.

## Likely supervisor questions
1. Your heatmap looks like Contact-GraspNet's contact score. What is new? Answer: ours is for a different kind of label (physical lift and level for a top-down pinch on a dense grid) and for a single known approach direction, so no pose regression is needed. The shared idea is a per-point success map on the observed cloud.
2. Why a 5 mm radius for them, and what is your analogue? Answer: they propagate sparse grasp labels to nearby points. We test every grid point directly, so we do not need a propagation radius. (Confirm in the code how grid labels are mapped onto the cloud points, for example nearest point or a radius, before answering this in detail.)
3. Why not use PointNet++ as they do? Answer: EdgeConv with a small graph and a global feature is small and fast on 0.2 M parameters for 185 objects. A backbone comparison is future work.

## Not verified
- The exact feature channel widths of the PointNet++ layers and the number of set-abstraction layers.
- The exact formula of Eq. 1 for the translation (the extraction shows t = c + (w/2) b + d a, ordering of the terms in the PDF was garbled).
- Which column of Table I carries the 62.7 value for 6-DOF GraspNet (the extraction places it in the "Success" column with empty other columns).
- The scale of the l_add-s weighting beyond beta = 10.
- Friction in ACRONYM: this paper says "under varying friction", while the ACRONYM paper itself says all objects use friction 1.0. I could not reconcile the two statements from the texts.

## Verification (second reader, 2026-11-16)
Checked against the extracted full text (arXiv 2103.14127): Table I (90.20 %, 84.31 %, 80.39 %, 68.63 %, 62.7 %), 20,000 input
points, 2,048 output points, ball radii [0.02, 0.04, 0.08], [0.04, 0.08, 0.16], [0.08, 0.16, 0.32] m, top-k = 512, loss
weights alpha 1 / beta 10 / gamma 1, batch 3, 144,000 iterations, about 40 hours on a V100, 0.28 s / 0.19 s, label radius
r = 5 mm, "varying friction", 110 objects in the earlier dataset, 10,000 scenes. All match. For our neighbourhood rows:
their smallest ball is 2 cm on whole-scene clouds of 20,000 points; our 10 and 15 mm balls on single objects are of the
same order.

## BibTeX
```bibtex
@inproceedings{sundermeyer2021contactgraspnet,
  title     = {Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
  author    = {Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2021},
  note      = {arXiv:2103.14127}
}
```
