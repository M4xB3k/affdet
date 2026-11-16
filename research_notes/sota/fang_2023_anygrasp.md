# AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains
**Authors**: Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, Cewu Lu
**Year / Venue**: arXiv Dec 2022 (revised 6 June 2023); accepted to IEEE Transactions on Robotics (T-RO), 2023 (the arXiv page says "accepted to T-RO"; the PDF header reads IEEE TRANSACTIONS ON ROBOTICS)
**Cite key**: fang_2023_anygrasp
**Source read**: https://arxiv.org/abs/2212.08333 (abstract page) and https://arxiv.org/pdf/2212.08333 (full text, read via local text extraction). The arXiv id is correct.

## In one paragraph
AnyGrasp is a grasp perception system for parallel-jaw grippers that outputs dense, accurate 7-DoF grasps from a single-view point cloud, and tracks them over time for moving objects. It builds on the GraspNet-1Billion data and a GSNet-style network that predicts, per point, a graspable heatmap, the best view, in-plane rotation, depth and width. The authors add three things: a "stable score" that encodes the distance from the grasp to the object's centre of gravity (CoG), a fifth shallow approach depth, and a temporal association module that matches grasp poses across frames. Training uses real depth data with analytic labels, not simulation. On more than 300 unseen objects in bin picking it reaches 93.3% success, on par with human subjects (93.9%), and over 900 mean picks per hour.

## Input and output
- Input: a single-view partial point cloud of the scene, voxel downsampled at 0.005 m (5 mm). The geometry module uses xyz; the temporal module also uses RGB for the grasp's cylinder region.
- Grasp representation: G = [R t w], rotation R (3 x 3), centre t (3), and the minimum gripper width w suited to the object. They call this the 7-DoF grasp configuration (6-DoF pose plus width).
- Intermediate outputs of the geometry module: per point an objectness mask and a graspable heatmap; M = 1024 seed points; 300 view scores per seed; for the chosen view 12 in-plane rotations x 5 approach depths (0.5, 1, 2, 3, 4 cm) of grasp scores and widths, and 12 stable scores. The last layer therefore predicts 12 x 5 x 2 + 12 values per seed point.
- Final score at inference: original grasp score x (1 - stable score). The top-100 grasps go to post-processing.
- Temporal module: a feature vector of size C = 256 for each of the M grasps; cosine similarity gives an M x M correspondence matrix across two frames.

## Architecture
- Geometry processing module: GSNet [11] (the paper refers to it, I did not read it). A 3D-convolutional backbone gives per-point features, then MLP heads for objectness and graspable heatmap, graspable farthest point sampling for the M seeds, graspable probabilistic view selection (300 views), cylinder grouping of the local features along the selected view, and MLP blocks for scores and widths.
- Loss for geometry: softmax loss for objectness and smooth-L1 for the heatmaps and pose parameters (as in GSNet).
- Temporal association: MLP blocks with pooling for the colour features (K = 16 points per cylinder), concatenation of colour, seed features, grasp features and the 12-number pose (rotation 9 + translation 3), then an MLP. Trained with supervised contrastive loss (sigma = 0.1 for the pose distance threshold, temperature tau = 0.1). Two grasps count as the same if their distance d <= sigma, where d = delta_t / w_max + gamma x delta_R / pi with w_max = 0.01 m and gamma = 0.1.
- Training: geometry module first from scratch (batch size 4), then the geometry weights are frozen and the temporal module is trained (the paper says it is trained "jointly with it"); Adam, learning rate 0.001, "poly" decay (power 0.9), one RTX 2080 Ti. Augmentation: horizontal flip, rotation within +/-30 degrees about Z, translation, random object removal (probability 0.2).
- Post-processing: collision check on the top-100 grasps with a gripper simplified to three cubes, and a gripper-centring step so both fingertips travel the same distance. Together 80 ms for 100 grasps. Perception alone about 100 ms per frame; the system runs at 7 Hz on an RTX 2060.

## How grasp labels are made
- Source: analytic antipodal scores on 3D scanned object meshes (the GraspNet-1Billion pipeline), projected into real scenes using hand-annotated object 6D poses. No physics simulation. The paper deliberately argues for real-world perception data over simulation.
- Data: the GraspNet-1Billion training set plus 168 extra scenes with 104 new objects; in total 268 scenes with 144 objects for training. About 10 objects per scene, 256 viewpoints per scene. Collision is checked with a simplified gripper of three cubes.
- Extra labels added here: (1) a fifth approach depth of 0.5 cm for small objects; (2) the stable score, the normalised perpendicular distance from the gripper plane to the object's CoG, computed assuming a solid rigid body of uniform density, normalised by the maximum distance for that object into [0, 1] (lower is more stable; the gripper is assumed to hold the object vertically, parallel to gravity, during transport); (3) temporal association labels from neighbouring viewpoints.
- Friction values used for the analytic scores are not restated here; they follow GraspNet-1Billion (friction sweep, see that note).
- Success in the real experiments: the object is lifted successfully (attempt-centric success rate, the ratio of successful attempts to total attempts).

## Evaluation and headline results
- Static bin picking, over 300 unseen objects, UR5 arm with Robotiq-85 gripper and RealSense D415 or D435: 93.3% success on average, against 93.9% for human subjects with the same two-finger jaw (human accuracy ranges 91.4% to 96.6% over object kinds, AnyGrasp ranges 81.5% to 100%). Over 900 mean picks per hour with a single arm (the paper compares with 300 MPPH for a dual-arm prior system).
- Dynamic grasping: catching robot fish in water, 75.5% average success (failures are partly due to slip in water).
- Sim versus real training: a model trained on perfect simulated depth images (or with added Gaussian noise) loses clearly on the GraspNet-1Billion benchmark and in real bin picking; the gap widens on novel objects.
- Stable score test: on a long heavy object, 25 attempts each, 16 in-hand slips (3 failed grasps) without the stable score and 11 slips (2 failed grasps) with it.

## How it differs from our setup
- Grasp definition: lifting success is only measured on the real robot. Training labels are analytic antipodal scores. The one place where stability enters the training label is the CoG-based stable score. This is the only paper here that explicitly separates "can clamp" from "will hold stably", and it does it with a geometric proxy for tilt/slip risk. That is a direct hint for separating our clamp, lift and level labels: the level label corresponds to their CoG-distance idea, but measured in simulation, not with a proxy.
- Our analytic baseline needs mass and CoG too; AnyGrasp assumes uniform density and shows this is sufficient to give a gain (16 versus 11 slips), which supports using a uniform-density CoG in our analytic baseline.
- 7-DoF in clutter with a real depth camera versus our top-down single object. Their heatmap "graspness" over points is again a per-point score (like ours), followed by a per-point best view.
- Backbone: 3D convolution versus our EdgeConv graph. Not directly comparable.
- Sim versus real: AnyGrasp's finding that simulated clean depth hurts transfer matches our observation that reconstruction from a single image (0.82) is worse than a perfect RGB-D view (0.86), though the two comparisons are not the same (theirs is clean simulated depth versus real depth, ours is a reconstruction versus a perfect view). We should say that all our labels and inputs are simulated, so a real-depth experiment would be the next step.
- Worth copying: report both attempt-centric success and object-centric completion rate, and compare to a human baseline on the same task; report slip counts for the "level" question separately from failed lifts.

## Likely supervisor questions
1. AnyGrasp trains on real data with analytic labels. Why is your simulated label better? Answer: it is not "better" in general. It measures a different thing (lift under gravity and tilt), which analytic antipodal labels cannot, and the supervisors accepted simulation as ground truth. AnyGrasp's own stable score shows that stability beyond the antipodal condition matters.
2. How do you handle centre of mass? Answer: (confirm in the code where MuJoCo takes mass and inertia from, for example a density times mesh volume, before saying this aloud) the simulation applies whatever mass the objects were given; our analytic baseline needs mass and CoG explicitly, and AnyGrasp shows a uniform-density assumption already helps.
3. Does your model run in real time? Answer: the model is small (about 0.2 M parameters) and one pass produces the map; AnyGrasp runs at 7 Hz with its larger model, so speed is not the bottleneck for either.

## Not verified
- Friction values for the analytic scores (the paper defers to GraspNet-1Billion).
- GSNet backbone details (3D convolution type, channel widths): the paper refers to GSNet [11] and I did not read that paper.
- The exact value of the AnyGrasp row and the DexNet 4.0 row in the paper's Table I beyond the 93.3% overall and 93.9% human figures.
- Peer-reviewed publication details (volume, pages) of the T-RO version.

## BibTeX
```bibtex
@article{fang2023anygrasp,
  title   = {AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains},
  author  = {Fang, Hao-Shu and Wang, Chenxi and Fang, Hongjie and Gou, Minghao and Liu, Jirong and Yan, Hengxu and Liu, Wenhai and Xie, Yichen and Lu, Cewu},
  journal = {IEEE Transactions on Robotics},
  year    = {2023},
  note    = {arXiv:2212.08333}
}
```
