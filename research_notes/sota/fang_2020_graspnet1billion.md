# GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping
**Authors**: Hao-Shu Fang, Chenxi Wang, Minghao Gou, Cewu Lu
**Year / Venue**: 2020 / CVPR 2020 (pp. 11444-11453; found on CVF open access, arXiv not checked)
**Cite key**: fang_2020_graspnet1billion
**Source read**: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf (full text, read via local text extraction). Located through a web search that returned https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html . I did not check whether an arXiv version exists. The supplementary file was not read.

## In one paragraph
GraspNet-1Billion tackles two problems in learning 6-DoF grasping for clutter: there is little training data with real sensor images, and there is no common evaluation. The authors capture 97,280 real RGB-D images of 190 cluttered scenes with two cameras, and annotate each with dense 6-DoF grasps computed analytically on the object meshes and projected into the scene through hand-annotated object poses. The evaluation does not need stored test labels: any predicted grasp is checked directly with the force-closure metric on the mesh, and the score is Precision@k averaged over friction values. They also propose an end-to-end network that predicts per point the approach direction, in-plane rotation, depth, width, and a "grasp affinity field" (tolerance to perturbation). The labels are analytic, not simulated.

## Input and output
- Input: a scene point cloud with N points, coordinates only (N x 3). During training 20k points are sampled per scene.
- Grasp representation: G = [R, t, w], with R the 3 x 3 gripper orientation, t the grasp centre and w the gripper width: a 7-DoF description (6-DoF pose plus width). The network decomposes this into: grasp point on an object surface, approaching vector chosen among V = 300 predefined viewpoints on a sphere, in-plane rotation (12 bins over 0 to 180 degrees), approach distance (4 bins: 1, 2, 3, 4 cm) and width.
- Output of the first stage: for each of M = 1024 sampled points, M x (2 + V) values: a graspable/not-graspable score and a score for each of the V viewpoints. The later stages give rotation, width and grasp confidence in 10 levels, plus a tolerance value.

## Architecture
- Approach Network: PointNet++ base with four set-abstraction layers (radii 0.04, 0.1, 0.2, 0.3 m, group sizes 64, 32, 16, 16, subsampling to 2048, 1024, 512, 256 points) then two feature propagation layers back to 1024 points with 256-dim features. An MLP head (256, 302, 302) outputs the graspable score and the V = 300 viewpoint scores.
- Operation Network: for each candidate, points inside a cylinder along the approach vector (in 4 depth bins) are transformed to a local frame, then MLPs (128, 128, 36) predict in-plane rotation class scores plus residuals, width and grasp confidence. Rotation uses classification, not regression.
- Tolerance Network: MLPs (128, 64, 12) predict the maximum perturbation a grasp can stand while its score stays above 0.5 (grasp affinity field).
- Loss weights lambda1, lambda2, lambda3, alpha, beta = 0.5, 1.0, 0.2, 0.5, 0.1. Classification: softmax or sigmoid cross-entropy; regression: smooth L1. Adam, learning rate 0.001, divided by 10 after 60 and 100 epochs, batch size 4, one RTX 2080.
- Training label for the graspable score: a point is positive if at least one ground-truth grasp lies within a 5 mm radius, negative if it is not on an object, and ignored if it is on an object but has no nearby ground-truth grasp.

## How grasp labels are made
- Source: analytic computation on 3D meshes, then projection into real scenes. No physics simulator.
- Objects: 88 objects with high-quality meshes (32 from YCB, 13 adversarial objects from Dex-Net 2.0, 43 own). Each scene has about 10 objects placed in a cluttered way. 256 viewpoints per scene, two cameras (RealSense 435 and Kinect Azure), 190 scenes, 97,280 images. Object 6D poses annotated by hand for the first frame and propagated by camera poses.
- Per object: mesh downsampled to uniform grasp points. For each grasp point, V views are sampled on a sphere. A 2D grid of gripper depths D times in-plane rotations A is searched. Gripper width is set so that there is no empty grasp and no collision.
- Score: the force-closure antipodal test is run with a friction coefficient mu decreased from 1.0 to 0.1 in steps of 0.1 until the grasp is no longer antipodal. The score is s = 1.1 - mu, in (0, 1]. A lower required friction means a more robust grasp.
- Collision check in the scene uses a simplified gripper model (their Fig. 4) and tests whether object points fall inside the gripper volume. The positive to negative label ratio is about 1:2.
- Counts: the paper gives "over one billion" in the abstract, "over 1.1 billion" in the text and "about 1.2B" in its comparison table. Treat the number as roughly 1.1 to 1.2 billion.
- Validation of the labels on a real robot: 10 objects, 100 random grasps per score setting, success about 0.96 on average for score s = 1, and low for s = 0.1 (Table 2 lists per-object values, for example banana 98%, 67%, 21% at s = 1, 0.5, 0.1).

## Evaluation and headline results
- Split: 100 training scenes and 90 test scenes (30 seen objects, 30 unseen but similar, 30 novel).
- Metric: Precision@k of the top 50 grasps after a pose-NMS, averaged over k = 1 to 50, giving AP at each friction mu. AP = mean over mu = 0.2, 0.4, 0.6, 0.8, 1.0.
- Table 3, RealSense / Kinect, AP: ours Seen 27.56 / 29.88, Unseen 26.11 / 27.84, Novel 10.55 / 11.51. GPD: 22.87 / 24.38, 21.33 / 23.18, 8.24 / 9.58. GG-CNN (rectangle based): 15.48 / 16.89, 13.26 / 15.05, 5.52 / 7.38.
- Ablation on the Kinect seen set (AP): full 29.88, regression instead of classification 23.74, without the Tolerance Network 28.53.

## How it differs from our setup
- Grasp definition: GraspNet-1Billion's label is analytic force closure with a friction sweep, on a simplified gripper. There is no lift, no clamp force, no tilt. This is the "geometry only" definition, like our analytic baseline, but it reports a graded score (1.1 - mu) rather than a binary value.
- Graded score is worth copying: instead of binary held/level we could report a margin, for example the largest tilt or slip tolerated, or the smallest friction or force at which the object is still held. That gives the supervisors a continuous "pickability".
- 6-DoF in clutter on real images versus our top-down, single object, simulated clouds.
- Evaluation without stored labels is the opposite of ours: we evaluate on precomputed simulation labels. But their idea of "scoring any prediction against the mesh on demand" matches our option of re-running the simulation at predicted points to validate them.
- Backbone: PointNet++ with a multi-stage pipeline; ours is a single small EdgeConv network and a per-point head.
- Labels at a point: their "graspable point" rule (within 5 mm of a ground-truth grasp, ignored if unlabelled) matches Contact-GraspNet's rule; ours has full coverage for every grid point so no ignore class is needed.
- Test split by seen, similar and novel objects is a good template for our held-out-object split: it separates "new instance" from "new category".

## Likely supervisor questions
1. GraspNet-1Billion used analytic labels. Why did you simulate? Answer: analytic force closure ignores gravity, centre of mass and tilt, which is exactly what our level label measures. Their own paper validates the analytic labels on a real robot (score 1 gives about 0.96 success on 10 objects), but that test covers geometry and friction, not tilt.
2. How large is your data compared to theirs? Answer: we have more objects than GraspNet-1Billion (185 versus 88) but far fewer than ACRONYM (8,872), and our labels are dense for each object (about 33 x 21 positions x 6 yaws), but only for single objects, top-down, in simulation.
3. How do you split train and test? Answer: by object, as their novel split does. We should also split by category to match "seen, similar, novel".

## Not verified
- Whether an arXiv version exists. I did not find or fetch one.
- Details in the supplementary material (gripper box dimensions, the pose-NMS procedure, camera set-up details).
- The exact number of grasps (1.1 or 1.2 billion; the paper gives different figures in different places).
- The exact gripper width limit and the finger and depth parameters of the simplified gripper model, beyond the 4 approach depths (1 to 4 cm).

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (our 185 objects, ACRONYM's 8,872). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@inproceedings{fang2020graspnet,
  title     = {GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping},
  author    = {Fang, Hao-Shu and Wang, Chenxi and Gou, Minghao and Lu, Cewu},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
  pages     = {11444--11453}
}
```
