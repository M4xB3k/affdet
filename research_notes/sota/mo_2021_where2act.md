# Where2Act: From Pixels to Actions for Articulated 3D Objects
**Authors**: Kaichun Mo, Leonidas Guibas, Mustafa Mukadam, Abhinav Gupta, Shubham Tulsiani
**Year / Venue**: 2021, ICCV 2021 (arXiv v2, 10 Aug 2021)
**Cite key**: mo_2021_where2act
**Source read**: https://arxiv.org/abs/2101.02692 (abstract page); https://arxiv.org/pdf/2101.02692 (full PDF including the supplementary, text extracted locally and read in full)

## In one paragraph
Where2Act asks: given one view of an articulated object such as a cabinet, where should a robot push or pull, and from which gripper orientation, so that a part moves? The labels are not drawn by humans. The agent tries interactions in the SAPIEN physics simulator and records whether a part moved enough. A network then learns to predict, for every pixel or point, an actionability score, a set of gripper orientation proposals, and a success score for each proposal. Random trials are very wasteful (about 1 % are positive for pulling), so the authors mix random sampling with sampling biased towards points the network already thinks are good. The model transfers to object categories it never saw in training, and the paper shows qualitative results on real scans and images.

## Input and output
- Input: either one RGB image (224 x 224 after downsampling from 448 x 448) or a partial point cloud from one RGB-D view. The point cloud is back-projected from depth, background rejected, then furthest point sampling to 10,000 points (10000 x 3 in the supplementary).
- Six action primitives, one network per primitive: pushing, pushing-up, pushing-left, pulling, pulling-up, pulling-left.
- Output per pixel or point on the visible articulated parts:
  - an actionability score a_p in [0, 1];
  - several gripper orientations R in SO(3) (3-DoF orientation, in the 6D rotation representation) drawn from noise z;
  - a success likelihood s in [0, 1] for each proposal. A value above 0.5 counts as a positive proposal.

## Architecture
- Backbone: for images, a UNet with an ImageNet-pretrained ResNet-18 encoder and a symmetric decoder. For point clouds, a PointNet++ segmentation network with 4 set-abstraction layers (single-scale grouping) and 4 feature-propagation layers. Both give a per-pixel or per-point feature f_p of 128 dimensions.
- Three decoder heads, each an MLP with one hidden layer of size 128:
  - actionability module D_a(f_p) with a sigmoid;
  - action proposal module D_r(f_p, z), with z a 10-dimensional Gaussian noise vector, a conditional generative model that outputs the orientation;
  - action scoring module D_s(f_p, R), output a scalar in [0, 1].
- Losses: binary cross-entropy for D_s. Min-of-N loss over 100 noise samples for D_r. For D_a, a squared error to the mean score D_s gives to 100 proposals from D_r (the "expected success rate"). Final loss L = Ls + Lr + 100 x La. D_s is trained first, then the three decoders jointly.
- Adam, learning rate 0.001. Training takes 4 to 5 days, 0.8 s per iteration. Inference is 4 ms for a batch of 32 images.
- Parameter count: not verified in the paper text.

## Simulation and labels
- Simulator: SAPIEN (NVIDIA PhysX), 500 fps, 20 solver iterations, gravity 9.81, static and dynamic friction 4.0, restitution 0.01. Parts are approximated by V-HACD convex hulls. The object root is fixed.
- Actuator: a Franka Panda "flying gripper" (8 DoF: 3 position, 3 orientation, 2 fingers). No robot arm.
- Data: PartNet-Mobility, 15 selected categories, 972 shapes. 10 training categories (773 shapes, split 586 train and 187 test) and 5 test-only categories (199 shapes). Each object starts at rest pose with 50 % chance and at a random part pose with 50 % chance.
- Interaction: a position p on the surface and an orientation R from the hemisphere above the tangent plane. Orientations from the other hemisphere are marked negative without simulation. A trial also fails without simulation if the gripper collides at the start. Each primitive is a hard-coded short trajectory (for example pushing moves 0.05 unit forward).
- Success: the contact point must move within 60 degrees of the intended direction, and the part joint must move more than 0.01 unit or 0.5 of its range.
- Sampling: offline random sampling until about 10,000 positive trajectories exist per primitive. Then online adaptive sampling: softmax over predicted scores picks the next position. Final mix is 50 % random and 50 % prediction-biased. Training batches use equal positives and negatives, and equal shapes per category.
- Cost: 2 to 8 s per simulated interaction. 3 to 4 days with parallel simulation to collect the offline data.
- Total number of simulated interactions: not verified in the paper text (only the 10,000 positives per primitive target is stated).

## Evaluation and headline results
- Metrics: F-score of the action scoring network on random interaction trials (positive class), and sample-success-rate (ssr): sample a point from the actionability map, sample 100 proposals, pick one by score, run it in the simulator, count the fraction of successes.
- Baselines: B-Random, B-Normal (ground-truth normal replaces the feature), B-PCPNet (predicted normals and curvature), and an ablation without online sampling.
- Pushing, 3D point cloud, numbers are "test shapes from training categories / unseen categories": F-score 43.76 / 26.61 and ssr 28.54 / 14.74. B-Random gets 12.02 / 7.40 F and 6.80 / 3.79 ssr. B-Normal gets 31.94 / 17.39 F and 21.72 / 11.57 ssr. The 2D image version gets 34.21 / 22.68 F.
- Pulling is much harder: 3D F-score 10.95 / 12.88, ssr 7.51 / 7.85.
- Online sampling helps in most rows (for example pushing F-score 40.54 to 43.76 on training categories).
- Real world: qualitative predictions on real 3D scans (Replica, Google Scanned Objects) and on real 2D images. There is no real-robot execution.

## How it differs from our setup
- Simulator and actuator: SAPIEN with a floating two-finger gripper and short scripted pushes and pulls on articulated parts. We use MuJoCo with parallel pads, a full grasp (squeeze to about 30 N, lift 50 cm, hold) and a level check.
- Label definition: they label an (point, orientation) action as success if a part moves. We label a grid point (and yaw) as success if the object is held, plus a second "stays level" label. Our success is stricter and longer-horizon.
- Dense versus sampled: they sample random interactions, find only 1 % positives for pulling, and need adaptive sampling and balanced batches. We simulate an exhaustive grid per object, so we have no sampling bias, no need for active sampling, and a full ground-truth map for evaluation. This is a clear advantage to state.
- Orientation: they predict orientation proposals with a conditional generator and score them. Our 6 yaw angles are enumerated, so we could predict a score per yaw (6 channels) instead of one per point.
- Backbone: PointNet++ with 128-d features, versus our small EdgeConv network (about 0.2 M parameters) with a global max-pool feature.
- Input: partial 10,000-point scan from one view. Ours is an object cloud with normals (complete or partial depending on the experiment), and the camera route (one RGB-D view, TripoSR) is a separate test.
- Single object versus pair: Where2Act is single object only. Pair reasoning appears in O2O-Afford.
- Evaluation: F-score and ssr from fresh simulation trials. We use per-point AUC on held-out objects and top-1 success. Their ssr is close in spirit to our "is the best predicted point a success", but they sample from the map instead of taking the arg-max.
- Worth copying: the 50 % random plus 50 % prediction-biased sampling idea if we ever need more data for a hard predicate; the B-Normal style baseline (does a normal-only model do as well?), which tests whether the network learns more than local surface orientation; holding out whole categories to test generalisation.

## Likely supervisor questions
1. Why do you not need active sampling like Where2Act? Because our simulator labels every grid point of every object exhaustively, so positives are never rare in the data we generate. Where2Act could afford only random trials over a huge space (position times orientation), so it needed adaptive sampling.
2. How is your per-point heatmap related to their "actionability"? Their actionability is the expected success over proposed orientations. If we pool our 6 yaws into one score per point (for example mean or max), we have the same kind of quantity. We should say which pooling we use.
3. Is your evaluation comparable to their numbers? No. They report F-score and ssr on random trials with 1 to 40 % positive rates, we report per-point AUC on dense labels. The ideas match (top-1 success is like ssr), the numbers do not.

## Not verified
- Parameter count of the networks.
- Total number of simulated interactions used for training.
- Whether the 4 to 5 day training time is per primitive or for all six (the paper gives one figure).

## Verification (second reader, 2026-11-16)
Checked against the extracted full text (arXiv 2101.02692): every number in this note occurs in the paper. The main table
(F-score and sample-success, columns B-Random, B-Normal, B-PCPNet, 2D-ours, 3D-ours, rows per action type) was matched by
order: pushing F 12.02 / 7.40, 31.94 / 17.39, 34.21 / 22.68 (2D), 43.76 / 26.61 (3D); pushing ssr 6.80 / 3.79, 21.72 / 11.57,
28.54 / 14.74; pulling F 10.95 / 12.88 and ssr 7.51 / 7.85 (3D). The online-sampling ablation row 40.54 / 25.66 to
43.76 / 26.61 also matches. The table text is scrambled by pdftotext; headers were matched by position, not read from the figure.

## BibTeX
```bibtex
@inproceedings{mo2021where2act,
  title     = {Where2Act: From Pixels to Actions for Articulated 3D Objects},
  author    = {Mo, Kaichun and Guibas, Leonidas J. and Mukadam, Mustafa and Gupta, Abhinav and Tulsiani, Shubham},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  eprint    = {2101.02692},
  archivePrefix = {arXiv}
}
```
