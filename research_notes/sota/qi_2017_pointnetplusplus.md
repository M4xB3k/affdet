# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
**Authors**: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas
**Year / Venue**: 2017, NeurIPS (NIPS 2017). Not shown in the PDF text or the arXiv page, which has no comments or journal field; venue taken from the task brief. arXiv v1, 7 Jun 2017.
**Cite key**: qi_2017_pointnetplusplus
**Source read**: https://arxiv.org/abs/1706.02413 (metadata only); https://arxiv.org/pdf/1706.02413 (full text including the supplementary). The fetch summariser could not read the binary PDF, so the text was extracted locally with pdftotext from the saved PDF.

## In one paragraph
PointNet has no local structure: it pools the whole set once. PointNet++ applies PointNet recursively on a nested partition of the points, like a CNN with growing receptive fields. At each level it picks centroids by farthest point sampling, groups the points near each centroid with a ball query, and runs a small PointNet on each group in a local frame. This produces fewer points with richer features, level after level. Because real scans have uneven density, the paper adds multi-scale and multi-resolution grouping, trained with random input dropout, so the network learns to rely on larger neighbourhoods where points are sparse. For per-point outputs, a decoder interpolates features back up to the dense points and merges them with skip links.

## Input and output
- Input: a set of points with d-dimensional coordinates and C-dimensional features, an N x (d + C) matrix. Coordinates only, or with normals. ModelNet40 default: 1024 points; the "with normal" run uses face normals and N = 5000 points. MNIST: 512 points (2D). ScanNet: cubes of 1.5 m by 1.5 m by 3 m, padded to a fixed 8192 points. SHREC15: 1024 points.
- Output: class scores (classification) or per-point class scores (segmentation).

## Architecture
Set abstraction (SA) level, three layers (Sec. 3.2):
- Sampling: iterative farthest point sampling (FPS) picks N' centroids from N points.
- Grouping: ball query returns the points within a radius of each centroid (an upper limit on K is set). The paper prefers ball query to kNN because a fixed radius gives a fixed region scale, which the paper says makes local features more generalisable across space. Table 5 (ModelNet40, 1024 points, classification accuracy): kNN k = 16 gives 89.3, kNN k = 64 gives 90.3, radius 0.1 gives 89.1, radius 0.2 gives 90.7.
- PointNet layer: coordinates are translated into the local frame of the centroid, then a shared MLP and max pool give one feature per group. Output: N' x (d + C').

Density-adaptive layers (Sec. 3.3): multi-scale grouping (MSG) concatenates features from several radii; multi-resolution grouping (MRG) concatenates features from the previous level with a PointNet run on all raw points in the region. Random input dropout draws a ratio uniformly from [0, p] with p = 0.95 per training set.

Feature propagation for segmentation (Sec. 3.4): interpolate features of the N_l coarse points onto the N_(l-1) finer points by an inverse distance weighted average of the k = 3 nearest neighbours (power p = 2), concatenate with the skip-linked features of the same level, then apply a "unit PointNet" (shared fully connected layers with ReLU).

Segmentation network for part and semantic segmentation (supplementary):
SA(512, 0.2, [64, 64, 128]) -> SA(128, 0.4, [128, 128, 256]) -> SA([256, 512, 1024]) -> FP(256, 256) -> FP(256, 128) -> FP(128, 128, 128, 128, K).
Notation: SA(number of regions, ball radius, MLP widths). The last SA is global. Dropout 0.5 in the last two fully connected layers. Batch norm and ReLU after all fully connected layers except the last.
Scene labelling network: SA(1024, 0.1) -> SA(256, 0.2) -> SA(64, 0.4) -> SA(16, 0.8) then four FP levels.

Size and speed (Table 7, batch size 8, one GTX 1080): model size in MB, not parameter count. SSG 8.7 MB, 82.4 ms; MSG 12 MB, 163.2 ms; MRG 24 MB, 87.0 ms; PointNet 40 MB, 25.3 ms; vanilla PointNet 9.4 MB, 11.6 ms.

## Training data and labels
- MNIST (60k train, 10k test), ModelNet40 (9,843 train, 2,468 test), SHREC15 (1200 shapes, 50 classes, five-fold cross validation), ScanNet (1201 train and 312 test scenes). Part segmentation on ShapeNet parts is in the supplementary (16,881 shapes, 50 parts), with a normal direction per point.
- Loss: for part segmentation, "Cross-entropy loss is minimized". Optimiser Adam, learning rate 0.001, augmentation by random scaling, perturbation of object and point locations, and random rotation. About 20 hours to train to convergence on a GTX 1080 or Titan X.

## Headline results
- ModelNet40 classification accuracy (Table 2): 90.7 with coordinates only (1024 points), 91.9 with normals (5000 points), against PointNet 89.2 and the multi-view MVCNN 90.1.
- ShapeNet part segmentation, mean IoU (Table 4 in the supplement): 85.1, against PointNet 83.7.
- MNIST error rate (Table 1): 0.51%, against PointNet 0.78%.
- SHREC15 non-rigid classification (Table 3): 96.09 with geodesic neighbourhoods and intrinsic features, against 60.18 with plain XYZ in a Euclidean metric.

## How it differs from our model
- It has: a hierarchy (downsampling by FPS, upsampling by interpolation), ball-query neighbourhoods with a fixed physical radius, multi-scale grouping, and random dropout of input points during training for density robustness.
- Ours has: a flat per-point resolution (no downsampling, so no interpolation blur), an asymmetric edge function [f_i, f_j - f_i] (PointNet++ uses h(x_j) only in its edge, in DGCNN's reading), and a fixed k = 16 neighbour count.
- Density point worth knowing: PointNet++ says kNN neighbourhoods change scale with density while a ball fixes it. Our k stays at 16 when N goes from 2048 (training) to 4096 (evaluation). On a surface the kNN radius scales roughly with sqrt(k / N), so evaluation neighbourhoods are about 0.7 times as wide (own estimate, not from the paper). Their Table 5 shows kNN k = 16 at 89.3 against a ball radius of 0.2 at 90.7 on ModelNet40 classification.
- Plausible impact of switching for per-point heatmaps on single objects of 2k to 8k points: a hierarchy would give a larger receptive field and might help where the label depends on the object at a coarser scale (for example where a handle is relative to the body). It brings interpolation blur at the per-point output, and FPS plus ball query are slower than our single static kNN. The most useful ideas to borrow cheaply are the ball query and random input dropout for density robustness, not the whole hierarchy.

## Likely supervisor questions
1. Q: Why no hierarchy like PointNet++? A: Our objects are single items of 2k to 8k points, and the label (will this grasp succeed) is largely local plus a global shape cue, which the global max provides. We have not shown that a hierarchy helps; it is an open ablation. The cost is interpolation blur at the per-point output.
2. Q: How does your model cope when the point density changes between training (2048) and evaluation (4096)? A: Honestly, only partly. PointNet++ was built for this problem (ball query, multi-scale grouping, random input dropout). Our k = 16 covers a smaller patch at 4096 points, so we should test evaluating at 2048 and 4096 and consider a ball query or random point dropout.
3. Q: Does PointNet++ use normals, and do you? A: Yes. The best ModelNet40 result (91.9) uses face normals with 5000 points, and the part segmentation gives each point its normal. Our simulated variant has normals; the camera variant sets them to zero, so a drop of accuracy there partly reflects missing normals.

## Not verified
- Venue (NeurIPS 2017): not in the PDF text or the arXiv metadata; taken from the brief.
- The ScanNet bars in Fig. 5 parsed badly from the PDF text, so no ScanNet number is cited.
- No parameter count: only model size in MB is stated.
- The remark that PointNet++ uses h(x_j) as its edge function comes from the DGCNN paper (Table 1), not from this paper.

## Verification (second reader, 2026-11-16)
Checked against the extracted full text (pdftotext of arXiv 1706.02413): Table 2 (90.7, 91.9, 89.2, 90.1), Table 4 (85.1, 83.7),
Table 1 (0.51 %, 0.78 %), Table 3 (96.09, 60.18), Table 5 (89.3, 90.3, 89.1, 90.7), Table 7 (sizes and times), dropout
p = 0.95, interpolation k = 3 and p = 2, dataset sizes, the SA/FP networks of B.1 (classification SSG and MSG, scene labelling,
part segmentation), "Cross-entropy loss is minimized", about 20 hours. All match. Exact quote for the ball query (Sec. 3.2):
ball query "guarantees a fixed region scale", preferred "for tasks requiring local pattern recognition (e.g. semantic point
labeling)"; the upper limit of K is "set in implementation". The paper does not say how a ball with fewer than K points is
filled. Correction to "How it differs": measured on our training items the kNN shrink from 2,048 to 4,096 points is 2-25 %,
not 0.7x, because partial views concentrate the points; the larger effect is the 4x spread across objects
(see architecture_defence_sheet.md, weak point 1).

## BibTeX
```bibtex
@inproceedings{qi_2017_pointnetplusplus,
  title     = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author    = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017},
  eprint    = {1706.02413},
  archivePrefix = {arXiv}
}
```
