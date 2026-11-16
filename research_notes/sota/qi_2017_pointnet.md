# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
**Authors**: Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
**Year / Venue**: 2017, CVPR (arXiv comment field: "CVPR 2017"; arXiv v2, 10 Apr 2017)
**Cite key**: qi_2017_pointnet
**Source read**: https://arxiv.org/abs/1612.00593 (comments field only); https://arxiv.org/pdf/1612.00593 (full text, including the supplementary). The fetch summariser could not read the binary PDF, so the text was extracted locally with pdftotext from the saved PDF.

## In one paragraph
A point cloud is an unordered set, so a network must give the same answer for any ordering of the points. PointNet does this in the simplest way. It applies the same small MLP to every point on its own, then combines all points with one symmetric function, max pooling. This gives a global feature vector that describes the whole shape. For per-point tasks, the global feature is concatenated back onto every per-point feature, so each point sees both its own features and the global context. There is no notion of a neighbourhood: points never exchange information except through the single global max. The paper proves that this form can approximate any continuous set function, and that the output depends only on a small set of "critical points".

## Input and output
- Input: n points, each with (x, y, z) by default. Extra channels (normals, colour) can be added. Classification used 1024 points sampled on mesh surfaces and normalised into a unit sphere.
- S3DIS scene segmentation: 9-dimensional points (XYZ, RGB, normalised room position); 1 m by 1 m blocks; 4096 points sampled per block at training time, all points at test time.
- Output: k class scores for classification. For segmentation, n x m scores, one per point and per class (m semantic classes or part labels).
- Part segmentation: the number of points per shape is not stated in the text I read.

## Architecture
Classification network (Fig. 2 and supplementary):
- Input transform: a T-Net (a mini PointNet) predicts a 3 x 3 matrix, applied to the xyz coordinates. T-Net body: shared MLP (64, 128, 1024), max pool, two fully connected layers (512, 256), output matrix initialised to the identity.
- Shared MLP (64, 64) -> n x 64 features.
- Feature transform: a second T-Net with the same layout but a 64 x 64 output matrix, regularised towards an orthogonal matrix, L_reg = ||I - A A^T||_F^2, with weight 0.001.
- Shared MLP (64, 128, 1024) -> n x 1024.
- Max pool over points -> 1024-dimensional global feature.
- MLP (512, 256, k) with dropout (keep ratio 0.7 on the 256-wide layer). Batch norm and ReLU everywhere.

Segmentation network (Fig. 2):
- Take the n x 64 per-point features (after the feature transform) and the 1024-dimensional global feature. Copy the global feature to every point and concatenate: n x 1088.
- Shared MLP (512, 256, 128), then shared MLP (128, m) -> n x m scores.
- The part segmentation variant (Fig. 9) adds a one-hot category vector, wider layers, and skip links that concatenate features from several layers (n x 3024 into an MLP (256, 256, 128)).

Parameters and cost (Table 6, classification model): vanilla PointNet 0.8 M parameters, 148 M FLOPs per sample; full PointNet with both transforms 3.5 M parameters, 440 M FLOPs. A parameter count for the segmentation network is not stated.

Training details: Adam, learning rate 0.001, batch size 32, learning rate halved every 20 epochs, no dropout in the segmentation network.

## Training data and labels
- ModelNet40 (12,311 CAD models, 40 classes; 9,843 train, 2,468 test). Augmentation: random rotation about the up axis, Gaussian jitter with standard deviation 0.02.
- ShapeNet part dataset (16,881 shapes, 16 categories, 50 parts). Labels are part annotations on sampled points. Part segmentation is formulated "as a per-point classification problem", scored by mIoU on points.
- Stanford 3D semantic parsing (S3DIS), 13 classes.
- Loss: a softmax classification loss, plus the 0.001 weighted orthogonality term. The segmentation loss is described only as per-point classification; an explicit formula is not given in the text I read.

## Headline results
- ModelNet40 classification, overall accuracy: 89.2 (Table 1; the MVCNN multi-view baseline is 90.1). Average class accuracy 86.2.
- ShapeNet part segmentation, mean IoU: 83.7 (Table 2). The paper states a 2.3 point gain over the best baseline (81.4 for Yi et al.; the paper's own 3D CNN baseline scores 79.4).
- Simulated Kinect partial scans (ShapeNet part): the network trained on complete and partial shapes loses only 5.3% mean IoU.
- S3DIS, mean IoU: 47.71, overall accuracy 78.62 (Table 3), against 20.12 and 53.19 for a handcrafted-feature baseline.

## How it differs from our model
- It has: learned input and feature alignment (T-Nets), a 1024-wide global bottleneck, a bigger MLP (about 3.5 M parameters for classification), batch norm with ReLU.
- Ours has: a local neighbourhood. Our k = 16 EdgeConv layers let each point see relative offsets to its neighbours. In PointNet, a point's per-point feature is computed from that point alone before the global max (PointNet++ states that PointNet "does not capture local structure"; DGCNN calls PointNet the special case k = 1).
- The global step is the same idea as ours. The n x 1088 concatenation (local 64 plus global 1024) is the direct ancestor of our "global max over N, Linear, copy back, concatenate" step.
- Plausible impact of switching for per-point heatmaps on single objects of 2k to 8k points: unlikely to help. A heatmap of grasp or support success depends on local geometry (edges, flat patches, handles), and plain PointNet has no local operator. It is a useful ablation floor (our model with k = 1), not a candidate replacement. The paper shows saturation around 1k points for classification (Fig. 15), so more points do not fix this.

## Likely supervisor questions
1. Q: Is your model just PointNet with neighbours? A: It keeps PointNet's skeleton (shared per-point MLPs, global max, global feature copied back and concatenated) but replaces the per-point MLP by EdgeConv over 16 neighbours, so the per-point features see local geometry. PointNet alone has no neighbour operator.
2. Q: Why max pooling for the global feature? A: PointNet argued it is a symmetric function that beat average pooling and an attention-weighted sum in their ordering experiment (Fig. 5), and that the output is set by a few critical points. We use max for the same reason, and because it does not depend on N, so training on 2048 points and testing on 4096 is possible.
3. Q: Does missing data matter, since our camera variant sees only one side? A: PointNet reports only a 5.3% mean IoU loss on simulated partial Kinect scans, and a 2.4 to 3.8 point accuracy drop with 50% of the points removed (classification). That is the best available evidence that max-pooled global features tolerate partial views, but it is for part labels and classes, not for grasp success.

## Not verified
- Number of points per shape for ShapeNet part segmentation: not stated in the text I read.
- An explicit segmentation loss formula: not given; only "softmax" is named, in the classification context.
- A parameter count for the segmentation network: not stated.
- Venue: CVPR 2017 is confirmed only by the arXiv comments field, not by the PDF text.

## BibTeX
```bibtex
@inproceedings{qi_2017_pointnet,
  title     = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author    = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017},
  eprint    = {1612.00593},
  archivePrefix = {arXiv}
}
```

## Verification (second reader, 2026-11-16)
Checked against the extracted text of arXiv 1612.00593. Confirmed in the text: shared MLP (64, 64) then (64, 128, 1024), max pool to a 1,024-wide global feature, and for segmentation the global feature is concatenated with each point feature to give n x 1088 (Fig. 2 and Section 4.2). Not checked: the result tables. Verdict: safe to quote for the architecture.
