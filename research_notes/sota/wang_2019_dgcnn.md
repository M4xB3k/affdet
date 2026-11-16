# Dynamic Graph CNN for Learning on Point Clouds
**Authors**: Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon
**Year / Venue**: 2019, ACM Transactions on Graphics (DOI 10.1145/3326362, printed in the PDF; the PDF header reads "Vol. 1, No. 1, Article 1", which is a placeholder, so volume and pages are not given here). arXiv v1 24 Jan 2018, v2 11 Jun 2019.
**Cite key**: wang_2019_dgcnn
**Source read**: https://arxiv.org/abs/1801.07829 (metadata only); https://arxiv.org/pdf/1801.07829 (full text, v2). The fetch summariser could not read the binary PDF, so the text was extracted locally with pdftotext from the saved PDF.

## In one paragraph
PointNet processes each point alone, and PointNet++ builds local groups only from the input coordinates. DGCNN introduces EdgeConv, an operator that builds a k-nearest-neighbour graph and computes a learned feature for every edge, from the centre point and its offset to the neighbour. A max over the neighbours of each point gives that point's new feature. The paper's second idea is to recompute the kNN graph in the feature space of each layer, not only once on the input xyz. Points that look alike semantically (for example the two wings of an aeroplane) then become neighbours even if they are far apart in space, which spreads information non-locally. PointNet is the special case k = 1. The authors also report a "baseline" with a fixed graph from input proximity, which is the design our model uses.

## Input and output
- Input: an F-dimensional point cloud with n points (F = 3 for xyz; colour, normals can be added). ModelNet40: 1024 points (2048 in one run), only xyz. ShapeNet part: 2048 points per shape. S3DIS: 1 m by 1 m blocks, 4096 points per block in training, all points in testing, 9-dimensional points (XYZ, RGB, normalised position).
- Output: class scores for classification (c classes), or per-point scores for p semantic labels in segmentation.

## Architecture
EdgeConv (Sec. 3.1):
- Directed kNN graph, including a self-loop. Edge feature e_ij = h_Theta(x_i, x_j). Output x'_i = max over neighbours j of h_Theta(x_i, x_j) (Eq. 1, 9).
- The paper compares five choices of h. The one it adopts is the asymmetric edge function h(x_i, x_j - x_i) (Eq. 7). Written out as a single layer: e'_ijm = ReLU(theta_m . (x_j - x_i) + phi_m . x_i) (Eq. 8), followed by a max over j. Using only x_j - x_i would be fully translation invariant but loses where the patch sits; using only x_i is PointNet. (Section 4.3 writes x_i - x_j in the ablation text; Eq. 7 has x_j - x_i.)
- Max is symmetric, so the layer is invariant to the order of the neighbours.

Dynamic graph (Sec. 3.2): in layer l the k nearest neighbours are found from the pairwise distances of the layer's input embedding, not of the xyz. The paper says the receptive field is then "as large as the diameter of the point cloud" while staying sparse. The cost is a pairwise distance matrix in feature space per layer.

Classification network (Sec. 4.1, Fig. 3 top):
- Spatial transformer (3 x 3 matrix) in the full version; four EdgeConv layers. The text writes "three shared fully-connected layers (64, 64, 128, 256)", which lists four widths; I report it as written. Graph recomputed after each layer; k = 20 (k = 40 for the 2048-point run, "to maintain the same density").
- Shortcut connections: concatenate the layer outputs, 64 + 64 + 128 + 256 = 512 channels per point; one shared fully connected layer (1024); global max (or sum) pooling; two fully connected layers (512, 256) with dropout keep probability 0.5. LeakyReLU and batch norm on all layers.

Segmentation network (Sec. 4.4, Fig. 3 bottom): after a spatial transformer, three EdgeConv layers; a shared fully connected layer (1024) aggregates them; shortcut connections include all EdgeConv outputs as local descriptors, concatenated with the global descriptor onto every point; then three shared fully connected layers (256, 256, 128). The per-layer widths of the three segmentation EdgeConv layers, the k for segmentation, and the spatial-transformer internals are in Fig. 3, an image, and are not in the running text.

Size and speed (Table 3, ModelNet40 classification, 1024 points): the fixed-graph baseline is 11 MB, 19.7 ms, 91.7%; the full dynamic model is 21 MB, 27.2 ms, 92.9%. For comparison PointNet++ is 12 MB, 163.2 ms, 90.7%. A parameter count is not given, only model size.

Training: SGD, learning rate 0.1 reduced to 0.001 by cosine annealing, momentum 0.9, batch size 32; k and other hyperparameters chosen on an 80/20 train/validation split.

## Training data and labels
- ModelNet40 (9,843 train, 2,468 test models), ShapeNet part (16,881 shapes, 16 categories, 50 parts), S3DIS (6 areas, 272 rooms, 13 classes, 6-fold cross validation).
- Labels are class or part or semantic annotations. No loss function is stated anywhere in the paper text.

## Headline results
- ModelNet40 overall accuracy (Table 2): 92.9 at 1024 points, 93.5 at 2048 points; mean class accuracy 90.2 and 90.7. The fixed-graph baseline gets 91.7 overall, which is 1.0 point above PointNet++ (90.7).
- ShapeNet part mean IoU (Table 6): 85.2, against PointNet 83.7 and PointNet++ 85.1.
- S3DIS mean IoU (Table 7): 56.1, overall accuracy 84.1, against PointNet 47.6 and 78.5.
- Ablation (Table 4, ModelNet40 classification): using x_i together with x_j - x_i ("centralisation") adds about 0.5 points overall accuracy; recomputing the graph dynamically adds about 0.7; 2048 points add 0.6 more. Neighbours (Table 5, 1024 points): k = 5, 10, 20, 40 give 90.5, 91.4, 92.9, 92.4 overall accuracy. These are classification numbers, not per-point results.

## How it differs from our model
- Almost identical in spirit. Our three EdgeConv layers use the paper's asymmetric edge function [f_i, f_j - f_i], max over k neighbours, shortcut concatenation of all layer outputs, and a global max feature copied back to every point. Our design is closest to the paper's own "baseline" row (fixed graph from input proximity), plus centralisation.
- It has: dynamic graph recomputation in feature space (ours is static, computed once on xyz), a spatial transformer, a 1024-wide aggregation layer, bigger widths (up to 256), LeakyReLU with batch norm, k = 20.
- Ours has: a two-layer edge MLP with LayerNorm and GELU (the paper's Eq. 8 is one linear map plus ReLU; its Fig. 3 block allows an MLP, with widths I could not read), a width of only 64 to 128, and a much smaller model (about 0.2 M parameters; at float32 that is roughly 0.8 MB, by my arithmetic, against 11 to 21 MB here).
- Density: the paper raised k from 20 to 40 when the points doubled "to maintain the same density". We keep k = 16 at 2048 (training) and 4096 (evaluation), so our evaluation neighbourhoods cover a smaller patch (about 0.7 of the radius on a surface, by my own estimate).
- Partial views: DGCNN's Fig. 9 removes parts of a shape from one of six sides and reports mean IoU against keep ratio, which is the closest published test to our camera variant.
- Plausible impact of switching to the full DGCNN for per-point heatmaps on single objects of 2k to 8k points: small to moderate. A dynamic graph is the one part we lack, and the paper gives about 0.7 points for it on classification, not on per-point tasks. The cost is an N by N distance matrix per layer; at 4096 points that is easy. It is a cheap, worthwhile ablation.

## Likely supervisor questions
1. Q: Your model looks like DGCNN. What is different? A: The edge function and the three-layer layout with concatenation and global max are the same idea. We fix the k = 16 graph once on xyz instead of recomputing it in feature space, we add LayerNorm and GELU and a second linear layer per edge, and we are much smaller (about 0.2 M parameters).
2. Q: Why a static graph? A: It is cheaper and the neighbourhoods stay geometric, which matters for physical quantities such as contact. The paper's own ablation says a dynamic graph adds about 0.7 points on ModelNet40 classification, so we may be leaving a small gain; we should say we have not tested it on our labels.
3. Q: Why k = 16 at both 2048 and 4096 points? A: The paper doubled k from 20 to 40 when going from 1024 to 2048 points to keep the same density, and found larger k can hurt (Table 5). Our fixed k means evaluation neighbourhoods are smaller than at training. We should report accuracy with N = 2048 at evaluation, or scale k, to show it does not matter.

## Not verified
- Segmentation EdgeConv layer widths, the segmentation k, and spatial-transformer internals (only in the Fig. 3 image).
- Any loss function: none stated in the paper text.
- Parameter count: only model size in MB.
- The text "three shared fully-connected layers (64, 64, 128, 256)" lists four numbers; reported as written.
- Venue volume, issue and pages: the PDF shows a placeholder "Vol. 1, No. 1, Article 1".
- The 0.7 of the radius estimate for a changed point count is my own arithmetic, not from the paper.

## BibTeX
```bibtex
@article{wang_2019_dgcnn,
  title   = {Dynamic Graph {CNN} for Learning on Point Clouds},
  author  = {Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  journal = {ACM Transactions on Graphics},
  year    = {2019},
  doi     = {10.1145/3326362},
  eprint  = {1801.07829},
  archivePrefix = {arXiv}
}
```

## Verification (second reader, 2026-11-16)
Checked against the extracted text of arXiv 1801.07829. Confirmed in the text: the adopted edge function h(x_i, x_j - x_i) (Eq. 7 and 8); k = 20 for all EdgeConv layers at 1,024 points and k = 40 at 2,048 points "to maintain the same density"; the fixed-graph baseline is "1.0% better than PointNet++"; Table 4 overall accuracy 91.7, 92.2, 92.9, 93.5, so centralisation adds about 0.5, the dynamic graph about 0.7 and 2,048 points 0.6; Table 5 values for k = 10, 20, 40. Our design (fixed graph plus centralisation) matches the 92.2 row of Table 4, not the 91.7 baseline row. Not checked: the segmentation results, Table 3 sizes and timings, the training settings. Verdict: safe to quote for the points listed as confirmed.
