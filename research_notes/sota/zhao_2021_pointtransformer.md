# Point Transformer
**Authors**: Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun
**Year / Venue**: 2021, ICCV. Not printed in the PDF text and the arXiv page has no comments or journal field; venue taken from the task brief. arXiv v1 16 Dec 2020, v2 26 Sep 2021.
**Cite key**: zhao_2021_pointtransformer
**Source read**: https://arxiv.org/abs/2012.09164 (metadata only); https://arxiv.org/pdf/2012.09164 (full text, v2). The fetch summariser could not read the binary PDF, so the text was extracted locally with pdftotext from the saved PDF.

## In one paragraph
Self-attention treats its input as a set, which makes it a natural operator for point clouds. Point Transformer applies attention locally: each point attends only to its 16 nearest neighbours. It uses vector attention, where the attention weight is a vector that modulates every feature channel separately, not one scalar per neighbour. The weight is computed from the difference between the features of the centre point and a neighbour, plus a learned encoding of their relative 3D position. The position encoding is added both to the attention weights and to the values. These layers are stacked in a U-Net with five stages that downsample the points by farthest point sampling. The result set new records on S3DIS, ModelNet40 and ShapeNetPart.

## Input and output
- Input: points with 3D coordinates p and features x. For ModelNet40 the paper samples points together with normal vectors from the meshes. The number of input points per S3DIS block or per ModelNet40 shape is not stated in the text I read; an appendix reports timing for 10k, 20k, 40k and 80k input points.
- Output: per-point logits (semantic segmentation, part segmentation) through a final MLP; or class logits (classification) from global average pooling plus an MLP.

## Architecture
Point transformer layer (Eq. 3): y_i = sum over x_j in the k nearest neighbours X(i) of softmax( gamma( phi(x_i) - psi(x_j) + delta ) ) (element-wise product) ( alpha(x_j) + delta ).
- phi, psi, alpha: linear projections. The relation is subtraction. gamma: an MLP with two linear layers and one ReLU, which outputs a vector of attention weights. The normalisation is softmax over the neighbours.
- Position encoding delta = theta(p_i - p_j), theta an MLP with two linear layers and one ReLU, trained end to end, used in both branches.
- Neighbourhood: k = 16 throughout.

Block and network (Fig. 3, Fig. 4):
- Point transformer block: linear, point transformer layer, linear, with a residual connection.
- Five stages with downsampling rates [1, 4, 4, 4, 4], so the point counts are N, N/4, N/16, N/64, N/256. Channel widths read from the figure labels: 32, 64, 128, 256, 512. The number of blocks per stage is not stated in the text I read.
- Transition down: farthest point sampling picks the subset; features go through linear, batch norm, ReLU, then max pooling over the k nearest neighbours onto each kept point.
- Transition up: linear, batch norm, ReLU, then "trilinear interpolation" onto the denser point set, then combined with the encoder features from the skip connection (the paper's word is "summarized"; it does not say whether this is a sum).
- Output head: MLP for segmentation. For classification, global average pooling (to a 1 x 512 vector) then an MLP.
- Parameters: 4.9 M for the S3DIS segmentation model (stated in the text, against 14.9 M for KPConv and 30.1 M for SparseConv).
- Inference (appendix, one Quadro RTX 6000): 44 ms, 86 ms, 222 ms, 719 ms for 10k, 20k, 40k, 80k points.

Training: SGD, momentum 0.9, weight decay 0.0001. S3DIS: 40K iterations, learning rate 0.5 with 10x drops at 24K and 32K. ModelNet40 and ShapeNetPart: 200 epochs, learning rate 0.05 with drops at epochs 120 and 160.

## Training data and labels
- S3DIS (271 rooms, 13 classes; Area 5 held out, or 6-fold), ModelNet40 (9,843 train, 2,468 test), ShapeNetPart (16,880 models, 50 parts; 14,006 train, 2,874 test). Point sets for ShapeNetPart follow the PointNet++ release.
- Loss: not stated in the text I read. The paper notes that it did not use loss balancing, which can raise category mIoU on ShapeNetPart.

## Headline results
- S3DIS Area 5 (Table 1): mIoU 70.4, mAcc 76.5, OA 90.8, the first model above 70% mIoU, 3.3 points above the previous best (MinkowskiNet 65.4, KPConv 67.1 are in the same table). 6-fold: mIoU 73.5.
- ModelNet40 overall accuracy 93.7 (Table 3), against DGCNN 92.2.
- ShapeNetPart instance mIoU 86.6 (Table 4), against PointNet 83.7, PointNet++ 85.1, DGCNN 85.1.
- Ablations on S3DIS Area 5 (mIoU): k = 4, 8, 16, 32, 64 give 59.6, 67.7, 70.4, 68.3, 67.7; no position encoding 64.6, absolute 66.5, relative 70.4; attention operator: MLP 61.7, MLP plus pooling 63.7, scalar attention 64.6, vector attention 70.4.

## How it differs from our model
- It has: attention-weighted aggregation (per-channel weights from the feature difference and the relative position) instead of an unweighted max, a learned relative position encoding, a five-stage downsampling hierarchy with skip connections, and 4.9 M parameters.
- Ours has: a flat resolution, a plain max over 16 neighbours of an MLP of [f_i, f_j - f_i], and about 0.2 M parameters. We also use cross-attention (4 heads, width 128) between two objects in the pair variant; that is a different use of attention from this paper's local self-attention.
- Same k: both use k = 16 neighbours, and the paper finds k = 16 best on S3DIS and larger k hurts (Table 5).
- The closest no-attention baseline in their ablation is "MLP plus pooling" (a pointwise MLP followed by max pooling inside each kNN neighbourhood), 63.7 against 70.4 for vector attention on S3DIS Area 5. It resembles but is not identical to EdgeConv, which feeds the neighbour offset into the MLP before the max. Also, the vector against scalar attention result is for local self-attention on S3DIS segmentation, and should not be stretched to our cross-attention blocks.
- Benchmarks here are multi-class softmax segmentation. Ours is per-point sigmoid binary cross-entropy on simulated success labels, so rankings may not transfer.
- Plausible impact of switching for per-point heatmaps on single objects of 2k to 8k points: possible gain from attention-weighted aggregation and relative position encoding (6.7 mIoU over their pooling baseline on S3DIS scenes), but the gap was measured on large indoor scenes with many classes; a 0.2 M parameter model is also much cheaper to train on thousands of simulated pairs. Worth a small ablation, not a rewrite.

## Likely supervisor questions
1. Q: Why max over neighbours rather than attention? A: Max over an MLP of [f_i, f_j - f_i] is cheap and has few parameters (0.2 M). Point Transformer's own ablation shows attention beats a pooling baseline by about 6.7 mIoU on S3DIS scenes, so we may be leaving accuracy on the table; we have not tested it on our labels.
2. Q: Do you use position information? A: Our edge function uses feature differences, and the first layer has xyz plus normals as its features, so the first layer's difference is the relative position. Point Transformer found relative position encoding in both the attention and the value branch matters most (70.4 against 64.6 with none).
3. Q: Why no downsampling? A: Objects of 2k to 8k points are small enough to process at full resolution, and a heatmap needs a value for every point. Point Transformer needs transition-up interpolation to get back to dense points; we avoid that blur.

## Not verified
- Venue (ICCV 2021): not in the PDF text or the arXiv metadata; taken from the brief.
- Loss function: not stated.
- Number of blocks per stage: not stated; the channel widths 32 to 512 come from figure labels.
- Input point count per sample for S3DIS and ModelNet40: not stated in the text I read.

## Verification (second read, 2026-11-16)
Checked against the ar5iv full text by a second reader. Confirmed: Eq. 3 and the two-layer MLPs, k = 16, the rates [1, 4, 4, 4, 4], transition down, 4.9 M parameters (KPConv 14.9 M, SparseConv 30.1 M), all numbers of Tables 1 to 7 quoted above, the SGD settings and the inference times.
Corrected above: the paper does not say the skip connection is summed.
Still not in the text: channel widths (figure only), blocks per stage, the loss, the input point count, the venue.

## BibTeX
```bibtex
@inproceedings{zhao_2021_pointtransformer,
  title     = {Point Transformer},
  author    = {Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip and Koltun, Vladlen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  eprint    = {2012.09164},
  archivePrefix = {arXiv}
}
```
