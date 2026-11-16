# PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers
**Authors**: Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, Jie Zhou (Tsinghua University)
**Year / Venue**: 2021, ICCV 2021 (oral, per the arXiv abstract page). arXiv v1 dated 19 August 2021.
**Cite key**: yu_2021_pointr
**Source read**: https://arxiv.org/abs/2108.08839 (abstract page, fetched). Full text: https://arxiv.org/pdf/2108.08839 (downloaded, converted with pdftotext; main text, implementation appendix and complexity table read; KITTI table checked in layout mode).

## In one paragraph
PoinTr completes a partial point cloud, such as the cloud you get from one depth view. The idea is to treat completion as translation between two sets. The partial cloud is cut into local groups ("point proxies"), each described by a feature vector and a position. A transformer encoder reads the proxies of what is seen, and a transformer decoder produces proxies for what is missing. A small FoldingNet turns each predicted proxy into a patch of points. The observed points are kept, and the predicted missing points are added to them. The authors also built two harder benchmarks (ShapeNet-55 and ShapeNet-34) with many categories, many viewpoints and 25 to 75 percent missing points. For us it is the natural "partial RGB-D view, then complete it" module between route (b) and the heatmap network.

## Input and output
- Input: a partial point cloud, 2048 points, xyz only. No colour, no camera, no RGB image. On PCN the input is also 2048 points.
- Output: the missing part, concatenated with the input. ShapeNet-55 and ShapeNet-34: 6144 predicted points, so 8192 in total. PCN: 14336 predicted, 16384 in total.
- Scale and pose: the output lives in the input's coordinate frame, so it inherits the input's scale and pose. The paper does not say how coordinates are normalised for training (unit sphere, centred, canonical orientation). Not verified. ShapeNet models come in a canonical orientation, so a learned model may rely on it. This is a point to check before using it on arbitrary tabletop poses.

## Architecture
- Point proxies: farthest point sampling picks N = 128 centres. A light DGCNN (Linear 3 to 8, then four DGCNN layers with 32, 64, 64, 128 channels; Appendix A says k = 16 for the DGCNN extractor, but the layer list in the same appendix shows K = 8) gives local features. A position MLP adds the location of each centre.
- Transformer: encoder with 6 layers and decoder with 8 layers (Appendix A; Appendix B says N = 6 for both), 6 heads, hidden size 384. Geometry-aware block: a kNN (k = 8 in Appendix A) plus max-pool branch is added next to attention, but only in the first block of encoder and decoder. Ablation: using it in every block over-fits slightly.
- Query generator: max-pools the encoder output, predicts M centre coordinates (M = 96 for ShapeNet-55 and ShapeNet-34, 224 for PCN) and builds dynamic query embeddings from them.
- Output: an MLP and FoldingNet turn each predicted proxy into local points around its centre.
- Size: 30.9M parameters and 10.41 GFLOPs (Table 8). Inference time in seconds: not stated.

## Training data and loss
- ShapeNet-55: all 55 categories, 80/20 split by object: 41,952 training and 10,518 test models, 8,192 points per object.
- ShapeNet-34: 34 seen and 21 unseen categories. 46,765 training models; seen test 3,400 objects; unseen test 2,305 objects.
- PCN dataset: the standard 8-category benchmark.
- Partial clouds on ShapeNet are made online: pick a random viewpoint, delete the n farthest points (n from 2048 to 6144 of 8192, i.e. 25 to 75 percent), downsample to 2048. The authors note that projecting a depth view (as in PCN) is closer to real scans, but they choose this cheaper rule.
- Loss: Chamfer distance, on both the predicted centres and the final points, summed. AdamW, learning rate 5e-4, 200 to 300 epochs.

## Headline results and known failure cases
- ShapeNet-55 (CD-l2 times 1000, average of simple, moderate and hard): 1.09 for PoinTr in Table 1 (1.07 in Table 8) vs GRNet 1.97. F-score at 1 percent: 0.464 vs 0.238.
- ShapeNet-34: seen categories average CD 1.23, unseen 2.05 (hard setting: 1.88 seen vs 3.44 unseen). F-score 0.421 seen, 0.384 unseen. So novel categories cost accuracy, more so with more missing points.
- PCN (CD-l1 times 1000): 8.38 average vs CRN 8.51, GRNet 8.83.
- Real data: only KITTI LiDAR cars (after fine-tuning on ShapeNet cars). MMD 0.526 vs GRNet 0.568. The fidelity score of 0.000 is presumably trivial, because the observed points are copied into the output (my reading, not stated in the paper).
- The paper has no explicit limitations section. Thin, flat or hollow objects, and sensor noise on tabletop objects, are not tested. Not verified.

## How it relates to our camera route
- It fits route (b), not route (c). Input is the real partial cloud from the lab RGB-D camera. Output is a completed cloud that keeps every measured point. It is a safer way of adding the hidden side than a generative image-to-3D model, because the visible part stays exact.
- It does not need size or pose from us if the input is metric, but it may expect normalised, canonical input. The honest test must give it the raw sensor cloud (with sensor noise and holes) after the same normalisation we will use in deployment.
- Possible gain is bounded. Our own numbers: mesh 0.88, one RGB-D view 0.86. Completion can recover at most about 0.02 AUC for grasp. It may matter more for the "place something on it" heatmap, which depends on the hidden top and bottom. That is an open question.
- Training data are ShapeNet. Our kitchen objects (pans, woks, trays) may be out-of-distribution; the paper shows an accuracy drop on unseen categories. We would probably need to fine-tune on our own meshes with simulated one-view clouds, which we can make in simulation.

## Likely supervisor questions
1. Why add shape completion if one RGB-D view already gives 0.86 AUC? Answer: the ceiling from a full mesh is 0.88, so for grasping the possible gain is small. Completion matters mainly for heatmaps that depend on hidden surfaces. We can test that directly.
2. Does PoinTr keep our measured points? Answer: yes. It only predicts the missing part and concatenates it. That makes it safer than a generative model.
3. Will it work on our objects? Answer: it was trained on ShapeNet and loses accuracy on unseen categories. We would fine-tune on simulated partial views of our own meshes. Real sensor noise is the other risk, since the paper tests only LiDAR cars.

## Not verified
- Coordinate normalisation used in training and whether canonical orientation is assumed, inference time, behaviour on thin or flat tabletop objects, and results on noisy RGB-D data.
- Internal inconsistencies in the paper: DGCNN neighbours k = 16 (Appendix A text) vs K = 8 (Appendix A layer list); decoder depth 8 (Appendix A) vs N = 6 for encoder and decoder (Appendix B). I quote both and do not know which one the released model uses.
- The exact column alignment of Tables 1 and 2 (taken from text extraction; the KITTI table was checked in layout mode).

## BibTeX
```bibtex
@inproceedings{yu2021pointr,
  title     = {{PoinTr}: Diverse Point Cloud Completion with Geometry-Aware Transformers},
  author    = {Yu, Xumin and Rao, Yongming and Wang, Ziyi and Liu, Zuyan and Lu, Jiwen and Zhou, Jie},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  note      = {arXiv:2108.08839}
}
```
