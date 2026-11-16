# 3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding
**Authors**: Shengheng Deng, Xun Xu, Chaozheng Wu, Ke Chen, Kui Jia
**Year / Venue**: 2021, CVPR 2021 (arXiv v2, 31 Mar 2021)
**Cite key**: deng_2021_3daffordancenet
**Source read**: https://arxiv.org/abs/2103.16397 (abstract page); https://arxiv.org/pdf/2103.16397 (full PDF including appendices, text extracted locally; main text and appendix E read in full, other appendices skimmed)

## In one paragraph
3D AffordanceNet is the human-annotated counterpart to the simulation papers. It gives 22,949 PartNet shapes from 23 categories a per-point score for up to 18 affordance types such as grasp, support, contain and pour. Annotators click a few keypoints per affordance on a 3D viewer, and label propagation on a k-nearest-neighbour graph spreads those clicks into a smooth score between 0 and 1 for every point. The paper then benchmarks three point cloud networks on full shapes, partial views and rotated shapes, and a semi-supervised setting. The labels say where a human would use a part, not whether a physical interaction succeeds. This is useful as a contrast: a human-defined "support" map has no object being supported and no physics.

## Input and output
- Input: a point cloud of a shape. For the partial-view task, 2048 points sampled by furthest point sampling from the points visible to one of 4 affine cameras placed at (1,1,1), (-1,-1,1), (1,-1,-1), (-1,1,-1) looking at the origin (shapes are aligned in the cube from (-1,-1,-1) to (1,1,1); clouds with fewer than 2048 points are up-sampled). Number of points for full-shape input: not verified in the paper text.
- Output: for each point and each of the 18 affordance types, a score in [0, 1] (multi-label: one point can have several). Each shape has at most 5 affordance types defined by its category.
- Evaluation binarises the ground truth at 0.5.

## Architecture
- Three benchmark backbones, used as shared per-point feature extractors:
  - PointNet++ (segmentation branch), 128-d point features, Adam, learning rate 0.001, halved every 20 epochs, 200 epochs, batch size 16, weight decay 1e-8;
  - DGCNN (segmentation branch), 256-d point features, SGD, learning rate 0.1, momentum 0.9, weight decay 1e-4, cosine annealing;
  - U-Net, initialised from PointContrast pre-trained weights and fine-tuned (the text does not say more about its layers, so I do not claim it is a sparse-convolution network).
- One classification head per affordance type (18 in total, same shape, different weights): FC(m, 128) with BatchNorm, then FC(128, 1), sigmoid, where m is the backbone feature size (128 or 256).
- Loss: cross-entropy plus a dice loss (to counter the many zero-score points), l = l_CE + l_DICE.
- Semi-supervised variant (DGCNN backbone): Virtual Adversarial Training. 1 % of the training data is labelled, the rest is unlabelled. The consistency loss is the MSE between predictions on the clean cloud and on an adversarially perturbed cloud. Batches of 16: 8 labelled and 8 unlabelled.
- The backbone DGCNN is the closest published relative of our network: EdgeConv on k-nearest-neighbour graphs. Ours uses a static graph on xyz, 3 layers (64, 64, 128), whereas the benchmark DGCNN settings (layer widths, k) are not given in the text I read (not verified).

## Simulation and labels
- No simulation. Labels are human annotations.
- Data: shapes from PartNet (itself built on ShapeNet and 3D Warehouse): 22,949 shapes, 23 categories, 18 affordance classes, 56,307 affordance annotations. The class distribution is very skewed: tables 7,990 shapes and chairs 6,113, versus scissors 68.
- Annotation: 42 professional annotators, about 2 minutes per shape, each shape annotated by 3 annotators. A web 3D GUI asks questions per affordance such as "if you want to put something on the table, at which points would you put the object?" (support). Annotators click at least 3 keypoints per affordance and may mark adjacent parts to which the affordance extends.
- Ground truth: a k-nearest-neighbour graph on N points sampled densely on the mesh; adjacency a_ij = distance for neighbours; symmetrised and degree-normalised matrix W~; scores S = (I - alpha W~)^(-1) Y with alpha = 0.998, then linear normalisation to [0, 1]. Propagation is restricted to the parts the annotator selected. k and N are not stated in the text I read (not verified).
- Annotators' agreement: reported as the average variance over the 3 annotators per affordance (appendix B, small values reported; I did not copy the numbers).

## Evaluation and headline results
- Metrics: mean average precision (mAP), AUC, average IoU over thresholds 0 to 0.99, and MSE. All metrics except MSE are macro-averaged over shapes per category.
- Split: 70 / 10 / 20 train, validation, test, made "according to the shape semantic category" (I read this as a per-category ratio split by shapes, with no category held out).
- Full-shape, average over affordances (mAP / AUC / aIoU): PointNet++ 48.0 / 87.4 / 19.3, DGCNN 46.4 / 85.5 / 17.8, U-Net 47.4 / 86.3 / 19.7.
- Support affordance, full shape: PointNet++ mAP 52.5, AUC 89.7, aIoU 17.5.
- Partial view (4 views, 2048 points): PointNet++ 45.7 / 85.2 / 16.9, DGCNN 42.2 / 83.7 / 13.8.
- Rotation: SO(3)/SO(3) rotation lowers PointNet++ to 41.8 mAP and 83.3 AUC.
- Semi-supervised with 1 % labels beats the fully supervised baseline trained on the same 1 %.
- Networks underfit even the training set (their Table 5), so the authors call the benchmark challenging.
- No robot and no real-world test.

## How it differs from our setup
- Label source: human clicks plus graph propagation, versus physics outcomes. Human labels express "where would a person use this", ours express "does this grasp or this drop work".
- Pair structure: affordance here is a property of one object alone. "Support" has no supported object, so the same table gets one support map regardless of whether a plate or a piano is put on it. Our support test has object A and object B and the label changes with A. This is the main conceptual contrast.
- Smoothness: propagated labels are smooth in [0, 1] and each affordance has about 3 clicks per annotator. Our labels are binary, sharp at boundaries and dense.
- Dataset size: 22,949 shapes and 56,307 annotations, versus our 185 objects for grasp and about 20,000 object pairs. Our grasp model has much less shape diversity.
- Backbone: they test PointNet++, DGCNN and a sparse U-Net, one head per affordance; ours is a small EdgeConv network. They show that DGCNN-type networks work on per-point affordance, which supports our backbone choice, but PointNet++ was slightly better on their benchmark (AUC 87.4 versus 85.5).
- Evaluation: per-shape macro mAP, AUC, aIoU on binarised smooth labels, versus our per-point AUC on held-out objects. Both use AUC, so the metric family is shared, but neither the labels nor the splits are the same.
- Worth copying: the dice loss, which helps when positives are rare per shape (relevant for "stays level" positives); the partial-view protocol of several fixed cameras with 2048-point FPS (it is a clean, reproducible way to turn a complete cloud into a partial one for our camera route); and the per-category breakdown table.

## Likely supervisor questions
1. Why simulate instead of annotate? Annotation tells where humans would grasp, not what a given gripper holds, and it ignores the second object. Our labels are exhaustive and physical, and cost compute, not annotators. The cost is that they only capture what the simulator models.
2. Are your AUC values comparable to theirs? No. Theirs average 0.855 to 0.874 (85.5 to 87.4 %, full shapes, three backbones) on smooth human-defined maps over 23 categories. Ours are on binary physical labels on 185 objects, held out by object. The higher value says nothing about being better.
3. Does your backbone have precedent? Yes. DGCNN is one of their three baselines for per-point affordance estimation, with a shared backbone and per-affordance heads. Ours is a smaller static-graph version.

## Not verified
- Number of points for full-shape input and for the label propagation (N), and the k of the kNN graph.
- DGCNN configuration details (layer widths, k) beyond the 256-d output feature.
- Annotator agreement values (appendix B numbers not copied).
- Whether the train/test split holds out any category (the text says split "according to the shape semantic category", which I interpret as per-category ratios).
- Parameter counts.

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (our 185 objects and 20,000 pairs). Which table row or column a number belongs to was not re-checked; quote those with the table open. The comparison range in question 2 was loose (0.84 to 0.88) and is now 0.855 to 0.874, the three backbones' AUC averages.

## BibTeX
```bibtex
@inproceedings{deng20213daffordancenet,
  title     = {{3D AffordanceNet}: A Benchmark for Visual Object Affordance Understanding},
  author    = {Deng, Shengheng and Xu, Xun and Wu, Chaozheng and Chen, Ke and Jia, Kui},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  eprint    = {2103.16397},
  archivePrefix = {arXiv}
}
```
