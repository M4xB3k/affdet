# Depth Anything V2
**Authors**: Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao (HKU, TikTok)
**Year / Venue**: 2024, NeurIPS 2024 (38th Conference on Neural Information Processing Systems; printed in the PDF). arXiv v1 June 2024, v2 20 October 2024.
**Cite key**: yang_2024_depthanythingv2
**Source read**: https://arxiv.org/abs/2406.09414 (abstract page, fetched). Full text: https://arxiv.org/pdf/2406.09414 (v2, downloaded, converted with pdftotext; main text, Tables 2 to 4 and the limitations section read).

## In one paragraph
Depth Anything V2 predicts a depth map from a single colour image, for any kind of scene. The paper's message is that data matters more than tricks. Real depth labels are noisy (glass, mirrors, thin parts), so the authors train a large "teacher" only on precise synthetic images. The teacher then labels 62 million unlabelled real images, and smaller "student" models learn from those pseudo-labels. The result has sharp edges, handles thin structures and transparent things better than V1, and is much faster than diffusion-based depth models. The main model outputs relative depth, not metres. Metric versions are made by fine-tuning on indoor and outdoor synthetic data.

## Input and output
- Input: one RGB image. Training resolution 518 x 518 (shorter side resized to 518, random crop). No camera intrinsics needed.
- Output of the main models: a per-pixel map of affine-invariant inverse depth. This means depth is only known up to an unknown scale and shift, and it is stored as inverse depth (disparity-like), not in metres.
- Metric models: the paper fine-tunes the encoder to metric depth, in two ways. (1) In-domain on NYU-D and KITTI with the ZoeDepth pipeline, for benchmark numbers. (2) On synthetic Hypersim (indoor) and Virtual KITTI (outdoor) for released metric models. These are scene-level, trained for rooms and streets, not tabletop objects.
- Scale and pose: relative output has neither metric scale nor an object-centred pose. It sees one side of the surface only.

## Architecture
- Encoder: DINOv2 (Small, Base, Large, Giant). Decoder: DPT, as in V1.
- Sizes: 25M to 1.3B parameters. Figure 1 (checked on the rendered page) gives Small 25M at 60 ms latency and Large 335M at 213 ms (V100 GPU). The Base size is not given in the text I read.
- Teacher: DINOv2-Giant trained purely on synthetic images. Students: S, B, L, G trained on pseudo-labelled real images only.

## Training data and loss
- Teacher data: five synthetic datasets, 595K images (includes Hypersim and Virtual KITTI). Student data: eight pseudo-labelled real datasets, 62M images. The top 10 percent largest-loss regions per pseudo-labelled sample are ignored as likely noise.
- Losses on labelled synthetic images: scale-and-shift-invariant loss plus gradient matching loss (weight ratio 1:2). On pseudo-labelled images an extra feature alignment loss keeps the DINOv2 semantics.
- Schedule: teacher batch 64 for 160K iterations; students batch 192 for 480K iterations; Adam, learning rate 5e-6 (encoder) and 5e-5 (decoder).

## Headline results and known failure cases
- DA-2K, a new benchmark of 1K images with 2K sparse pairs ("which of two pixels is closer"): accuracy 95.3 (Small), 97.0 (Base), 97.1 (Large), 97.4 (Giant). Marigold 86.8, GeoWizard 88.1, DepthFM 85.8, Depth Anything V1 88.5 (model size not given in the table). This measures ordering only, not metric accuracy.
- Zero-shot relative depth, standard benchmarks: V2 is "comparable" to V1, not clearly better. NYU-D AbsRel 0.045 (Large) and 0.044 (Giant); KITTI 0.074 (Large). The authors say standard benchmarks have noisy labels and do not show V2's strengths.
- Metric depth, in-domain fine-tuning (ZoeDepth pipeline): NYU-D AbsRel 0.056 (Large), KITTI AbsRel 0.045 (Large). For scale: 5.6 percent relative error at a 0.7 m working distance is about 4 cm (my arithmetic, not a paper number). That is larger than the thickness of a plate or tray.
- Failure cases and limits the paper states: training with 62M images is very costly; the synthetic training sets are not diverse enough; the synthetic-only teacher failed on sky and on a human head (Figure 6). V1 was weak on transparent and reflective things. V2 scores 83.6 percent on a transparent-surface challenge, zero-shot.
- Objects on a table are not a benchmarked case in the paper. Not verified.

## How it relates to our camera route
- This is route (d). Our 0.54 AUC is close to chance. This note gives a likely reason, which we should check first: the output is relative inverse depth. If the point cloud was back-projected without recovering the correct scale and shift, the shape is distorted by an unknown amount. That alone can destroy a per-point heatmap, without any "model quality" problem. (Hypothesis. I do not know how the cloud was built.)
- Even with perfect scale and shift, a single depth map is only a one-view surface, like the partial RGB-D cloud in route (b), but noisier. So route (b) should normally beat route (d), and it does in our results (0.86 vs 0.54). Note that our route (b) used a perfect rendered view. Real sensor depth has holes at reflective and low-texture spots and has not been tested yet.
- The lab RGB-D camera gives metric depth directly. A separate study (Nolte et al. 2025) notes that depth resolves location and scale ambiguities, and that low-cost RGB-D depth has holes at reflective and low-texture spots. Depth Anything could be useful for filling such holes, but needs a scale and shift fit to the valid sensor pixels. That is a possible extension, not a result.
- A fair test: fit scale and shift per image against the true depth (an upper bound) and against sparse sensor depth (realistic), then compare AUC with route (b).

## Likely supervisor questions
1. Why did monocular depth do so badly (0.54)? Answer: the model outputs relative inverse depth, not metres, and only one surface. We should check scale and shift recovery first. Even if fixed, it cannot beat a real depth sensor.
2. Is the metric version of Depth Anything the fix? Answer: it is fine-tuned on rooms and streets (Hypersim, Virtual KITTI). Its AbsRel is a few percent of distance, which is centimetres at tabletop range. That is too coarse for thin objects.
3. Where could it still help? Answer: as a fill-in for missing sensor depth, or when only RGB exists. Both need alignment to valid sensor pixels first.

## Not verified
- Base model parameter count, behaviour on tabletop objects, how our 0.54 AUC cloud was built, and metric accuracy of the released Hypersim and Virtual KITTI models (the paper shows only qualitative comparison to ZoeDepth).

## BibTeX
```bibtex
@inproceedings{yang2024depthanythingv2,
  title     = {Depth Anything {V2}},
  author    = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2406.09414}
}
```
