# InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models
**Authors**: Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, Ying Shan (ARC Lab Tencent PCG; ShanghaiTech University)
**Year / Venue**: 2024, arXiv technical report (cs.CV), v1 10 April 2024, v2 14 April 2024. Not a peer-reviewed venue as far as the fetched pages show.
**Cite key**: xu_2024_instantmesh
**Source read**: https://arxiv.org/abs/2404.07191 (abstract page, fetched). Full text: https://arxiv.org/pdf/2404.07191 (v2, downloaded, converted with pdftotext; full 8-page report read; the results page with Tables 2 to 4 viewed as an image to confirm the columns).

## In one paragraph
InstantMesh does not reconstruct from the one image directly. First a multiview diffusion model (Zero123++) imagines six views of the object at fixed camera angles. Then a sparse-view LRM-style transformer turns those six views into a triplane, and a differentiable mesh extractor (FlexiCubes) makes the final mesh. Because the final stage works on a mesh, the authors can supervise it with depth and normal maps, which gives smoother surfaces. The whole thing takes about 10 seconds. The invented views are what give it "imagination" about the back of the object, but they can be inconsistent with each other. It beat TripoSR and others on GSO and OmniObject3D in the authors' own tests.

## Input and output
- Input: one RGB image. The diffusion step makes a 960 x 640 image holding 6 views in a 3 x 2 grid, so each view is 320 x 320. The reconstruction model reads the 6 views at 320 x 320 with fixed, known camera poses.
- Camera needed: none from the user. The six view poses are fixed by Zero123++ (elevations alternate 20 and -10 degrees, azimuths start at 30 degrees relative to the input and step by 60 degrees).
- Output: a triplane (64 x 64) decoded to an SDF, then a FlexiCubes mesh on a 128^3 grid (the "Mesh" variants). "NeRF" variants output a density field instead.
- Scale and pose: the reconstruction model works in a canonical world space whose z-axis points against gravity, unlike view-space models such as TripoSR. So the up axis is meaningful. Metric size is not recovered: for evaluation the authors rescale all meshes into a [-1, 1]^3 cube. How the yaw is fixed relative to the input image is not stated. Not verified in the paper text.

## Architecture
- Stage A: Zero123++ multiview diffusion model, fine-tuned for 1000 steps (learning rate 1e-5, batch 48) on the LVIS subset of Objaverse to output white backgrounds instead of grey. This removes floaters.
- Stage B: sparse-view reconstruction model, initialised from OpenLRM. A ViT image encoder with adaptive layer norm for the camera pose, then a transformer triplane decoder.
- Variants (Table 1 of the paper): base = 12 transformer layers, triplane dim 40; large = 16 layers, triplane dim 80. All have transformer width 1024, 6 input views, 64 x 64 triplane, input size 320. NeRF variants render 192; mesh variants render 512.
- Mesh head: the density MLP is reused as an SDF predictor, with two more MLPs for the FlexiCubes deformation and weights. The last SDF layer is initialised as the negated density layer so training starts stable.
- Parameter count and GPU used for the 10-second figure: not stated in the paper text.

## Training data and loss
- Data: Objaverse, filtered from about 800k to about 270k objects (must have textures, fill at least 10 percent of the view, a single object, a Cap3D caption, and not low-poly). 32 random views per object at 512 x 512, plus depth and normal maps. Training uses 6 random input views and 4 supervision views.
- Stage 1 (NeRF): image loss, LPIPS (weight 2.0) and mask loss (weight 1.0); learning rate 4e-4 cosine to 4e-5. Renders 192 x 192 patches.
- Stage 2 (mesh): adds depth loss (0.5), normal loss (0.2) and a FlexiCubes regulariser (0.01); learning rate 4e-5 cosine to 0. Both stages on 8 H800 GPUs.
- Robustness: random rotation and scaling of the input cameras, and noise on camera parameters, to cope with inconsistent generated views.

## Headline results and known failure cases
(Reconstructed meshes aligned to ground truth, rescaled to [-1, 1]^3, 16K surface points, F-score threshold 0.2. Confirmed from the table image.)
- GSO: Chamfer distance 0.180 (Mesh) and 0.177 (NeRF) vs TripoSR 0.217. F-score 0.880 and 0.882 vs TripoSR 0.843.
- OmniObject3D orbiting views: Chamfer 0.203 (Mesh), 0.199 (NeRF) vs TripoSR 0.245. F-score 0.864 and 0.869 vs 0.811.
- Time: about 10 seconds per object.
- Failure cases the authors admit: (i) the 64 x 64 triplane may limit high-definition detail; (ii) quality is limited by inconsistency between the diffusion-generated views; (iii) FlexiCubes is "less effective on modeling tiny and thin structures compared to NeRF" (their Figure 4). The third point is directly relevant to us.
- The authors also say TripoSR gives degraded geometry on the back for free-style (non-Objaverse-like) inputs.

## How it relates to our camera route
- It could replace TripoSR as the single-image arm. The gain in the authors' own numbers is about 0.04 in Chamfer (on a [-1, 1] scale). That is probably small next to the effect of thin flat objects, and the paper itself says its mesh stage is weaker on thin structures.
- It partly shares the scale problem (no metric size), but it improves the pose problem: a gravity-aligned canonical frame could cut our 24 axis rotations to a yaw search. That needs checking on the released model.
- Risk for us: the six imagined views come from a diffusion model, so the shape is a sample, not a measurement. Two runs can differ. For a fair test, run several seeds and report the spread.
- A fair test without given size and pose: scale from depth (RGB-D) or table plane, yaw from ICP, then heatmap AUC. Compare with TripoSR under identical alignment. A separate independent study (Nolte et al. 2025, see that note) found InstantMesh gave worse grasp transfer than ZeroShape at similar Chamfer error.

## Likely supervisor questions
1. Why does InstantMesh need two models? Answer: one diffusion model imagines other views, so the second model sees a "multi-view" input and can reconstruct more reliably than from a single view.
2. Is it better than TripoSR for us? Answer: on the authors' benchmarks yes, by a small margin. For thin flat kitchen objects the authors themselves report a weakness of the mesh stage, so we would have to test it.
3. Does it give us a gravity-aligned object? Answer: the paper says its output space has z against gravity. That would reduce the pose search. We have not verified it on the released weights.

## Not verified
- Parameter counts, hardware for the 10-second figure, how yaw is defined in the canonical frame, and any metric scale. Also whether the released checkpoints match the paper variants.

## BibTeX
```bibtex
@article{xu2024instantmesh,
  title   = {{InstantMesh}: Efficient {3D} Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author  = {Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal = {arXiv preprint arXiv:2404.07191},
  year    = {2024}
}
```
