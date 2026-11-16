# Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS)
**Authors**: Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, Jiaolong Yang (Tsinghua University, USTC, Microsoft Research)
**Year / Venue**: 2025, CVPR 2025 (arXiv v1 Dec 2024, v3 30 May 2025). The arXiv page and the PDF text do not name the venue. CVPR 2025 was confirmed from a web search result pointing to the CVF open access page (pages 21469 to 21480); I could not open that page myself (HTTP 403).
**Cite key**: xiang_2025_trellis
**Source read**: https://arxiv.org/abs/2412.01506 (abstract page, fetched). Full text: https://arxiv.org/pdf/2412.01506 (v3, downloaded, converted with pdftotext; main text, Table 6 and appendices A, B, C and E read).

## In one paragraph
TRELLIS is a generative model, not a measuring one. It makes a 3D asset from a text or an image prompt. Its key idea is a "structured latent" (SLAT): a sparse 64^3 voxel grid where only the voxels touching the surface are active, and each active voxel carries a latent vector built from DINOv2 features of many rendered views. The same latent can be decoded into 3D Gaussians, radiance fields or a mesh. Generation runs in two steps with rectified flow transformers: first the sparse structure (which voxels are occupied), then the latent for each occupied voxel. For us it is the strongest current single-image to mesh option, but it is a sampler, so the output for one photo is one plausible shape, not the measured shape.

## Input and output
- Input for the image version: one image, encoded with DINOv2 and injected by cross-attention. Image prompts were rendered during training with field of view between 10 and 70 degrees. Input resolution and background handling are not described in the parts I read. Not verified. No camera parameters are supplied.
- Output: SLAT, a sparse set of about 20K active voxels at 64^3 (average L = 20K), decoded to Gaussians, radiance field or mesh. The mesh decoder upsamples to 256^3 and uses FlexiCubes.
- Scale and pose: not stated as a convention. Training assets were rendered by cameras on a sphere of radius 2 looking at the origin, and Gaussians are sampled in a (-0.5, 0.5)^3 cube, so the output is in a normalised cube without metric size (my inference from these details). The pose relative to the input view is not stated. Not verified.

## Architecture
- Encoding (training only): render 150 views, take DINOv2 features, project each active voxel to the feature maps and average. A sparse VAE (transformer with 3D shifted-window attention, 12 layers, width 768, 12 heads) encodes to latents. Decoders (12 layers each) give Gaussians, radiance fields or meshes.
- Stage 1, structure: a small 3D conv VAE compresses the 64^3 occupancy grid to a 16^3 feature grid (8 latent channels). A flow transformer G_S denoises it.
- Stage 2, latents: a sparse flow transformer G_L with sparse-conv down- and up-samplers.
- Sizes (Table 6): text models B, L and XL with total about 342M, 1.1B and 2B parameters. The image model is size L: G_S 556M plus G_L 600M, 24 layers, width 1024, 16 heads each.
- Inference: 50 sampling steps, classifier-free guidance strength 3. About 10 seconds per asset (Figure 1 caption). GPU for that figure is not stated.

## Training data and loss
- Data: about 500K assets (500,777) after an aesthetic-score filter. Sources: Objaverse(XL) Sketchfab 168,307 and GitHub 311,843; ABO 4,485; 3D-FUTURE 9,472; HSSD 6,670. 150 renders per asset; GPT-4o captions for text training.
- Losses: Gaussian and radiance field decoders use L1 + 0.2 (1 - SSIM) + 0.2 LPIPS. Mesh decoder uses L1 on rendered depth and normals plus colour and regularisation terms. Sparse structure VAE uses Dice loss. The generators use the conditional flow matching objective.
- Compute for XL: 64 A100 (40 GB), 400K steps, batch 256.

## Headline results and known failure cases
- Reconstruction of a known 3D asset through the latent (Toys4k subset of 500, not from an image): Chamfer distance 0.0083 and F-score 0.9999, vs CLAY 0.0124 / 0.9976. This only shows the latent can hold geometry well. It is not a single-image result.
- Image-to-3D (Toys4k): the paper reports only distribution and prompt-alignment metrics (FD, KD, CLIP) and a user study with over 100 participants. It states it is better than InstantMesh, LGM, Shap-E and others on all of them. It gives no Chamfer distance against the ground truth shape of the photographed object. So there is no paper number for shape accuracy from one image.
- Scaling ablation: larger models are better (B < L < XL on CLIP and FD). Rectified flow beats diffusion at both stages.
- Failure cases the authors admit (Appendix E): two-stage generation is less efficient than one-stage; the image-to-3D model "does not separate lighting effects", so shading and highlights from the photo are baked into the asset.
- Thin, flat, hollow and occluded shapes are not listed as limitations. Not verified.

## How it relates to our camera route
- It could replace TripoSR as the single-image arm and is probably the best current candidate. It is explicit about occupancy (a 64^3 grid first), then fills in detail, and its mesh decoder goes to 256^3.
- It does not solve scale or pose. It also adds a new problem: it is stochastic. Each seed gives a different shape for the unseen side. We would need to run several samples, measure the spread, and report the mean AUC with a spread.
- The paper gives no evidence on thin flat objects, so a test on our trays, woks and plates is new information.
- A fair test without given size and pose: same as for TripoSR (scale from RGB-D or the table plane, pose by ICP against the partial cloud), several seeds, and one baseline that is the measured partial cloud alone. If the measured cloud wins, the generative arm is not needed when RGB-D is present.

## Likely supervisor questions
1. Why not simply use TRELLIS instead of TripoSR? Answer: it is probably better, but it is generative. Its output is one plausible shape. We would have to test it on our objects, with several samples, and with size and pose estimated, not given.
2. Does TRELLIS give accurate geometry from a photo? Answer: the paper does not measure that. Its image-to-3D scores are about visual quality and prompt match. Its low Chamfer number is for reconstructing a known mesh through the latent.
3. Does it help if we have RGB-D? Answer: not by design. It reads only the image. Its output could be aligned to the depth cloud, but we have no evidence it beats using the depth cloud directly.

## Not verified
- Input resolution and background handling for the image model, pose and scale convention of the output, inference hardware, any claim about thin or flat shapes, and the exact image-to-3D rows of Table 2 (text extraction of that table was garbled, so I quote only the paper's sentence about it).
- The venue was confirmed through a search result, not through the arXiv page or PDF.

## BibTeX
```bibtex
@inproceedings{xiang2025trellis,
  title     = {Structured {3D} Latents for Scalable and Versatile {3D} Generation},
  author    = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  pages     = {21469--21480},
  note      = {arXiv:2412.01506}
}
```
