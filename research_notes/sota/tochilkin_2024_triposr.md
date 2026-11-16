# TripoSR: Fast 3D Object Reconstruction from a Single Image
**Authors**: Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, Yan-Pei Cao (Stability AI and Tripo AI)
**Year / Venue**: 2024, arXiv technical report (cs.CV), v1 dated 4 March 2024. Not a peer-reviewed venue.
**Cite key**: tochilkin_2024_triposr
**Source read**: https://arxiv.org/abs/2403.02151 (abstract page, fetched). Full text: https://arxiv.org/pdf/2403.02151 (downloaded, converted with pdftotext; the full 5-page report read, and the page with Table 1 viewed as an image).

## In one paragraph
TripoSR is an open-source model that makes a 3D mesh from one image in about half a second. It is built directly on LRM. It keeps the idea of a vision transformer encoder, a transformer that writes a triplane, and a small NeRF MLP. The authors change four things: cleaner training data and more realistic rendering, a smaller triplane (40 channels instead of 80), an extra mask loss that removes floating blobs, and training on random 128 x 128 crops that favour the object. The biggest design change is that the camera is no longer given. The model has to "guess" the camera itself, which makes it robust to in-the-wild photos. It beat other open feed-forward models on two public benchmarks. It is the model in our camera route (c).

## Input and output
- Input: one RGB image, 512 x 512 (patch size 16). No camera parameters are used. The paper says the model guesses extrinsics and intrinsics during training and inference. The foreground should be segmented, but the paper does not describe background removal. Not verified in the paper text.
- Output: a triplane NeRF with shape 64 x 64 x 40. A mesh is extracted with marching cubes. The paper does not give the marching cubes resolution (our 256 is a code setting, not verified in the paper text). The paper says the isosurface is extracted with marching cubes.
- Scale and pose: not metric. Rendering uses a radius of 0.87, so the object sits in a normalised volume. The pose is view-centred. The TripoSR paper implies this when it says that some other methods cannot predict view-centric shapes. The InstantMesh paper states directly that TripoSR reconstructs in view space. For their own evaluation the authors align every prediction to ground truth by a brute-force search over one rotation angle, then ICP. How scale was handled in that evaluation is not stated.

## Architecture
Values from the paper's Table 1:
- Image tokenizer: DINOv1 ViT, 12 attention layers, 768 feature channels, 512 x 512 input, patch 16.
- Triplane tokenizer: 32 x 32 x 3 tokens, 16 channels (as printed in the table).
- Backbone: 1024 channels, 16 attention layers, 16 heads, head dimension 64, cross-attention dimension 768. Each layer has self-attention and cross-attention.
- Triplane upsampler: factor 2, 1024 input channels, 40 output channels, output 64 x 64 x 40.
- NeRF MLP: width 64, 10 layers, SiLU. Renderer: 128 samples per ray, radius 0.87, exp density activation, density bias -1.0.
- Speed: about 0.5 s per mesh on an A100.
- Parameter count: not stated in the paper.

## Training data and loss
- Data: a curated subset of Objaverse (CC-BY licence). The subset size is not stated. Rendering uses several different techniques meant to look more like real images. Details not stated.
- Loss: MSE + 2.0 x LPIPS + 0.05 x mask loss (binary cross-entropy between rendered and true masks), averaged over supervision views.
- Optimiser: AdamW, learning rate 4e-4, cosine schedule, 2000 warm-up steps. Training compute: not stated.

## Headline results and known failure cases
- Google Scanned Objects (about 300 filtered objects): Chamfer distance 0.111, F-score at 0.1 = 0.651, at 0.2 = 0.871, at 0.5 = 0.980. OpenLRM gets 0.180 / 0.430. TGS gets 0.122 / 0.637.
- OmniObject3D (about 300 filtered objects): Chamfer distance 0.102, F-score at 0.1 = 0.677 vs OpenLRM 0.155 / 0.486.
- Metrics use 10,000 sampled surface points. The authors removed many simple shapes (boxes, spheres, cylinders) by hand to avoid bias toward them.
- The paper has no limitations section and no failure analysis. Thin, flat, hollow or occluded shapes are not discussed. Our finding (worst on trays, woks, plates) is therefore our own observation, not a known limit from the paper.
- Warning for comparisons: the InstantMesh paper reports TripoSR Chamfer 0.217 on GSO, not 0.111. The protocols differ (different normalisation, point counts, thresholds). Numbers from different papers cannot be compared.

## How it relates to our camera route
- It is route (c). Everything we measure about "TripoSR" is measured on this model.
- It does not give scale or pose. The authors themselves needed rotation search and ICP to evaluate it. Our 24 axis rotations + ICP + scaling to true size is the same kind of fix, but we feed the true size, which the authors did not (or at least did not say they did). So our 0.82 AUC is an upper bound for this model.
- Why thin flat objects fail is not answered by the paper. Candidate reasons, all my inference and all testable: the 64 x 64 triplane is coarse for objects a few millimetres thick; a deterministic regressor averages the unseen underside; flat objects seen at 35 degrees give weak depth cues.
- A fairer test needs: the segmented RGB crop only; size estimated from depth or a table plane; pose estimated by ICP against the partial RGB-D cloud; then the same heatmap AUC. Report the drop from 0.82 to the honest number.

## Likely supervisor questions
1. Why does TripoSR work without a camera? Answer: it was trained to guess the camera, so it outputs a view-centred shape in a normalised volume. That is why we need alignment afterwards.
2. Is 0.82 AUC a fair number for the TripoSR route? Answer: no, it is optimistic. We give the true size and pose. Only shape error is measured. A real pipeline must estimate both.
3. Why is TripoSR worst on plates and woks? Answer: the paper does not say. We suspect a coarse 64 x 64 triplane and the averaged unseen side. We can test this by plotting error against thickness.

## Not verified
- Parameter count, training set size, training compute, marching cubes resolution, how the input is segmented, how scale was treated in the authors' evaluation, and any statement on thin or flat objects.

## Verification (second read, 2026-11-16)
Checked against the arXiv HTML text by a second reader. Every number of Tables 1 to 3 quoted above is confirmed, as are the "guess the camera" sentence, the loss, the 128 x 128 patches, the optimiser, the rotation search plus ICP, and the Objaverse CC-BY subset. Marching cubes resolution, parameter count and training set size are indeed not in the paper.

## BibTeX
```bibtex
@article{tochilkin2024triposr,
  title   = {{TripoSR}: Fast {3D} Object Reconstruction from a Single Image},
  author  = {Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal = {arXiv preprint arXiv:2403.02151},
  year    = {2024}
}
```
