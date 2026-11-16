# LRM: Large Reconstruction Model for Single Image to 3D
**Authors**: Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, Hao Tan (Adobe Research; Hong also ANU)
**Year / Venue**: 2024, ICLR 2024 (arXiv v1 Nov 2023, v2 Mar 2024). The venue comes from the arXiv abstract page. The PDF text does not print it.
**Cite key**: hong_2024_lrm
**Source read**: https://arxiv.org/abs/2311.04400 (abstract page, fetched). Full text: https://arxiv.org/pdf/2311.04400 (v2, downloaded and converted to text with pdftotext; main text and appendix read).

## In one paragraph
LRM asks whether one big transformer, trained on about a million 3D objects, can turn a single photo of any object into a 3D model. It takes the image, encodes it with a pre-trained DINO vision transformer, and lets a transformer decoder write the answer into a "triplane", which is three 2D feature maps standing for a 3D volume. A small MLP reads colour and density from the triplane, so the result is a neural radiance field (NeRF). Training needs no 3D ground truth shapes as such. The model is only asked to render the object from other views and match the real images. It is deterministic and fast (about 5 seconds), and it is the ancestor of TripoSR. For us it matters because it explains what TripoSR can and cannot know: the shape of the unseen side is a learned average, and the camera is assumed, not measured.

## Input and output
- Input: one RGB image, 512 x 512, object squared and background removed. No depth.
- Camera: during training the camera (4x4 extrinsics, focal length, principal point, 20 numbers) is fed in. At inference the camera is not known, so the paper assigns the same fixed normalised camera that was used for the Objaverse training data. Training cameras were normalised to position [0, -2, 0] with the z-axis up.
- Output: a triplane NeRF, 3 planes of 64 x 64 with 80 channels each. A mesh is made by marching cubes on a 384 x 384 x 384 grid of queries.
- Scale and pose: no metric scale. Objects live in a normalised box [-1, 1]^3. The pose is view-centred (relative to the assumed camera). The paper says LRM "does not depend on a canonical pose of the object".

## Architecture
- Image encoder: DINO ViT-B/16, 12 layers, width 768. It makes 1025 tokens (1024 patch tokens plus [CLS]) from the 512 x 512 input.
- Image-to-triplane decoder: 16 transformer layers, hidden size 1024. Each layer has cross-attention (to the image tokens), self-attention (among triplane tokens) and an MLP. The queries are learnable positional embeddings of size (3 x 32 x 32) x 1024. Camera features modulate every sub-layer through adaptive layer norm (as in DiT).
- Upsampling: a learnable deconvolution turns 3 x 32 x 32 tokens into 3 x 64 x 64 planes, 1024 channels down to 80.
- NeRF MLP: 10 layers, width 64, outputs RGB and density. 128 samples per ray, 128 x 128 renders for supervision.
- Size: more than 500 million learnable parameters. Time: under 5 seconds on one A100 (about 1.14 s image to triplane, 1.14 s to query 384^3 points, 1.91 s marching cubes).

## Training data and loss
- Data: Objaverse (730,648 assets after pre-processing, 32 random views each, 1024 x 1024 renders, white background) and MVImgNet (220,219 real object videos, cropped and background-removed). About one million shapes in total.
- Loss: for each shape the input view plus 3 side views are rendered. Loss is pixel MSE plus 2.0 times LPIPS. There is no mask loss, no depth loss and no 3D supervision.
- Compute: 128 A100 (40 GB) GPUs, batch 1024, 30 epochs, about 3 days.

## Headline results and known failure cases
- Google Scanned Objects, 100 random objects (paper's own appendix table): Chamfer distance 0.053 (LRM) vs 0.099 (Point-E), 0.085 (Shap-E), 0.123 (One-2-3-45). FID 31.44 vs 97.05 for the next best. The unit of the Chamfer distance is not stated in the text I read.
- Data ablation (small model): synthetic only PSNR 15.5, real only 17.5, both 19.0. Real videos matter a lot.
- Failure cases the authors admit: (1) blurry textures on occluded regions, because the model is deterministic and likely outputs an average of plausible answers; (2) wrong assumed camera intrinsics and extrinsics distort the shape, especially when the image was cropped or resized; (3) no background or scenes; (4) Lambertian appearance only.
- Thin, flat or hollow shapes are not discussed. Not verified in the paper text.

## How it relates to our camera route
- It is the architecture behind our TripoSR arm. TripoSR keeps the DINO encoder, the 16-layer decoder and the triplane NeRF, and changes data, loss and camera handling. So what we learn here about the "averaged unseen side" and the "assumed camera" carries over.
- It does not solve scale and pose. It makes them worse in one sense: the shape is only right up to the assumed camera. A wrong assumed field of view distorts the shape (the authors show this).
- The triplane is 64 x 64 over a normalised box. A flat plate or tray occupies only a few cells in thickness. This is my inference, not a claim of the paper. It is a plausible reason for TripoSR being worst on thin flat objects, and it is a testable one (compare error against thickness divided by object size).
- Not a replacement for TripoSR in our test. A fair test without given size and pose needs: (a) a scale estimate (from depth or a known table plane), (b) a pose estimate (for example ICP against the partial RGB-D cloud, as in route b), and (c) error reported with and without that estimation.

## Likely supervisor questions
1. Why does a single image give a full 3D shape at all? Answer: the model has learned a shape prior from about a million objects. The unseen side is a learned guess, and the paper admits it is an average of plausible answers.
2. Does LRM know the real size of the object? Answer: no. It reconstructs in a normalised box with an assumed camera. Our TripoSR numbers fix the size and pose by hand, so they are optimistic.
3. Why train with rendering loss instead of 3D shapes? Answer: it lets the model use videos and renders, so it scales. The price is that the geometry is only as good as the views it was supervised with.

## Not verified
- Unit and normalisation of the Chamfer distance in the appendix table.
- Anything about thin, flat or hollow objects (the paper does not discuss them).
- The venue (ICLR 2024) is taken from the arXiv abstract page only.

## BibTeX
```bibtex
@inproceedings{hong2024lrm,
  title     = {{LRM}: Large Reconstruction Model for Single Image to {3D}},
  author    = {Hong, Yicong and Zhang, Kai and Gu, Jiuxiang and Bi, Sai and Zhou, Yang and Liu, Difan and Liu, Feng and Sunkavalli, Kalyan and Bui, Trung and Tan, Hao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2311.04400}
}
```
