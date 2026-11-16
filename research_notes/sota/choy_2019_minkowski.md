# 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks
**Authors**: Christopher Choy, JunYoung Gwak, Silvio Savarese
**Year / Venue**: 2019, CVPR (arXiv comments field: "CVPR'19"). arXiv v1 18 Apr 2019, v4 13 Jun 2019 (the version read).
**Cite key**: choy_2019_minkowski
**Source read**: https://arxiv.org/abs/1904.08755 (comments field and metadata); https://arxiv.org/pdf/1904.08755 (full text, v4). The fetch summariser could not read the binary PDF, so the text was extracted locally with pdftotext from the saved PDF.

## In one paragraph
This is the paper behind the MinkowskiEngine library. Instead of working on points, it first snaps the points to a voxel grid and stores only the occupied voxels as a sparse tensor: a list of integer coordinates, a list of features, and a batch index. It defines a "generalised sparse convolution" that is evaluated only at chosen output coordinates and with an arbitrary kernel shape, so it covers ordinary dense convolution as a special case. With this, standard 2D architectures such as ResNet can be copied almost directly into 3D, and into 4D (3D plus time) for 3D videos. The paper adds a "hybrid" kernel that keeps a cube in space and a cross in time to save memory, and a conditional random field to make predictions consistent in space and time. Its sparse U-Nets were the leading scene segmentation backbones at the time.

## Input and output
- Input: points quantised to a voxel grid with step size v_l (Alg. 1): coordinates are divided by v_l and floored, then hashed to find unique voxels. Each voxel keeps one feature vector (the paper's augmentation includes chromatic jitter, so colour is used as input on the real datasets). The tensor is stored in COO format with a batch index per row. When labels are used, a voxel that contains more than one distinct label is marked IGNORE_LABEL.
- Voxel sizes used: 5 cm and 2 cm on ScanNet and S3DIS. A whole room is fed in without cropping, fully convolutionally.
- Output: a per-voxel class score; to get per-point predictions, the paper propagates the prediction from the nearest voxel centre to each point.

## Architecture
- Generalised sparse convolution (Eq. 3): x_out(u) = sum over offsets i in N(u, C_in) of W_i x_in(u + i), for output coordinates u in C_out. The kernel shape N and the input and output coordinate sets are free. Kernel maps (pairs of input and output index lists per offset) are built first, then the convolution is a loop of matrix multiplications over offsets (Alg. 2).
- Pooling: max pooling and average, global and sum pooling are implemented for sparse tensors (Alg. 3, Alg. 4). Non-spatial layers such as ReLU and batch norm act directly on the feature matrix.
- Hybrid kernel (Sec. 5.1): a cubic kernel on the spatial axes and a cross-shaped kernel on the time axis. This only matters for 4D data; for a single 3D scan the kernel is a plain cube.
- MinkowskiNet18 (Fig. 4) mirrors ResNet18: a first sparse convolution of size 5x5x5x1 with 64 channels instead of a 7x7 convolution, pooling, then residual stages with 64, 128, 256 and 512 channels using the 3x3x3+3 kernel, then a linear layer.
- MinkowskiUNet32 (Fig. 5): the same stages in a U shape, with strided sparse convolutions 2x2x2x1 (stride 2) down, transposed sparse convolutions up with skip connections between layers of equal stride, and a final 1x1x1x1 convolution to the labels. Other depths (MinkNet20, 32, 42) are variants; only the 18 and UNet32 layouts are drawn, so the layer list of MinkowskiNet42 is not given.
- Trilateral stationary CRF (Sec. 6): a CRF in 7D (space, time, colour) on top of the network, trained jointly by variational inference with a 7D sparse convolution.
- Parameter count: not stated in the paper. (The PTv3 paper's Table 1 lists a "MinkUNet" with kernel size 3 at 37.9 M parameters; that is PTv3's configuration, not a number from this paper.)
- Global context comes only from stride and depth: the receptive field grows with each strided convolution. The paper implements global pooling layers (Sec. 4.4), but the UNet32 drawing in Fig. 5 shows none.

Training: momentum SGD with the Poly scheduler, initial learning rate 0.1; augmentation by random scaling, rotation about the gravity axis, translation, elastic distortion, chromatic translation and jitter.

## Training data and labels
- ScanNet (1.5k room reconstructions), Stanford 3D Indoor Spaces (S3DIS, Fold 1, Area 5 test), RueMonge 2014 (cropped on the fly to a temporal sequence), and a new synthetic Synthia 4D set (20k train, 815 validation, 1886 test scenes) with added noise.
- Labels are per-voxel semantic classes. Loss: cross-entropy per voxel (with IGNORE_LABEL voxels skipped), plus the CRF for the 4D networks.

## Headline results
- ScanNet benchmark test mIoU (Table 1): 67.9 at submission with a 5 cm voxel (MinkowskiNet42). The paper states "+19% mIoU on ScanNet" over the best published work at that deadline. Post-submission: 73.4 at 2 cm in Table 1, against SparseConvNet 72.5 (the running text says 72.1 for the 2 cm model).
- S3DIS Area 5 (Table 4): MinkowskiNet32 mIoU 65.35, mAcc 71.71, against PointNet 41.09; the text states +7% over the best published.
- Synthia 4D (Table 2): 4D MinkNet20 77.46 mIoU, against 76.24 for the 3D MinkNet20; with the CRF 78.30.

## How it differs from our model
- It has: voxelisation, translation-equivariant convolution kernels with a fixed spatial offset layout (not learned relative offsets), a strided U-Net hierarchy for context, and large depth and width. It works best with big, dense-enough scans.
- Ours has: no quantisation (continuous xyz, sub-voxel geometry is kept), a graph neighbourhood with relative feature offsets, a global max feature for context, and every input point has its own output.
- Voxel scale: the paper uses 2 to 5 cm voxels on room scans. For a 10 to 30 cm object, a 2 cm voxel leaves only a handful of cells per side. To resolve edges, handles and rims you would need millimetre voxels, and then the voxel grid is about as large as the point set. All points in one voxel share one prediction (the paper copies the voxel's value to its points), so the heatmap resolution equals the voxel size.
- The paper does not test single objects, and its tasks are multi-class softmax segmentation, whereas ours is per-point sigmoid binary cross-entropy on simulated success labels. Rankings on S3DIS or ScanNet may not transfer.
- Plausible impact of switching for per-point heatmaps on single objects of 2k to 8k points: unlikely to help unless the voxel size is very small; a sparse U-Net has far more parameters than our 0.2 M model, and it is also a heavier software dependency (a CUDA extension). It is a sensible baseline to name, not to adopt.

## Likely supervisor questions
1. Q: Why not use a sparse-voxel backbone, which dominated scene segmentation? A: Those results are on rooms at 2 to 5 cm voxels. Our objects are 10 to 30 cm, so the voxel grid would have to be millimetre-scale, and each voxel gives one prediction. A point-based EdgeConv model keeps continuous coordinates and gives each point its own output.
2. Q: How does your model get global context compared to MinkowskiNet? A: Through a global max over all points copied back to every point. In the drawn U-Net, MinkowskiNet gets context from strides and depth only. For a heatmap that depends on the object as a whole (for example whether a plate is wide enough for a pan) an explicit global feature is a simple and direct route.
3. Q: Is generalised sparse convolution relevant to your pair model? A: Not directly. The paper's generalisation lets the output coordinates differ from the input coordinates, so in principle one could convolve one object's voxels onto another's. We use cross-attention for that, and have not tried this.

## Not verified
- Parameter count: not stated in the paper.
- The layer list of MinkowskiNet42 (used for the ScanNet result): only the 18 and UNet32 variants are drawn.
- ScanNet at 2 cm: 72.1 in the running text but 73.4 in Table 1; I report both without choosing.
- Which voxel size gives the best 3D accuracy at object scale: not studied.
- The "millimetre voxels" point is my reasoning, not a claim in the paper.

## BibTeX
```bibtex
@inproceedings{choy_2019_minkowski,
  title     = {4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author    = {Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
  eprint    = {1904.08755},
  archivePrefix = {arXiv}
}
```
