# Part-level affordance detection and object part segmentation (2D/3D), incl. handle detection — fit to UTAMP-2 "members"

Scope: can off-the-shelf part or affordance models automatically produce UTAMP-2 members (body / handle / edge sub-regions with boxes aligned to the parent object box) and grasp/support judgments for: salt shaker, glass mug, cutting board, frying pan, wooden tray (Franka Panda, CoppeliaSim).

Environment caveat: arxiv.org, ar5iv, openaccess.thecvf.com, umd.edu, sites.google.com, csc.liv.ac.uk and semanticscholar were unreachable from this sandbox (DNS/proxy blocks). Facts below come from GitHub pages (fetched directly), search-engine snippets of the primary papers and dataset pages, and curated lists. Items marked **[UNVERIFIED]** come from my prior knowledge and could not be confirmed against a source in this session.

---

## Q1. AffordanceNet (Do, Nguyen, Reid, ICRA 2018): architecture, outputs, code

### Takeaway
AffordanceNet is a Mask R-CNN-style, two-branch network (object detection plus a multi-class affordance mask per RoI) trained on IIT-AFF. Its official code is Caffe (MIT license) with one pretrained IIT-AFF model, and no official PyTorch port exists. It outputs IIT-AFF labels that include "grasp" and "support", but those labels do not match UTAMP's meaning of grasp/support per box face (see Q2).

### Cited Findings
- **Paper:** "AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection", Thanh-Toan Do, Anh Nguyen, Ian Reid, ICRA 2018 (arXiv 1709.07326). The GitHub README credits Do*, Nguyen*, Reid, Caldwell, Tsagarakis. — [ACM DL ICRA listing](https://dl.acm.org/doi/10.1109/ICRA.2018.8460902); [arXiv](https://arxiv.org/pdf/1709.07326); [author PDF](https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2018_ICRA_AffordanceNet.pdf); [mirror README](https://github.com/bygreencn/affordance-net)
- **Architecture:** two branches, one for object detection and one for affordance detection. The key parts are a sequence of deconvolutional layers, a "robust resizing strategy" and a multi-task loss. RoIAlign produces 7×7 RoI maps (box regression and classification) and 14×14 maps (affordance mask). Conv-deconv layers then upsample the RoI feature map to a high-resolution mask. — [search excerpt of paper](https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2018_ICRA_AffordanceNet.pdf)
- The README calls AffordanceNet "a general version of Mask-RCNN when we have multiple classes inside each instance". — [bygreencn mirror](https://github.com/bygreencn/affordance-net)
- **Speed:** 150 ms per image. — [paper via search](https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2018_ICRA_AffordanceNet.pdf)
- **Accuracy:** average weighted F-beta of 73.35% on IIT-AFF, 3.7% better than the previous best. — [search summary of thesis 1903.09761 / AffordanceNet paper](https://arxiv.org/pdf/1709.07326)
- **Outputs:** per detected object, the class, the bounding box and a pixel-wise multi-class affordance mask inside the box. Demos: `demo_img.py` (static image) and `demo_asus.py` (depth camera; the user picks an object and affordance, e.g. bottle + grasp, and a 3D grasp pose is shown in rviz). — [nqanh/affordance-net](https://github.com/nqanh/affordance-net)
- **Code:** Caffe, which must be built with Python layers (`caffe-affordance-net` folder; based on Faster R-CNN). MIT License. Pretrained `AffordanceNet_200K.caffemodel` on the IIT-AFF training set (Google Drive/OneDrive). Data must be converted to Pascal-VOC format. — [nqanh/affordance-net](https://github.com/nqanh/affordance-net)
- **PyTorch:** no official port. A 2017 PyTorch-forum thread asked for one without result. One paper re-trained an MMDetection Mask R-CNN on IIT-AFF as a baseline instead. `apicis/aff-seg` depends on a `HuchieWuchie/affnetDR` repo with a `lib/roi_heads.py`, which suggests a torchvision-style re-implementation **[nature UNVERIFIED]**. — [PyTorch forum](https://discuss.pytorch.org/t/mask-rcnn-and-affordancenet-in-pytorch/8381); [Are standard object segmentation models sufficient…, arXiv 2107.02095](https://arxiv.org/pdf/2107.02095); [apicis/aff-seg](https://github.com/apicis/aff-seg)
- **Modern IIT-AFF models:** AffKernel (RT-DETR-based, Hugging Face `anhrisn/affkernel-iit-aff`, code anh0001/affkernel) reports Mask R-CNN at 0.844 and 45 ms as a baseline on IIT-AFF. These are model-card figures, not independently verified. YOLOA (arXiv 2512.03418, "Real-time affordance detection via LLM adapter") also evaluates on IIT-AFF. — [HF model card](https://huggingface.co/anhrisn/affkernel-iit-aff); [YOLOA](https://arxiv.org/pdf/2512.03418)
- **Reproducibility:** "Segmenting Object Affordances: Reproducibility and Sensitivity to Scale" (arXiv 2409.01814) re-benchmarks affordance segmentation methods on UMD and IIT-AFF. — [arXiv 2409.01814](https://arxiv.org/html/2409.01814v1)

### Inferences
- **Verdict (fit): low to moderate.**
  - AffordanceNet's object vocabulary is the 10 IIT-AFF classes. "cup" and "pan" are close to our glass mug and frying pan. Salt shaker, cutting board and tray are absent, so a retrain is needed.
  - The masks are 2D image regions. A per-member 3D OBB would need back-projection with depth, and the glass mug will give poor depth.
  - The Caffe stack is obsolete. A practical route is to re-implement it as torchvision/MMDetection Mask R-CNN with multi-label masks, or to use AffKernel. Both are trained on the same labels.
- AffordanceNet's own label semantics (Q2) are tool-use oriented ("grasp" = handle region, "support" = spatula blade). They do not express UTAMP's "this face can support an object placed on it" or "this face can be approached by the gripper".

### Gaps
- Exact mask resolution (the often-quoted 244×244) and the details of the resizing threshold could not be verified because the PDF was unreachable.
- No confirmed maintained PyTorch AffordanceNet port was found.

---

## Q2. UMD Part Affordance dataset (Myers et al., ICRA 2015) and IIT-AFF (Nguyen et al., IROS 2017): label sets, definitions, categories

### Takeaway
- **UMD:** 7 affordances (grasp, cut, scoop, contain, pound, support, wrap-grasp) over 17 tool categories, including mug, cup, bowl, pot, turner, ladle, etc.
- **IIT-AFF:** 9 affordances (contain, cut, display, engine, grasp, hit, pound, support, w-grasp) over 10 categories, including cup, bowl, pan and spatula.
- **"support":** in UMD, flat parts that hold loose material (the turner/spatula blade).
- **"wrap-grasp":** a part held with the whole hand and palm (the outside of a cup or mug body).
- **Our objects:** neither dataset contains trays, cutting boards or salt shakers. Mugs/cups are in both. Pans are only in IIT-AFF; UMD has "pot".

### Cited Findings
**UMD Part Affordance (RGB-D Part Affordance Dataset)**
- **Paper:** Austin Myers, Ching L. Teo, Cornelia Fermüller, Yiannis Aloimonos, "Affordance Detection of Tool Parts from Geometric Features", ICRA 2015. — [UMD project page](https://users.umiacs.umd.edu/~fer/affordance/Affordance.html); [ICRA15 PDF](http://users.umiacs.umd.edu/~fermulcm/affordance/ICRA15_affordance_parts_final.pdf); [ResearchGate](https://www.researchgate.net/publication/282930046_Affordance_detection_of_tool_parts_from_geometric_features)
- **Contents:** RGB-D images and ground-truth affordance labels for 105 kitchen, workshop and garden tools, plus 3 cluttered scenes. More than 10,000 RGB-D frames with pixel-level labels over 360° of views. Each part also gets a ranking of multiple affordances besides the pixel-wise label of the most likely one. — [dataset page](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/index.html); [dataset page alt](http://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/index.html)
- **7 affordances** "associated with the surfaces of tool parts": grasp, cut, scoop, contain, pound, support, wrap-grasp. — [dataset page](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/index.html)
- **17 categories:** bowl, cup, hammer, knife, ladle, mallet, mug, pot, saw, scissors, scoop, shears, shovel, spoon, tenderizer, trowel, turner. — [search over dataset page / 2004.08644](https://arxiv.org/pdf/2004.08644)
- **Definitions** (search snippets of the dataset's definitions):
  - **wrap-grasp:** "can be held with the hand and palm (the outside of a cup)".
  - **support:** "flat parts that can hold loose material (turners/spatulas)".
  — [dataset page](http://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/index.html)
- **Other definitions [UNVERIFIED, from memory of the dataset page]:**
  - grasp: can be enclosed by a hand for manipulation (handles)
  - cut: separating another object (knife blade)
  - scoop: curved surface with a mouth for gathering soft material (trowel)
  - contain: deep cavity to hold liquid (inside of bowls)
  - pound: striking other objects (hammer head)
- **Split:** a predefined split of about 14,823 training and 14,020 test images, holding out about half of the object instances per category. — [arXiv 2409.01814 via search](https://arxiv.org/pdf/2409.01814)

**IIT-AFF**
- **Paper:** Anh Nguyen, Dimitrios Kanoulas, Darwin G. Caldwell, Nikos Tsagarakis, "Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields", IROS 2017 (DOI 10.1109/IROS.2017.8206484). — [ResearchGate](https://www.researchgate.net/publication/318284584_Object-Based_Affordances_Detection_with_Convolutional_Neural_Networks_and_Dense_Conditional_Random_Fields); [IIT-AFF site](https://sites.google.com/site/iitaffdataset/)
- **10 object categories:** bottle, bowl, cup, drill, hammer, knife, monitor, pan, racket, spatula.
- **9 affordances:** contain, cut, display, engine, grasp, hit, pound, support, w-grasp.
- **Size:** 8,835 images, 14,642 object boxes (7,866 from ImageNet) and 24,677 affordance parts. Images mix ImageNet with RGB-D captures from two sensors, all in cluttered scenes, with pixel-level affordance masks.
— [IIT-AFF site via search](https://sites.google.com/site/iitaffdataset/); [YOLOA](https://arxiv.org/pdf/2512.03418); [thesis 1903.09761](https://arxiv.org/pdf/1903.09761)

### Inferences
- **"support" (UMD/IIT-AFF) differs from UTAMP "support".**
  - In UMD and IIT-AFF it is a **tool-use** affordance: the flat blade of a turner/spatula that holds loose material on top of it. In IIT-AFF it appears mainly on spatula heads (by analogy with UMD; per-object mapping unverified).
  - UTAMP needs "this face of this member can support another object placed on it" (tray top, cutting-board top, pan bottom), plus possibly "this face can rest on a surface". A model trained on these datasets is unlikely to label a wooden tray or cutting-board top as "support". If it did, that would be extrapolation.
  - A geometric test (upward-facing planar face of sufficient area) is more reliable for UTAMP support.
- **"wrap-grasp" vs "grasp".**
  - wrap-grasp = the whole hand wraps around a convex body (mug/cup outer wall, bottle body).
  - grasp = handle-like parts enclosed by the fingers.
  - For a parallel-jaw Panda gripper both can be graspable. Which applies depends on the member's width versus the ~80 mm gripper opening (Panda spec, not checked here). wrap-grasp on a mug body maps naturally to a UTAMP "body" member; grasp maps to the "handle" member.
- **Coverage of our objects:**
  - glass mug: UMD mug/cup; IIT-AFF cup.
  - frying pan: IIT-AFF pan; UMD pot (similar).
  - salt shaker: loosely IIT-AFF bottle.
  - cutting board, tray: no coverage.
- **Verdict (fit):** useful as a weak prior for mug and pan handles (grasp) and bodies (wrap-grasp/contain). Not usable for support on trays or boards, and not usable for per-box-face reasoning.

### Gaps
- Verbatim IIT-AFF definitions of "support" and "w-grasp", and the exact per-category label assignment (e.g. whether a pan's interior is labeled contain or support), could not be retrieved because the IIT-AFF site and paper were unreachable.
- The full verbatim UMD definitions of grasp, cut, scoop, contain and pound are unverified.

---

## Q3. 3D AffordanceNet, LASO, OpenAD, AffordanceLLM and 2023–2026 affordance grounding

### Takeaway
- **3D affordance benchmarks:** 3D AffordanceNet (CVPR 2021) is the base benchmark: 18 affordances on about 23k PartNet shapes. It does include "support", "wrap-grasp", "grasp", "contain" and "pour", and it covers Mug, Bowl, Bottle, Table and Vase. OpenAD and LASO build on it with language queries.
- **2D grounding models:** OOAL, AffordanceLLM and GLOVER output heatmaps or masks on images for "grasp/hold"-type queries.
- **Robot-oriented methods:** RAM and UAD output contact points or directions for manipulation.
- **Bottom line:** none outputs UTAMP-style per-member, per-box-face grasp/support labels directly. The 3D ones work on clean, complete object point clouds (PartNet distribution), not on single partial RGB-D views of glass objects.

### Cited Findings
**3D AffordanceNet**
- Deng, Xu, Wu, Chen, Jia, "3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding", CVPR 2021 (arXiv 2103.16397). MIT-licensed code at Gorilla-Lab-SCUT/AffordanceNet. — [GitHub](https://github.com/Gorilla-Lab-SCUT/AffordanceNet); [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_3D_AffordanceNet_A_Benchmark_for_Visual_Object_Affordance_Understanding_CVPR_2021_paper.html)
- About 23k shapes (22,949) from 23 categories, 56k affordance annotations, 18 affordance categories, at most 5 affordance types per category. Labels are multi-label per point. — [GitHub](https://github.com/Gorilla-Lab-SCUT/AffordanceNet); [paper via search](https://arxiv.org/pdf/2103.16397)
- **18 labels:** grasp, contain, lift, open, lay, sit, support, wrap-grasp, pour, move, display, push, pull, listen, wear, press, cut, stab. The search confirmed push, pull, listen, wear and press in the paper's category table, plus the naming lay/sit/pour/display/wrap-grasp. — [paper via search](https://arxiv.org/pdf/2103.16397)
- **Tasks:** full-shape, partial-view, rotation-invariant and semi-supervised affordance estimation. — [GitHub](https://github.com/Gorilla-Lab-SCUT/AffordanceNet)
- **[UNVERIFIED]:**
  - The 23 categories are PartNet categories (Bag, Bed, Bottle, Bowl, Chair, Clock, Dishwasher, Display, Door, Earphone, Faucet, Hat, Keyboard, Knife, Laptop, Microwave, Mug, Refrigerator, Scissors, StorageFurniture, Table, TrashCan, Vase).
  - Mug carries grasp / contain / pour / wrap-grasp.
  - Table, Bed and Chair carry support.
  - Annotation used keypoints propagated into soft per-point heatmaps over 2048-point clouds.

**OpenAD**
- Toan Nguyen, Minh Nhat Vu, An Vuong, Dzung Nguyen, Thieu Vo, Ngan Le, Anh Nguyen, "Open-Vocabulary Affordance Detection in 3D Point Clouds", IROS 2023 (arXiv 2303.02401). It learns point and text features jointly, gives zero-shot detection of unseen affordance labels, runs at about 100 ms inference, and has a project page at openad2023.github.io. — [arXiv](https://arxiv.org/abs/2303.02401); [Bath portal](https://researchportal.bath.ac.uk/en/publications/open-vocabulary-affordance-detection-in-3d-point-clouds/); [Liverpool repository](https://livrepository.liverpool.ac.uk/id/eprint/3177437)
- Training data is 3D AffordanceNet **[the "trained on 3D AffordanceNet label set" detail is UNVERIFIED here; the repo page fetch returned 404 at Fsoft-AIC/Open-Vocabulary-Affordance-Detection]**.

**LASO**
- Yicong Li, Na Zhao, Junbin Xiao, Chun Feng, Xiang Wang, Tat-Seng Chua, "LASO: Language-guided Affordance Segmentation on 3D Object", CVPR 2024, pp. 14251–14260. Dataset (AffordQ): 19,751 point–question pairs, 8,434 shapes, 870 expert-written questions. Method: PointRefer, where language-conditioned queries generate dynamic kernels that produce the mask. Code was listed at anonymous.4open.science/r/LASO during review. — [CVPR poster](https://cvpr.thecvf.com/virtual/2024/poster/30491); [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.html); [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)

**Other 2023–2026 methods** (from the curated list; venue and title verified there)
- **OOAL** "One-Shot Open Affordance Learning with Foundation Models" (Gen Li, Deqing Sun, Laura Sevilla-Lara, Varun Jampani), CVPR 2024. Code at Reagan1311/OOAL (MIT per snippet). There is also a UMD-trained variant at yhzhengoo/OOAL_umd. It segments affordance regions in 2D images from one example per category, using DINOv2 + CLIP. — [project](https://reagan1311.github.io/ooal/); [GitHub](https://github.com/Reagan1311/OOAL); [OOAL_umd](https://github.com/yhzhengoo/OOAL_umd)
- **AffordanceLLM** (CVPRW 2024): "Grounding Affordance from Vision Language Models", arXiv 2401.06341. It outputs a 2D affordance map from an image + text.
- **3D-AffordanceLLM** (ICLR 2025): arXiv 2502.20041, open-vocabulary 3D affordance detection.
- **GLOVER** (arXiv 2411.12286, "Generalizable open-vocabulary affordance reasoning for task-oriented grasping") and **GLOVER++** (CoRL 2025, arXiv 2505.11865).
- **RAM** (CoRL 2024): retrieval-based affordance transfer for zero-shot manipulation.
- **UAD** (ICRA 2025): unsupervised affordance distillation, arXiv 2506.09284.
- **RoboPoint** (CoRL 2024): spatial affordance point prediction.
- **ManipVQA** (IROS 2024).
- **Affordance-R1** (AAAI 2026).
- **RoboAfford / RoboAfford++** (ACM MM 2025): datasets.
- **AffordDexGrasp** (ICCV 2025).
- **Other datasets:** HANDAL (IROS 2023), InstructPart (ACL 2025), RAGNet (ICCV 2025), SceneFun3D (CVPR 2024), AffordPose (ICCV 2023).
— [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning); [GLOVER citation](https://arxiv.org/pdf/2504.15650)

**Training-free 3D**
- 3D-TAFS (arXiv 2409.10078) and "Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge" (arXiv 2510.08316) exist. — [3D-TAFS](https://arxiv.org/pdf/2409.10078); [2510.08316](https://arxiv.org/html/2510.08316v2)

### Inferences
- **Verdict on 3D AffordanceNet / OpenAD / LASO: moderate for mug, low for others.**
  - Mug is in the training distribution with grasp and wrap-grasp per-point labels, so a handle/body split via "grasp" vs "wrap-grasp" points is plausible.
  - Their "support" is learned on furniture (tables/beds), so a tray or cutting board might be labeled support by open-vocabulary transfer, but this is untested.
  - Inputs are normalized, complete (or synthetic partial-view) object point clouds. A single RealSense view of a glass mug would be far out of distribution.
  - In simulation (CoppeliaSim) we can sample complete point clouds from the meshes, which suits these models well.
- **2D grounding models (OOAL, AffordanceLLM, GLOVER):** give a heatmap or mask for "grasp", "hold" or "handle" on an RGB image. Back-projected with depth, that is a candidate handle region. They give no per-face semantics and no confidence calibration.
- **Recommendation:** affordance detectors can at best supply a "grasp region" prior. UTAMP-style face-level support/grasp should be computed geometrically from the member OBB plus physics (face normal, area, gripper-width test).

### Gaps
- The verbatim 3D AffordanceNet category→affordance table, including whether Mug includes "support", is unverified.
- Code, license and weights status for GLOVER, RAM, UAD and AffordanceLLM were not confirmed.
- No source evaluated any of these on trays, cutting boards, salt shakers or frying pans.

---

## Q4. Part segmentation: PartNet, PartSLIP/++, PartDistill, SAM-based, open-vocabulary part segmentation, Find3D, SAMPart3D, PartField

### Takeaway
- **2D image models:** VLPart (MIT, PACO/Pascal-Part trained, custom text prompts) is the most practical off-the-shelf 2D model for a "mug handle" / "pan handle" mask.
- **3D shape models:** PartField, SAMPart3D and Find3D give zero-shot part decompositions of whole 3D shapes. Find3D also takes text queries. All three need a mesh or reasonably complete point cloud. That is good for CoppeliaSim meshes but weaker for single real views.
- **PartNet-trained methods:** limited to PartNet's 24 categories, which include Mug, Bowl and Bottle but no pan, tray or board **[pan/tray/board absence UNVERIFIED]**.

### Cited Findings
**PartNet**
- Mo et al., CVPR 2019: 573,585 part instances over 26,671 3D models in 24 categories. Three tasks: fine-grained semantic, hierarchical and instance segmentation. — [arXiv 1812.02713](https://arxiv.org/pdf/1812.02713); [mlanthology](https://mlanthology.org/cvpr/2019/mo2019cvpr-partnet/)
- Bottle, Bowl and Mug are among the 24. A derived PartNet-Symh table lists mug with at most 4 parts per shape, bowl 3, bottle 5. — [PartNet_symh](https://github.com/FENGGENYU/PartNet_symh)
- The mug hierarchy (handle / body / containing things / bottom) is **[UNVERIFIED]**.

**PartSLIP / PartSLIP++**
- PartSLIP: "Low-Shot Part Segmentation for 3D Point Clouds via Pretrained Image-Language Models" (Liu et al., CVPR 2023, arXiv 2212.01558). It uses GLIP on multi-view renderings, then superpoint lifting. — [arXiv html](https://arxiv.org/html/2212.01558v2)
- PartSLIP++ (zyc00/PartSLIP2) is MIT-licensed. It is a GLIP + SAM detection-based pipeline with an improved superpoint algorithm. — [GitHub PartSLIP2](https://github.com/zyc00/PartSLIP2)
- Input is a full-object coloured point cloud with multi-view renders, evaluated on PartNet-Ensembled (PartNet-Mobility) categories.

**PartDistill**
- "3D Shape Part Segmentation by Vision-Language Model Distillation" (arXiv 2312.04016, CVPR 2024). Category-specific 2D→3D distillation. Code and data are only partially released (two categories). — [arXiv](https://arxiv.org/pdf/2312.04016); [Find3D paper noting release status](https://arxiv.org/html/2411.13550v2)

**Find3D** ("Find Any Part in 3D")
- Ziqi Ma, Yisong Yue, Georgia Gkioxari, ICCV 2025. Billed as the first method that is at once open-world, cross-category, part-level and feed-forward. It takes a point cloud plus a free-text part query. — [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Find_Any_Part_in_3D_ICCV_2025_paper.pdf); [arXiv html](https://arxiv.org/html/2411.13550v2)
- Its code license was not found.

**SAMPart3D**
- Yunhan Yang et al., arXiv 2411.07184 (2024). Zero-shot, multi-granularity part segmentation.
  - Backbone: a 3D backbone (Point Transformer V3) distilled from text-agnostic 2D foundation models, pretrained on Objaverse, using point clouds sampled from mesh surfaces.
  - Labels: scale-conditioned part features are clustered, then a VLM names parts from multi-view renders, so text is not a segmentation input.
  - Introduces the PartObjaverse-Tiny benchmark (200 objects).
— [arXiv html](https://arxiv.org/html/2411.07184v1)
- Repo, license and per-shape optimization time not confirmed.

**PartField**
- Liu et al. (NVIDIA Toronto AI Lab), ICCV 2025, arXiv 2504.11451. A feed-forward model predicting a continuous part feature field for 3D shapes. Clustering the field yields a hierarchical part decomposition. Only a single 3D feed-forward pass is needed at inference. Code at nv-tlabs/PartField. — [project page](https://research.nvidia.com/labs/toronto-ai/partfield-release); [mlanthology](https://mlanthology.org/iccv/2025/liu2025iccv-partfield); [arXiv html](https://arxiv.org/html/2504.11451v1)
- License not confirmed (NVIDIA research repos are often non-commercial) **[UNVERIFIED]**.
- **Later related work:** P3-SAM (arXiv 2509.06784), GeoSAM2 (arXiv 2508.14036) and UniPart (arXiv 2609.12898, zero-shot language-grounded 3D part segmentation for embodied interaction). — [P3-SAM](https://arxiv.org/pdf/2509.06784); [GeoSAM2](https://arxiv.org/pdf/2508.14036); [UniPart](https://arxiv.org/pdf/2609.12898)

**2D open-vocabulary part segmentation**
- **VLPart:** "Going Denser with Open-Vocabulary Part Segmentation" (Sun et al., ICCV 2023), facebookresearch/VLPart. MIT license, with some components under CLIP/Detic/dino-vit-features licenses. Outputs open-vocabulary objects plus their part masks. Model zoo available. Parts can come from a predefined taxonomy (Pascal Part, PACO) or a custom text prompt. Training uses PACO's 641K part masks. — [GitHub](https://github.com/facebookresearch/VLPart); [ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Going_Denser_with_Open-Vocabulary_Part_Segmentation_ICCV_2023_paper.pdf)
- **PACO:** Ramanathan et al., CVPR 2023. 75 object categories, 456 object-part categories, 55 attributes, 641K part masks over 260K object boxes (LVIS images + Ego4D). The part taxonomy was mined from web queries like "parts of a car". — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ramanathan_PACO_Parts_and_Attributes_of_Common_Objects_CVPR_2023_paper.html)
- **[UNVERIFIED]:** PACO categories include mug, cup, pan (for cooking), tray and chopping board, with parts like handle, body, rim, base and bottom. Check the facebookresearch/paco taxonomy before relying on this.
- **OV-PARTS:** NeurIPS 2023 Datasets & Benchmarks (arXiv 2310.05107). Built on refined Pascal-Part-116 and ADE20K-Part-234. Settings: generalized zero-shot, cross-dataset and few-shot part segmentation. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/dde53059fdb0f45e1e9ad9c66997d662-Paper-Datasets_and_Benchmarks.pdf)
- **PartGLEE:** arXiv 2407.16696 (ECCV 2024 **[venue UNVERIFIED]**). A foundation model that parses parts top-down from their parent objects and unifies object- and part-level training. — [arXiv](https://arxiv.org/pdf/2407.16696)
- **Multi-granularity OV part segmentation (PartCLIPSeg):** arXiv 2406.11384. — [arXiv](https://arxiv.org/html/2406.11384v1)

### Inferences
- **Verdict per model family:**
  - **VLPart (2D): best near-term option** for "mug handle", "pan handle" and "tray handle" masks from RGB. Its top-down detect-object-then-parts structure matches UTAMP's parent-box → member hierarchy.
  - **PartGLEE:** similar fit to VLPart.
  - **Find3D (3D, text query):** good fit in simulation, where complete object point clouds from CoppeliaSim meshes are available. A query like "handle" gives a point mask, and an OBB follows from it.
  - **PartField / SAMPart3D (3D, unlabeled clusters):** good for automatically splitting meshes into members, but labels need a VLM or heuristics. PartField's hierarchy can provide "body" vs "handle" at a coarse level.
  - **PartSLIP++:** requires few-shot fine-tuning per category. Fine as an offline mesh-authoring tool.
  - **PartNet-supervised segmenters:** category-limited; not recommended.
- **"Edge" members** (e.g. tray rims, board edges) are not a standard part class anywhere. Expect "rim" or "edge" queries to be unreliable. A geometric definition (thin slabs at the parent-box boundary) is more robust.
- **Offline vs online:** UTAMP members are static per object model, so we could run PartField/Find3D/SAMPart3D **offline on the CAD meshes** once, fit OBBs in the object frame, and attach them to the object pose at runtime. This avoids real-time perception of parts entirely.

### Gaps
- Licenses for Find3D, SAMPart3D and PartField were not verified.
- PACO membership of mug, pan, tray, chopping board and salt shaker was not verified.
- No benchmark numbers were found for these methods on our specific objects.

---

## Q5. Handle detection for robotic grasping, and which methods yield a handle mask from RGB-D for OBB fitting

### Takeaway
- **Modular approach:** the most directly applicable pipeline is AnyPart-style: GroundingDINO detects the object, an open-vocabulary part segmenter (VLPart-style) gives the "handle" mask, depth back-projection gives a point set, and an OBB is fitted to it. This works for opaque mugs and pans at about 61–70% end-to-end grasp success in the published real-robot evaluation.
- **3D-only alternative:** heat-kernel-signature + XGBoost handle segmentation, which explicitly covers mugs, pans and pots.
- **Glass mug:** needs special handling, because transparent objects break depth.

### Cited Findings
- **AnyPart / "Open-Vocabulary Part-Based Grasping"** (van Oort et al., arXiv 2406.05951)
  - Setup: RGB-D from a wrist camera, a text prompt of the form (object, part) such as "mug, handle", and a parallel gripper.
  - Pipeline: GroundingDINO for the object box, then open-vocabulary part segmentation, then 6-DoF grasp prediction.
  - v1 results: 69.52% grasp success; correct part chosen 88.57% of the time ignoring robot failures; 360 trials over 28 household objects; about 800 ms inference.
  - v2 results: 60.8% grasp success in clutter with 60× faster inference.
  - Contributes two datasets.
  — [arXiv v1](https://arxiv.org/abs/2406.05951v1); [arXiv v2 html](https://arxiv.org/html/2406.05951v2); [review](https://www.themoonlight.io/review/open-vocabulary-part-based-grasping)
- **OVAL-Prompt** (arXiv 2404.11000): LLM-grounded open-vocabulary affordance localization. In real-robot trials, mug handle segmentation succeeded 10/10 but pickup only 7/10. Failures were mostly imprecise segmentation that included the whole object. — [arXiv](https://arxiv.org/pdf/2404.11000)
- **"Object Handle Segmentation in 3D Point Cloud for Robot Grasping Using Scale Invariant Heat Kernel Signature With Optimized XGBoost Classifier"** (2025). Targets slender handle regions on mugs, pans and pots. Uses SI-HKS descriptors, XGBoost per-point classification, and skeleton post-processing to remove non-graspable regions. — [ResearchGate](https://www.researchgate.net/publication/396238022_Object_Handle_Segmentation_in_3D_Point_Cloud_for_Robot_Grasping_Using_Scale_Invariant_Heat_Kernel_Signature_With_Optimized_XGBoost_Classifier)
- **PointGrasp** (arXiv 2403.12631): point-cloud handle segmentation on YCB objects, including a skillet and a mug. — [arXiv](https://arxiv.org/pdf/2403.12631)
- **Self-supervised 3D shape/viewpoint estimation** (arXiv 1910.07948): predicts hidden mug parts (handles) from single images, which let a PR2 grasp mugs by the handle. — [arXiv](https://arxiv.org/pdf/1910.07948)
- **Pose-agnostic functional grasping** (arXiv 2606.21148) notes mug handles are thin, curved and instance-dependent, so small pose errors cause empty grasps or collisions. — [arXiv](https://arxiv.org/pdf/2606.21148)
- **Transparent objects:** A4T, "Hierarchical Affordance Detection for Transparent Objects Depth Reconstruction and Manipulation", addresses affordance detection plus depth reconstruction for transparent objects (relevant to the glass mug). — [ResearchGate](https://www.researchgate.net/publication/362032958_A4T_Hierarchical_Affordance_Detection_for_Transparent_Objects_Depth_Reconstruction_and_Manipulation)

### Inferences
- **Candidates that can produce a handle mask from one RGB-D view** (ranked by practicality):
  1. **VLPart, or GroundingDINO + SAM/SAM 2 with a "handle" prompt** (2D mask), back-projected via depth.
     - Fit an OBB to the handle points with PCA or a minimum-volume box.
     - To keep it "aligned to the parent object box", project onto the parent OBB axes and take axis-aligned extents in the parent frame.
     - AnyPart is the published reference pipeline.
  2. **IIT-AFF/UMD-trained AffordanceNet / AffKernel / OOAL_umd:** use the "grasp" mask as the handle and "wrap-grasp" (or contain) as the body. This only works for cup/mug/pan-like objects.
  3. **3D methods (SI-HKS handle segmentation, Find3D "handle", OpenAD "grasp"):** need a fused or complete point cloud.
- **Practical recommendation for UTAMP-2 in CoppeliaSim:**
  - Perception is not the bottleneck there, because ground-truth meshes and poses are available.
  - Run PartField or Find3D (or VLPart on rendered views) **offline** on each mesh to propose body, handle and edge members, fit OBBs in the object frame, then manually verify.
  - Compute grasp/support per OBB face geometrically:
    - support: face normal near +z when placed, face area at least the footprint of the supported object, flatness.
    - grasp: face-pair separation below the gripper max opening, with approach direction collision-free.
  - Use learned affordance labels only as a tiebreaker or prior.
- **Glass mug:** RGB part segmentation still works, but depth back-projection will be unreliable on real sensors. In simulation this is moot.

### Gaps
- No published method directly outputs "grasp/support per bounding-box face". This appears to be a genuine gap, to be filled by geometric reasoning (no source found).
- Code availability for the SI-HKS handle segmentation paper and for AnyPart was not confirmed.
- No quantitative handle-mask IoU numbers for frying pans or trays were found.
