# Estimating Mass, Material, Fragility and Other Physical Properties from Images, Depth or VLMs (for UTAMP-2 canLift / support predicates)

Scope note: the target scene has a salt shaker, a glass mug, a cutting board, a frying pan and a wooden tray, with a Franka Panda (3 kg payload) in CoppeliaSim. Direct fetches of arxiv.org, proceedings.mlr.press, the project pages and GitHub (DNS and permission errors) did not work in this environment. Every fact below comes from search-engine snippets of primary sources (arXiv HTML/PDF, CVF and ECVA open access, the GitHub README text as indexed by search). Items marked **[UNVERIFIED]** come from my background knowledge or from a secondary summary I could not check against the primary source. Code and data licenses could not be confirmed except for TransCG.

---

## 1. Image2Mass (Standley et al., 2017): data, method, accuracy

### Takeaway
image2mass (CoRL 2017) predicts mass from a single RGB image plus the object's bounding dimensions. It factors mass into volume × density with a shape-aware "geometry module". It was trained on about 150k Amazon products, and its reported mean ALDE (absolute log-difference error) is about 0.47 to 0.67 depending on the test split. It is now a weak baseline on ABO-500. The data is heavily skewed: nearly half of it is cell-phone cases. For our objects it gives a rough prior at best and should not be trusted alone for a 3 kg threshold decision.

### Cited Findings (fact sheet)
- **Paper:** Trevor Standley, Ozan Sener, Dawn Chen, Silvio Savarese, "image2mass: Estimating the Mass of an Object from Its Image", Proc. 1st Conference on Robot Learning (CoRL 2017), PMLR 78:324-333. — [PMLR PDF](http://proceedings.mlr.press/v78/standley17a/standley17a.pdf); [Semantic Scholar](https://www.semanticscholar.org/paper/image2mass:-Estimating-the-Mass-of-an-Object-from-Standley-Sener/09c1c3a7693d5790a2be0cab506debf26069bc95)
- **Data:** the authors collected product information for about 3 million Amazon.com products (weight, dimensions, one image each). After "extensive cleaning" they kept a dataset of about 150k products. They also captured a small set of household objects with a scale and a digital camera for testing. — [PMLR PDF](http://proceedings.mlr.press/v78/standley17a/standley17a.pdf)
- **Inputs:** one product image plus width, height and length. Each sample is image + dims + weight. — [IEICE FIT2023 paper describing image2mass](https://www.ieice.org/publications/conference-FIT-DVDs/FIT2023/data/html/program/pdf/I-011.pdf)
- **Method:** a "shape-aware" model that uses an estimate of the 3D shape. Its parts are a Geometry Module, a Volume Tower and a Density Tower, and their outputs are multiplied (mass = density × volume). The paper says it "outperforms the baselines and compares favorably to the performance of humans" (a separate human study). — [PMLR PDF](http://proceedings.mlr.press/v78/standley17a/standley17a.pdf); [IEICE FIT2023](https://www.ieice.org/publications/conference-FIT-DVDs/FIT2023/data/html/program/pdf/I-011.pdf); [Learning to Weigh Waste, arXiv 2603.00931](https://arxiv.org/pdf/2603.00931)
- **Accuracy (as re-reported by later work):** mALDE 0.470 and mAPE 0.651, per one follow-up. A 2023 paper reports 0.672 for image2mass on the original Amazon test set, against 0.709 for its own model. The two sources differ in metric or protocol. — [Physically Guided Visual Mass Estimation, arXiv 2601.20303](https://arxiv.org/html/2601.20303); [An Extensible Multimodal Multi-task Object Dataset with Materials, arXiv 2305.14352](https://arxiv.org/html/2305.14352)
- **On ABO-500 (100-object test set), from the PhysGS paper's Table 1:** image2mass ADE 12.496 kg, ALDE 1.792, APE 0.976, MnRE 0.341. — [PhysGS arXiv 2511.18570](https://arxiv.org/pdf/2511.18570)
- **Dataset bias:** a critique notes that nearly half of the 150k objects are cell-phone cases. — [arXiv 2305.14352](https://arxiv.org/html/2305.14352) (as surfaced in search snippet)
- **Follow-ups:** Andrade & Moreno, "Improving the Estimation of Object Mass from Images", ICARSC 2023; Image2Mass++ (hybrid condition-guided mass estimation, Measurement/ScienceDirect, 2025); "Physically Guided Visual Mass Estimation from a Single RGB Image" (arXiv 2601.20303, 2026). — [ScienceDirect Image2Mass++](https://www.sciencedirect.com/science/article/abs/pii/S026322412503235X); [arXiv 2601.20303](https://arxiv.org/html/2601.20303)
- **Code/data availability:** **[UNVERIFIED]** I recall a public GitHub release (tstandley/image2mass) with the dataset and model. I could not access GitHub to confirm it or the license.

### Inferences
- image2mass needs the object's dimensions as input. In CoppeliaSim the true dimensions are available, so dims are free. In a real deployment they would come from the depth bounding box, which is unreliable for the glass mug (see Section 5).
- An ALDE of about 0.5 to 0.7 means a typical error factor of e^0.5 to e^0.7, roughly 1.6 to 2×. A frying pan of about 1.0 to 2.5 kg could therefore be predicted anywhere from about 0.6 to 5 kg. That is not discriminative enough near the Panda's 3 kg payload (in practice the margin is smaller once the gripper and the edge-grasp torque are included). This is my own arithmetic on the cited metric.
- The training distribution is retail products with packaging, which is a poor match for cookware and trays.

### Gaps
- I could not read the exact human-comparison figures or the "within factor of 2" style metrics in the original paper (PMLR fetch blocked).
- Code and data license not confirmed.

---

## 2. Physically Grounded VLMs / PhysObjects (Gao et al., ICRA 2024)

### Takeaway
PhysObjects (built on EgoObjects images) holds 39.6K crowd-sourced and 417K automated labels for eight physical concepts. Fine-tuning InstructBLIP on it (PG-InstructBLIP) raised average test accuracy from 66.2% to 87.5%. Plugged into an LLM planner, it raised planning accuracy from 56.9% (base InstructBLIP) to 88.2% on 51 real-scene tasks. The model scores pairwise or categorical judgments ("which is heavier", "is it fragile"), not kilograms. That suits symbolic predicates like canLift or support-fragile when paired with object-class priors, but it cannot check a 3 kg payload threshold directly.

### Cited Findings (fact sheet)
- **Paper:** Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha Majumdar, Dorsa Sadigh, "Physically Grounded Vision-Language Models for Robotic Manipulation", arXiv 2309.02561 (Sep 2023), ICRA 2024. The authors are from Stanford, Google DeepMind and Princeton. — [arXiv HTML v4](https://arxiv.org/html/2309.02561v4); [Princeton listing](https://collaborate.princeton.edu/en/publications/physically-grounded-vision-language-models-for-robotic-manipulati/); [project page](https://iliad.stanford.edu/pg-vlm) (could not fetch); [Google Sites project page](https://sites.google.com/view/physically-grounded-vlms)
  - Author list beyond "Gao, Sarkar, Xia, Xiao et al." is **[UNVERIFIED]** (from recall).
- **Dataset:** PhysObjects contains 39.6K crowd-sourced and 417K automated physical-concept annotations of common household objects. Images come from EgoObjects. Annotations cover **eight physical concepts**. Some sources cite 36.9K crowd-sourced labels, so the count differs between versions. — [arXiv 2309.02561](https://arxiv.org/pdf/2309.02561); [arXiv HTML v4](https://arxiv.org/html/2309.02561v4)
- **The eight concepts** — **[UNVERIFIED in detail]**, from recall and consistent with snippets: mass, fragility, deformability, material, transparency, contents, can-contain-liquid, is-sealed. Snippets confirm mass, fragility, deformability, material, transparency and contents. In my recollection, mass, fragility and deformability are continuous concepts labeled by **pairwise comparison** ("which object is heavier/more fragile"), while the rest are categorical. My recollection is also that **density and liquid capacity are not separate concepts** (density appears only as a held-out generalization concept, if at all). Verify against Table I of the paper.
- **Model:** PG-InstructBLIP is InstructBLIP (Flan-T5-XXL LLM backbone) fine-tuned on PhysObjects. Weights are on Hugging Face as `bidiptas/PG-InstructBLIP`. — [HF model card](https://huggingface.co/bidiptas/PG-InstructBLIP); [Dataloop listing](https://dataloop.ai/library/model/bidiptas_pg-instructblip/)
- **Accuracy (crowd-sourced test, Table II):** base InstructBLIP averages 66.2%. PG-InstructBLIP averages 87.5% and beats the base model on every concept, with the largest gain on *contents*. Per concept, PG-InstructBLIP scores **mass 80.0%, fragility 94.6%, deformability 93.0%, material 87.1%**. Single-concept fine-tuned models average 85.8%. — [arXiv HTML v4](https://arxiv.org/html/2309.02561v4)
- **Generalization:** fine-tuning improves understanding, including on held-out concepts the model was not trained on. The authors attribute this to learning human priors of these concepts from visual appearance. — [arXiv 2309.02561](https://arxiv.org/pdf/2309.02561)
- **Planning:** an LLM planner queries the VLM about scene objects. On 51 real-scene tasks across 8 scenes (graded by a non-author human), planning accuracy was **33.3% with no VLM, 56.9% with InstructBLIP and 88.2% with PG-InstructBLIP**. Per category, PG-InstructBLIP scored single-concept 84.1%, multi-concept 94.4% and common-knowledge 85.7%. On multi-concept tasks the base-VLM planner scored 27.8%, the same as the no-VLM planner. — [arXiv HTML v4](https://arxiv.org/html/2309.02561v4)
- **Real robot:** the plans were evaluated on a real Franka Emika Panda, and the paper reports improved task success. A secondary summary gives 9/10 vs 4/10 success on tasks needing physical reasoning. **[UNVERIFIED]**, since I could not see the paper's table. — [arXiv HTML v4](https://arxiv.org/html/2309.02561v4); [Lacuna summary](https://lacuna.tiptreesystems.com/work/physically-grounded-vision-language-models-for-robotic-manipulation/wrk_672e02e185d664783fea0174456cf484)
- **Code/data:** the dataset is released via the project page. The model weights are on HF. License **[UNVERIFIED]**; note that EgoObjects images carry their own terms.

### Inferences
- PG-InstructBLIP fits **fragile(glass mug) → restrict support** well, since fragility accuracy is 94.6% and transparency/material are also covered. It fits **canLift** less directly, because mass is relative (pairwise) and its accuracy is the lowest of the reported concepts at 80%.
- A practical pattern for UTAMP-2 is to use a VLM (PG-InstructBLIP or a modern VLM) to classify material and category, then look up a mass or density prior per (category, material), then threshold against the payload with a safety margin. Using VLM pairwise mass only to rank objects is a second option.
- InstructBLIP with Flan-T5-XXL is a large model (on the order of 10B+ parameters, **[UNVERIFIED]**). A DINOv2 + small linear head trained on PhysObjects labels is a cheaper alternative, but I found no paper reporting that exact configuration.

### Gaps
- Exact concept list, whether each concept is continuous or categorical, and transparency/contents accuracies were not visible in snippets.
- Real-robot table and dataset/model license were not confirmed.

---

## 3. NeRF2Physics, PUGS, PhysGS and 2024–2026 works estimating mass, friction or material from images or 3D

### Takeaway
A line of zero-shot work does the following: reconstruct the object in 3D (NeRF or Gaussian splats), attach CLIP/language features, ask an LLM for candidate materials with densities, propagate them over the volume and integrate to get mass. ABO-500 is the standard benchmark. NeRF2Physics (CVPR 2024) started the line. PUGS (2025) and PhysGS (2025) report improvements. 2026 preprints (SiPhy, PhysVGGT, Gen2Physics) move to single-image or feed-forward estimation. Reported numbers differ across papers because of protocol, so treat cross-paper comparisons with caution. All of these methods need multi-view posed images, which is fine in simulation but slow. For five known objects, they are overkill unless the object set is open.

### Cited Findings (fact sheets)

**NeRF2Physics: "Physical Property Understanding from Language-Embedded Feature Fields"**
- Albert J. Zhai et al. (UIUC; **[UNVERIFIED]** full author list: Albert J. Zhai, Yuan Shen, Emily Y. Chen, Gefen X. Wang, Xinlei Wang, Sheng Wang, Kaiyu Guan, Shenlong Wang), CVPR 2024. arXiv 2404.04242. Code and project page at ajzhai.github.io/NeRF2Physics. — [arXiv HTML](https://arxiv.org/html/2404.04242v1); [CVPR virtual poster](https://cvpr.thecvf.com/virtual/2024/poster/29832); [Illinois experts](https://experts.illinois.edu/en/publications/physical-property-understanding-from-language-embedded-feature-fi/)
- **Input:** a collection of posed images of an object. **Method:** reconstruct a NeRF, have an LLM propose candidate materials (with property values), use CLIP features to link 3D points to materials, then do zero-shot kernel regression of properties per point. Volume integration with thickness estimation gives mass. — [arXiv HTML](https://arxiv.org/html/2404.04242v1); [Moonlight review](https://themoonlight.io/review/physical-property-understanding-from-language-embedded-feature-fields)
- **Outputs:** dense fields of physical properties. Mass is evaluated on ABO-500 (a subset of Amazon Berkeley Objects with a 100-object test split). Friction and hardness are evaluated on an in-house real-object set with measured friction and hardness. The paper reports better mass estimation than other zero-shot baselines and some supervised baselines. Ablations support the thickness estimation and kernel regression. — [arXiv HTML](https://arxiv.org/html/2404.04242v1)
- **Numbers (as re-reported by SiPhy, 2026):** NeRF2Physics ADE 8.74, ALDE 0.78, APE 1.06, MnRE 0.55 on ABO-500. **[Protocol-dependent; verify against the original Table.]** — [SiPhy arXiv 2607.22355](https://arxiv.org/pdf/2607.22355)
- License **[UNVERIFIED]**.

**PUGS: "Zero-shot Physical Understanding with Gaussian Splatting"**
- Yinghao Shuai et al. (includes Hao Zhao), arXiv 2502.12231 (Feb 2025). Venue **[UNVERIFIED]** (possibly ICRA 2025). Code, data and models at GitHub EverNorif/PUGS. — [arXiv HTML v2](https://arxiv.org/html/2502.12231v2); [radiancefields.com](https://radiancefields.com/papers/pugs-zero-shot-physical-understanding-with-gaussian-splatting)
- **Method:** Gaussian-splatting reconstruction with a geometry-aware regularization loss and a region-aware feature contrastive loss, then feature-based property propagation and Gaussian volume integration. Estimates mass, friction and hardness zero-shot. Claims state of the art on ABO-500 mass and claims it can help real-world grasping. — [arXiv HTML v2](https://arxiv.org/html/2502.12231v2)
- **Conflict:** SiPhy (2026) reports PUGS at ADE 30.30, ALDE 1.59, APE 7.68, MnRE 0.30, which is worse than NeRF2Physics on three of four metrics. That contradicts the PUGS SOTA claim and probably reflects a different protocol or re-run. — [SiPhy arXiv 2607.22355](https://arxiv.org/pdf/2607.22355) vs [PUGS](https://arxiv.org/html/2502.12231v2)

**PhysGS: "Bayesian-Inferred Gaussian Splatting for Physical Property Estimation"**
- arXiv 2511.18570 (Nov 2025). Bayesian GS with uncertainty. It improves mass estimation accuracy by up to 22.8% over deterministic baselines across ABO-500 and two other datasets, and also covers friction and hardness. Its Table 1 includes image2mass (ADE 12.496 / ALDE 1.792 / APE 0.976 / MnRE 0.341). — [arXiv HTML](https://arxiv.org/html/2511.18570v1); [ResearchGate](https://www.researchgate.net/publication/397934508_PhysGS_Bayesian-Inferred_Gaussian_Splatting_for_Physical_Property_Estimation)
- PhysGS's own ADE/ALDE values were not visible in snippets.

**2026 preprints (all [UNVERIFIED] beyond titles and abstracts; recent, not peer-review-confirmed)**
- *SiPhy: Single-Image Physical Property Reasoning* (arXiv 2607.22355). Single-image input. Contains the cross-method ABO-500 table cited above. — [arXiv](https://arxiv.org/pdf/2607.22355)
- *PhysVGGT: Feed-Forward Dense Physical Property Estimation from a Single Image* (arXiv 2609.18920). Reports beating GaussianProperty, AURA, and NeRF/3DGS pipelines on ABO-500 from a single image. Its table includes a GPT-4V row. — [arXiv](https://arxiv.org/pdf/2609.18920)
- *Gen2Physics* (arXiv 2608.23869). Multi-view material decomposition on generated meshes. Better on MnRE, comparable on ADE/ALDE. — [arXiv](https://arxiv.org/pdf/2608.23869)
- *Physically Guided Visual Mass Estimation from a Single RGB Image* (arXiv 2601.20303). — [arXiv](https://arxiv.org/html/2601.20303)
- *Estimating Object Physical Properties from RGB-D Vision and Depth Robot Sensors Using Deep Learning* (arXiv 2507.05029). RGB-D mass estimation for robots. — [arXiv](https://arxiv.org/html/2507.05029)
- *PhysGraph: A Physics-aware 3D Scene Graph* (arXiv 2606.08655). — [arXiv](https://arxiv.org/pdf/2606.08655)

**VLMs as direct mass estimators**
- GPT-4V on 20 portable ABO objects (PhysGen appendix): mean absolute error 0.39 kg, 75% within ±30% of ground truth. This is a small sample. — [PhysGen arXiv 2409.18964](https://arxiv.org/pdf/2409.18964)
- PhysBench (ICLR 2025), a general physical-world understanding benchmark that is not mass-specific: the best model, GPT-4o, reaches 49.49% against about 95.9% for humans, and most VLMs average about 40%. — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2501.16411)
- SIMPACT (arXiv 2512.05955): VLMs predict mass and friction of rigid objects to parameterize a physics simulator for action selection. — [arXiv](https://arxiv.org/pdf/2512.05955)

### Inferences
- For our 5 objects, a VLM zero-shot estimate (GPT-4V-class MAE of about 0.4 kg on portable objects) is likely good enough for coarse classes (salt shaker about 0.1 to 0.3 kg, glass mug about 0.3 to 0.5 kg, cutting board, frying pan about 1 to 3 kg, large wooden tray). It is not reliable for a hard threshold near 3 kg without a margin. My suggestion is to output a calibrated interval and make canLift conservative: treat it as infeasible if the upper bound exceeds payload × margin, or if the grasp-induced moment exceeds limits for edge grasps.
- canLift for the "frying pan cannot be edge-grasped" case is driven by **torque about the grasp point** (mass × lever arm), not just total mass. None of the mass estimators produce a center of mass. NeRF2Physics/PUGS-style dense density fields could in principle give one (my inference), but none of the papers I saw report CoM accuracy.
- In CoppeliaSim, ground-truth masses are set in the scene. A learned estimator mainly matters for sim-to-real or open-set objects, so the planner could use sim ground truth with injected noise matching the reported estimator error, to test robustness.

### Gaps
- PhysGS's own numbers, PUGS's own numbers and NeRF2Physics's original table could not be read directly.
- No CoM or inertia-estimation results were found.
- No code licenses confirmed.

---

## 4. Material recognition datasets and models (MINC, DMS, Materialistic, foundation-model material segmentation)

### Takeaway
MINC (CVPR 2015; 23 categories, 3M point samples) and DMS (ECCV 2022; 44,560 images, 3.2M segments, 52 labels in my recollection) are the main dense material-segmentation datasets. Their classes cover wood, glass, metal, ceramic and plastic, which is enough to label our five objects. Materialistic (SIGGRAPH/TOG 2023, not CVPR) does exemplar-based "select similar material" rather than classification. Foundation-model material classifiers (DINOv2-based FMMC, 2026; SAMa, 2024) exist, but evidence about them is thin.

### Cited Findings (fact sheets)
- **MINC (Materials in Context Database):** Sean Bell, Paul Upchurch, Noah Snavely, Kavita Bala, "Material Recognition in the Wild with the Materials in Context Database", CVPR 2015. 23 material categories, built from OpenSurfaces plus crowdsourcing, about 3 million material samples. Best segmentation is about 73.1% mean class accuracy (CNN + fully connected CRF). — [CVF open access](https://openaccess.thecvf.com/content_cvpr_2015/papers/Bell_Material_Recognition_in_2015_CVPR_paper.pdf); [CVF supplement](https://www.cv-foundation.org/openaccess/content_cvpr_2015/ext/2B_016_ext.pdf). A 2025 paper describes MINC as over 430,000 images. — [Nature Heritage Science 2025](https://www.nature.com/articles/s40494-025-01888-4). Co-author list beyond Bell **[UNVERIFIED]**.
- **DMS (Dense Material Segmentation):** Paul Upchurch, Ransen Niu, "A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing", ECCV 2022 (arXiv 2207.10614). 44,560 indoor and outdoor images with 3.2 million dense segments. A model trained on DMS generalizes more consistently across datasets than one trained on OpenSurfaces. The DMS-25 model reaches about 0.777 per-pixel accuracy on validation. — [ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680440.pdf); [arXiv](https://arxiv.org/html/2207.10614?context=cs). **[UNVERIFIED]:** 52 material labels, released by Apple (apple/ml-dms-dataset); license unconfirmed.
- **Materialistic:** Prafull Sharma et al., "Materialistic: Selecting Similar Materials in Images", arXiv 2305.13291, published in ACM Transactions on Graphics 42(4), 2023 (SIGGRAPH), **not CVPR**. It frames the task as similarity-based grouping from a user-clicked pixel, which means it selects regions of the same material rather than naming the material. — [arXiv abs](https://arxiv.org/abs/2305.13291)
- **Foundation-model material work:** FMMC, "Harnessing the Power of Foundation Models for Accurate Material Classification" (arXiv 2603.17390, 2026; uses DINOv2 per snippet). SAMa, "Material-aware 3D Selection and Segmentation" (arXiv 2411.19322, 2024). MaRI, material retrieval (arXiv 2503.08111). MaterialSeg3D (arXiv 2404.13923) segments dense materials for 3D assets from 2D priors. — [FMMC](https://arxiv.org/pdf/2603.17390); [SAMa](https://arxiv.org/pdf/2411.19322); [MaRI](https://arxiv.org/pdf/2503.08111); [MaterialSeg3D](https://arxiv.org/html/2404.13923v1). Methods and accuracy not verified.
- **NeRF2Physics** and **PUGS** themselves use CLIP features plus LLM-proposed materials as an open-vocabulary material labeler. — [arXiv 2404.04242](https://arxiv.org/html/2404.04242v1)

### Inferences
- A "DINOv2 features → material head" module is plausible: train a linear or MLP probe on DMS or MINC crops. FMMC suggests DINOv2 is competitive for material classification, but I have no accuracy figure. Material then maps to density (glass about 2.5 g/cm³, steel/cast iron about 7.8, wood about 0.5 to 0.8 are standard engineering values) and to fragility (glass and ceramic → fragile). Combined with a volume estimate, this is essentially the image2mass/NeRF2Physics factorization.
- In CoppeliaSim renders, material appearance depends on the shaders and textures used. A classifier trained on real photos (MINC/DMS) may transfer poorly to sim renders, so it is safer to use sim object metadata or a VLM on renders.

### Gaps
- Licenses for MINC, DMS and Materialistic were not confirmed.
- No benchmark of DINOv2 linear probes on DMS was found.

---

## 5. Transparent object perception (ClearGrasp, TransCG, Dex-NeRF) and implications for the glass mug

### Takeaway
Commodity depth sensors return missing or wrong depth on transparent glass. Real-robot evidence: a PhoXi-based grasp pipeline scored 0/10 on transparent objects where Dex-NeRF scored up to 10/10. Depth-completion methods (ClearGrasp, TransCG/DFNet) restore usable geometry and report 72% (ClearGrasp, parallel-jaw) to above 85% grasp success. For the glass mug, any depth-based grasp detector, support-surface estimator or volume-based mass estimate needs depth completion, an RGB-only or NeRF-based pipeline, or simulated depth that ignores transparency.

### Cited Findings (fact sheets)
- **ClearGrasp:** Shreeyak Sajjan et al. (Google/Columbia), "ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation", ICRA 2020, arXiv 1910.02550. Predicts surface normals, transparent masks and occlusion boundaries from RGB, then completes depth. It reports **86% picking success with suction and 72% with a parallel-jaw gripper** on a real robot when combined with off-the-shelf manipulation algorithms. — [arXiv HTML](https://arxiv.org/html/1910.02550v2). **[UNVERIFIED]:** synthetic training set of about 50k images plus a real test set, Apache-2.0 code.
- **TransCG:** Hongjie Fang, Hao-Shu Fang, Sheng Xu, Cewu Lu, "TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline", IEEE RA-L 7(3), 2022 (arXiv 2202.08471). **57,715 RGB-D images from 130 scenes**, with ground-truth depth, surface normals and transparent masks in cluttered scenes. Proposes DFNet (RGB + noisy depth → refined depth). A real-robot experiment shows robust grasping of novel objects. Data at graspnet.net/transcg. **Dataset license CC BY-NC-SA 4.0** (non-commercial). A later repo fix reduced depth shifting to about 2 cm. — [GitHub README](https://github.com/Galaxies99/TransCG/blob/main/README.md); [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.08471)
- In the SAID-NeRF evaluation, TransCG-based point clouds failed mostly by grasping "noisy pixels that float around in the air", even after outlier removal. SAID-NeRF scored highest, with 10/10 on its objects. — [SAID-NeRF arXiv 2403.19607](https://arxiv.org/pdf/2403.19607)
- **Dex-NeRF:** Jeffrey Ichnowski*, Yahav Avigal*, Justin Kerr, Ken Goldberg, "Dex-NeRF: Using a Neural Radiance Field to Grasp Transparent Objects", CoRL 2021, arXiv 2110.14217 (authors and venue **[UNVERIFIED]**, from recall). On a physical ABB YuMi it reached 10/10 on several transparent objects, while a **PhoXi depth camera baseline scored 0/10 on all of them**. In simulation, success plateaus between 80% and 98% at about 50k to 60k NeRF iterations. — [arXiv PDF](https://arxiv.org/pdf/2110.14217)
- **Recent:** ClearDepth (stereo, arXiv 2409.08926) reports 86.2% average multi-finger grasp success. DCIRNet (arXiv 2506.09491) reports a 44% improvement in grasp success for transparent and reflective objects with DexGraspNetV2. TransGraspNet (arXiv 2607.29567) reports 91% overall (96% simple, 86% hard scenes over 100 attempts). DistillGrasp (2408.00337) and GAA-TSO (2503.17106) are further depth-completion methods. A 2026 Robotics (MDPI) paper uses VLM-guided transparent-object perception for task-oriented grasping. — [ClearDepth](https://arxiv.org/html/2409.08926v3); [DCIRNet](https://arxiv.org/html/2506.09491); [TransGraspNet](https://arxiv.org/pdf/2607.29567); [DistillGrasp](https://arxiv.org/pdf/2408.00337); [GAA-TSO](https://arxiv.org/pdf/2503.17106); [Robotics 15(7):135](https://doi.org/10.3390/robotics15070135)
- These numbers come from different robots, objects and protocols, so they are not directly comparable. — (search synthesis across the above)

### Inferences
- **CoppeliaSim caveat:** CoppeliaSim's simulated vision/depth sensors render depth from geometry, so a glass mug produces clean depth unless transparency artifacts are deliberately simulated (my understanding of rasterized depth sensors; **[UNVERIFIED]** for CoppeliaSim specifically). Sim results for grasping the glass mug will therefore overestimate real performance. Consider injecting dropout or noise into depth on transparent masks to emulate real sensors.
- On a real robot, the glass mug's affordances (graspable rim or handle, a top surface that can support items) depend on completed depth. A wrong or missing mug top surface would corrupt the support predicate as well as grasp detection. Transparency detection also signals fragility, since transparent usually means glass, and can flag lower confidence for depth-based geometry.

### Gaps
- No head-to-head comparison of ClearGrasp, TransCG and Dex-NeRF under one protocol was found.
- Licenses for ClearGrasp and Dex-NeRF were not confirmed.

---

## 6. Work linking estimated mass to liftability or grasp-feasibility predicates in task planning

### Takeaway
I found no paper that estimates mass from vision and turns it into a canLift(obj, effector) predicate checked against payload in a symbolic or TAMP planner. The closest are: PG-VLM (VLM physical concepts feeding an LLM planner); a 2026 human-robot assembly planner with a payload-aware capability predicate (masses come from design data, not vision); SIMPACT (VLM-predicted mass/friction feeding a simulator); and OWL-TAMP and ViLaIn (VLM-grounded predicates and constraints for planners). UTAMP-2's mass- and material-dependent canLift/support predicates would be a reasonably novel combination.

### Cited Findings
- PG-VLM: an LLM planner queries a physically grounded VLM (mass, fragility, etc.), raising planning accuracy from 56.9% to 88.2%, with real Franka evaluation. — [arXiv 2309.02561v4](https://arxiv.org/html/2309.02561v4)
- "From Perception to Symbolic Task Planning: Vision-Language Guided Human-Robot Collaborative Structured Assembly" (arXiv 2601.00978, 2026): VLMs map RGB-D scenes to symbolic states, with a **payload-aware capability predicate**. Component masses come from design data, not the VLM. The paper reports VLM state-inference accuracy (GPT-5 highest, Claude-Opus-4.5 lowest among those tested). **[UNVERIFIED in detail]**. — [arXiv](https://arxiv.org/pdf/2601.00978)
- SIMPACT (arXiv 2512.05955): a VLM predicts mass and friction of rigid objects to set up a physics simulator for action planning. — [arXiv](https://arxiv.org/pdf/2512.05955)
- OWL-TAMP (arXiv 2411.08253): VLM-generated open-world constraints as code, checked by a TAMP system together with grasp stability and kinematics. — [arXiv](https://arxiv.org/html/2411.08253v3)
- ViLaIn (arXiv 2304.08587 and OpenReview): a VLM grounds PDDL initial-state predicates from images. No mass handling. — [arXiv](https://arxiv.org/pdf/2304.08587); [OpenReview](https://openreview.net/pdf?id=P6kGDik2JK)
- Task-oriented manipulation with VLMs (arXiv 2410.15863): objects annotated with attributes including mass from a fine-tuned VLM. — [arXiv](https://arxiv.org/html/2410.15863v1)
- PUGS claims its property estimates help real-world grasping (e.g., grip force from mass and friction). — [arXiv 2502.12231](https://arxiv.org/html/2502.12231v2)

### Inferences
- Recommended design for UTAMP-2, as a synthesis rather than an established method:
  1. Get material, transparency and fragility from a VLM (PG-InstructBLIP or a modern VLM) or a DINOv2 probe trained on DMS plus PhysObjects labels.
  2. Estimate mass as a category/material prior (or density × volume from geometry). Keep an uncertainty interval, using reported error levels (image2mass ALDE about 0.5 to 0.7; GPT-4V about 0.4 kg MAE on portable objects) to size it.
  3. canLift(obj, member) := upper_mass × g × lever_arm(member) < torque limit, and upper_mass < payload − gripper mass. The frying pan edge-grasp then fails on the lever arm, and the wooden tray fails on size or mass and is slid instead.
  4. support(fragile obj) := allow only objects with mass below a small threshold. Fragility comes from material = glass.
- Because the targeted objects are a fixed set of five, a lookup-table baseline (category → mass interval, material, fragility) is the strong, honest baseline that any learned module must beat. That makes the learned module mainly a generalization claim.

### Gaps
- No paper found that directly evaluates vision-estimated mass → liftability predicate accuracy, or planner success under mass-estimation noise.
- The UTAMP-2 paper itself (Niaz, Piater & Agostini, arXiv 2608.25641) was not checked in this pass.
