# Off-the-shelf parallel-jaw (Franka Panda) grasp detectors and primitive/superquadric grasp learning, for a canGraspFrom(obj, member, palm-face, finger1-face, finger2-face) front end

Research method note: arxiv.org, ar5iv, sites.google.com, dougsm.github.io, raw.githubusercontent.com and the GitHub API were all unreachable from this environment (DNS failures or proxy 403). Evidence therefore comes from rendered GitHub repo pages (via WebFetch), search-engine snippets of arXiv/CVF/RSS/ResearchGate pages, and readthedocs pages. LICENSE file contents could not be opened directly for most repos. Where a fact rests only on my prior knowledge, it is placed under Gaps or marked "(unverified, from memory)".

## Q1. Contact-GraspNet (Sundermeyer et al., ICRA 2021): training data, gripper, I/O, license, PyTorch ports

### Takeaway
Contact-GraspNet (TF2, NVlabs) takes a single-view depth map (+K, optional segmentation) or raw Nx3 point cloud and outputs 6-DoF grasps anchored at observed contact points, with a predicted gripper width and confidence. It was trained on ~17M simulated ACRONYM grasps on ShapeNet meshes. ACRONYM is (from memory) a Franka Panda gripper dataset, but I could not confirm "Panda" against a primary source here. The code is reported by a third party to be under a non-commercial NVIDIA license (License.pdf). An unofficial PyTorch port with weights exists.

### Cited Findings
- Paper: "Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes", Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, Dieter Fox, ICRA 2021. arXiv 2103.14127. Code: github.com/NVlabs/contact_graspnet — [GitHub NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet); [ACM DL record](https://dl.acm.org/doi/10.1109/ICRA48506.2021.9561877)
- Training data: the Acronym dataset with ShapeNet meshes made watertight, plus 10,000 generated table-top scenes with contact-grasp annotations. Scene generation takes "~3 days in a single thread" — [GitHub README](https://github.com/NVlabs/contact_graspnet)
- Trained on 17 million simulated grasps (ACRONYM: 8,872 ShapeNet meshes, 17.7M grasps simulated under varying friction) and "generalizes well to real world sensor data" — [arXiv 2103.14127 (search snippet)](https://arxiv.org/pdf/2103.14127); [ACRONYM site (search snippet)](https://sites.google.com/view/graspdataset)
- ACRONYM dataset license: CC BY-NC 4.0. The ACRONYM FAQ mentions a Franka Panda gripper model in a question about the gripper reference frame — [ACRONYM site (search snippet)](https://sites.google.com/view/graspdataset)
- Repo contains `gripper_models` and `gripper_control_points` folders. The README text does not name the gripper — [GitHub README](https://github.com/NVlabs/contact_graspnet)
- Grasp representation: observed 3D points are treated as potential grasp contacts. Each contact point is paired with a grasp, which "reduces the dimensionality of the grasp representation to 4-DoF". A predicted grasp width w and a baseline-to-base offset d reconstruct the full 6-DoF pose. An ablation shows that removing weighted binning in the width loss lowers success rate and coverage, so width is an explicit output head — [arXiv 2103.14127 (search snippet)](https://arxiv.org/pdf/2103.14127); [ResearchGate copy](https://www.researchgate.net/publication/350457061_Contact-GraspNet_Efficient_6-DoF_Grasp_Generation_in_Cluttered_Scenes)
- Inputs: (a) a .npy/.npz with a depth map in meters, camera matrix K and an optional 2D segmentation map, or (b) an Nx3 point cloud (`xyz`, optional `xyz_color`). Outputs: a 6-DoF grasp distribution with contact confidences, filtered by thresholds — [GitHub README](https://github.com/NVlabs/contact_graspnet)
- Real-robot result: over 90% success grasping unseen objects in structured clutter, halving the failure rate of a recent SOTA baseline (the baseline was not identified in the snippets; it is likely 6-DOF GraspNet, unverified) — [arXiv 2103.14127 (search snippet)](https://arxiv.org/pdf/2103.14127); [ResearchGate](https://www.researchgate.net/publication/355434800_Contact-GraspNet_Efficient_6-DoF_Grasp_Generation_in_Cluttered_Scenes)
- Software and hardware: tested with Python 3.7, TensorFlow 2.2, CUDA 11.1. Inference needs a GPU with at least 8 GB VRAM; training needs at least 24 GB VRAM and 64 GB RAM. The repo has about 10 commits (low maintenance) — [GitHub README](https://github.com/NVlabs/contact_graspnet)
- License: the repo ships a `License.pdf`, which I could not open. A third-party NVIDIA-adjacent repo (facebookresearch/NGDF) states "ContactGraspNet is licensed under a non-commercial NVidia License" — [NGDF README (search snippet)](https://github.com/facebookresearch/NGDF/)
- PyTorch port: elchun/contact_graspnet_pytorch is explicitly "not an official implementation", and its results "may not match the results in the original paper". Weights are included in `checkpoints/`. It uses the same depth+K+(optional) segmentation or xyz input, was tested with Python 3.9, needs at least 8 GB GPU for inference, uses FastSAM for segmentation in its demo, and also ships a License.pdf. It has about 29 commits and 87 stars — [GitHub elchun/contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch)

### Inferences
- Contact-GraspNet is the most directly "face-mappable" detector: each grasp has an observed contact point (finger 1 contact on the visible surface), a baseline (closing) direction, an approach direction and a width. The second contact is approximately contact + w * baseline. Both fingertip contacts and the palm/approach axis can be expressed in the member's bounding-box frame without extra geometry.
- It works on single-view depth plus K, which is exactly what a CoppeliaSim vision sensor provides. Synthetic depth is noise-free, so the sim-to-sim domain gap is small, since training data was also rendered depth. Optional segmentation (CoppeliaSim can render object-ID masks for free) lets it produce per-object grasps ("local regions"), which is convenient for per-object binning.
- The non-commercial license is fine for academic UTAMP-2 research but should be stated.

### Gaps
- I could not open License.pdf to quote the exact terms. "NVIDIA Source Code License – Non-commercial" is my unverified recollection.
- Explicit confirmation that the training gripper is the Franka Panda hand (I believe ACRONYM grasps are for the Panda gripper, which is why default control points match the Panda) could not be read from a primary page.
- Inference runtime figure was not retrieved.
- Real-robot platform used in the paper's experiments was not confirmed.

## Q2. AnyGrasp / GraspNet-1Billion (Fang et al.): license, gripper, input

### Takeaway
AnyGrasp is distributed only as a closed binary SDK that needs a per-machine license (feature ID submitted via a form; replies within about 5 workdays). It takes an (N,3) point cloud and returns graspnetAPI GraspGroup objects (rotation, translation, width, height, depth, score) for a generic parallel-jaw gripper. GraspNet-1Billion data, code and models (graspnet-baseline) are free for non-commercial use only.

### Cited Findings
- Citations: Fang et al., "AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains", IEEE T-RO 2023 (arXiv 2212.08333); Fang et al., "GraspNet-1Billion: A large-scale benchmark for general object grasping", CVPR 2020; Wang et al., "Graspness discovery in clutters for fast and accurate grasp detection", ICCV 2021 — [GitHub graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)
- Licensing: "we can only release the SDK library file of AnyGrasp in a licensed manner" (IP concerns). Users must submit their machine's feature ID via a Google Form, and replies usually arrive within 5 workdays. A July 4, 2026 update replaced the license tool (`lib_cxx.so` and `license_checker` were removed). Expiry and commercial terms are not stated on the README — [GitHub anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)
- Platform: Python 3.6–3.14, PyTorch 1.7.1+ with CUDA 11.x/12.x/13.x, and MinkowskiEngine v0.5.4. An aarch64 build is in testing (July 2026). The repo includes `grasp_detection` and `grasp_tracking` folders, and a July 2026 update adds steering grasps toward specific regions and approach vectors — [GitHub anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)
- SDK API: `detector.get_grasp(points, ...)` takes an (N,3) float32 point cloud in meters and returns a `graspnetAPI.GraspGroup` of parallel-jaw grasps. Options include `dense_grasp` and `collision_detection` (filters grasps colliding with the cloud) — [AnyGrasp SDK docs (search snippet)](https://docsearch.algolia.com/mcp/docs/repo/graspnet/anygrasp_sdk); [BestMan AnyGrasp wrapper](https://bestman-pybullet.readthedocs.io/en/latest/_modules/Perception/Grasp_Pose_Estimation/AnyGrasp/Anygrasp.html)
- Grasp representation: a 7-DoF pose (rotation, translation, width). Per view, the model scores 48 grasps (12 in-plane rotations x 4 approach depths) and predicts 48 widths. A later version adds a stability score that rescales the grasp score at inference — [AnyGrasp arXiv 2212.08333 (search snippet)](https://arxiv.org/pdf/2212.08333)
- graspnetAPI `Grasp` exposes `translation`, `rotation_matrix` (3x3), `width`, `height`, `depth`, `score`. `GraspGroup` exposes vectorised `translations`, `widths`, `scores` — [graspnetAPI docs](https://graspnetapi.readthedocs.io/en/latest/graspnetAPI.html)
- GraspNet-1Billion: 97,280 RGB-D images with over one billion grasp poses, and an analytic evaluation system — [CVPR 2020 open access](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)
- graspnet-baseline license: data, labels, code and models "belong to the graspnet team, MVIG, SJTU and are freely available for free non-commercial use"; contact them for commercial use — [graspnet-baseline fork README (search snippet)](https://github.com/xiaomi2077/graspnet-baseline)
- GraspGen reports a real-robot overall success of 81.3%, outperforming M2T2 and AnyGrasp by 28% and 17.6% respectively — [GraspGen arXiv 2507.13097 (search snippet)](https://arxiv.org/pdf/2507.13097)

### Inferences
- AnyGrasp's gripper model is a generic parallel-jaw box gripper (GraspNet convention: width, height, depth), not a Panda mesh. Width must be clipped to 0.08 m for the Panda, and the depth convention converted to the Panda TCP offset.
- The license key is bound to a machine feature ID, which complicates container, cloud and CI use. graspnet-baseline / GSNet (open, non-commercial) is a friendlier fallback with the same GraspGroup output.
- The GraspGroup fields (center, rotation, width, depth) make face binning simple. In GraspNet convention (unverified, from memory), the rotation matrix's x-axis is the approach direction and its y-axis is the closing direction.

### Gaps
- Exact AnyGrasp license expiry and commercial terms are not stated on the README.
- AnyGrasp runtime and real-world success figures from T-RO were not retrieved here.
- graspnetAPI axis convention (approach = x, closing = y) is from memory and not verified in this session.

## Q3. GraspGen, M2T2, GG-CNN, GR-ConvNet, Edge Grasp Network, VGN, GIGA, 6-DOF GraspNet, and 2024–2026 grasp foundation models: which support Panda out of the box?

### Takeaway
Out-of-the-box Panda support is confirmed for GraspGen (ships Franka Panda, Robotiq-2F-140 and suction models) and VGN (real experiments and ROS launch on a Panda). Contact-GraspNet and 6-DOF GraspNet are very likely Panda (ACRONYM / NVIDIA lineage), but this is unverified here. GG-CNN and GR-ConvNet are planar (top-down) and gripper-agnostic. GraspGen (ICRA 2026) is the strongest current open option. GraspGen-X (CVPR 2026) generalises across grippers.

### Cited Findings
**GraspGen (NVIDIA, 2025; ICRA 2026)**
- "GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training", Murali, Sundaralingam, Chao, Yamada, Yuan, Carlson, Ramos, Birchfield, Fox, Eppner. arXiv 2507.13097. Accepted to ICRA'26. Code v1.0.0 released 07/16/2025; latest listed update 06/21/2026 (Mixture-of-Experts, top-down/side grasps) — [GitHub NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- Grippers: Franka Panda, Robotiq-2F-140, and a 30 mm single-contact suction cup. Other grippers need GraspGen-X — [GitHub NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- Input: segmented object point cloud or mesh (.obj/.stl/.ply/USD). Scene point clouds are supported with optional collision filtering. Segmentation is not included; the README recommends SAM2. Output: 6-DoF poses with discriminator scores, top-k (default 100), saved to YAML. Width is not described as an output — [GitHub NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- Data: over 57M grasps across 8,515 Objaverse-XL (LVIS) objects for the three grippers. The dataset is CC BY 4.0 (the paper abstract says ">53M") — [GitHub](https://github.com/NVlabs/GraspGen); [arXiv snippet](https://arxiv.org/pdf/2507.13097)
- Performance: about 20 Hz before TensorRT and 21x less memory than prior methods. SOTA on FetchBench (Franka Panda, 100 scenes; +17%). 81.3% overall real-robot success, +28% over M2T2 and +17.6% over AnyGrasp — [GitHub](https://github.com/NVlabs/GraspGen); [arXiv 2507.13097 snippet](https://arxiv.org/pdf/2507.13097)
- Code license: LICENSE "Copyright © 2025, NVIDIA Corporation & affiliates", with commercial inquiries to NVIDIA Research Licensing (so effectively non-commercial). Weights are on Hugging Face. Install via uv/pip (Py3.10) or Docker. A PointNet++ CUDA extension must be built. The PTv3 backbone does not run on CUDA 12.8/Blackwell. A ZMQ server and an MCP tool-calling server are provided — [GitHub NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- GraspGen-X (2026): cross-embodiment extension conditioned on a swept-volume gripper representation. 350M grasps (CVPR 2026 version) vs "2 billion" (arXiv version), a discrepancy between versions. Claims zero-shot generalisation to unseen grippers and objects in the real world — [GraspGen-X CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.pdf); [arXiv 2606.00998](https://arxiv.org/html/2606.00998v1)

**M2T2 (CoRL 2023)**
- "M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place", Wentao Yuan, Adithyavairavan Murali, Arsalan Mousavian, Dieter Fox. Input: raw scene point cloud. Output: collision-free gripper poses for 6-DoF grasping and orientation-aware placement, plus contact points and gripper targets. Training data was not released (copyright); only sample scenes are provided. Reported gains are about 19% over the baseline system overall and 37.5% in re-orientation scenes. License text and gripper were not shown on the README render — [GitHub NVlabs/M2T2](https://github.com/NVlabs/M2T2)

**6-DOF GraspNet (ICCV 2019)**
- "6-DOF GraspNet: Variational Grasp Generation for Object Manipulation", Arsalan Mousavian, Clemens Eppner, Dieter Fox, ICCV 2019, pp. 2901–2910. A VAE samples grasps and an evaluator scores and refines them. It takes depth-camera point clouds (object-centric), was trained only in simulation, and reaches 88% real-robot success on common objects — [CVF open access](https://openaccess.thecvf.com/content_ICCV_2019/html/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.html); [NVIDIA Research](https://research.nvidia.com/publication/2019-10_6-dof-graspnet-variational-grasp-generation-object-manipulation)

**VGN (CoRL 2020)**
- "Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter", Breyer et al., arXiv 2101.01132. Real experiments used a Franka Emika Panda with its stock parallel-jaw gripper and a wrist-mounted RealSense D435, in a 30 cm workspace cube, plus a PyBullet replica. Planning takes about 10 ms, and it clears 92% of objects in real clutter without explicit collision checking. The repo includes `panda_grasp.launch` and `sim_grasp.py` — [arXiv 2101.01132](https://arxiv.org/abs/2101.01132); [vgn fork README](https://github.com/vfdev-5/vgn)
- License: BSD-3-Clause (ethz-asl/vgn) — [GitHub ethz-asl/vgn (search snippet)](https://github.com/ethz-asl/vgn)
- A third-party benchmark page lists about 80% real grasp success. I could not verify this against the paper — [codesota](https://www.codesota.com/robotics/grasping/vgn)

**GIGA / Edge Grasp Network**
- GIGA (UT-Austin-RPL), "Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations": MIT license, code "largely based on VGN" — [GitHub UT-Austin-RPL/GIGA (search snippet)](https://github.com/UT-Austin-RPL/GIGA)
- Edge Grasp Network ("A Graph-Based SE(3)-invariant Approach to Grasp Detection", Huang et al.): MIT license. It uses the VGN/GIGA pretrained models and simulation setup for comparison. EdgeGraspNet and GIGA have similar grasp success rates, with EdgeGraspNet higher in declutter rate in one comparison — [Edge Grasp NSF PAR copy](https://par.nsf.gov/servlets/purl/10487651); [GitHub HaojHuang/Edge-Grasp-Network](https://github.com/HaojHuang/Edge-Grasp-Network)

**GG-CNN (RSS 2018) and GR-ConvNet (IROS 2020): planar**
- GG-CNN: Morrison, Corke, Leitner, "Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach", RSS XIV. Predicts per-pixel grasp quality, angle and width from a depth image. Closed loop runs at up to 50 Hz. Reports 83% success on adversarial objects, 88% on household objects moved during grasping, and 81% in dynamic clutter — [RSS proceedings](https://m.roboticsproceedings.org/rss14/p21.html); [arXiv 1804.05172](https://arxiv.org/abs/1804.05172v2)
- GR-ConvNet: Kumra, Joshi, Sahin, "Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network", IROS 2020. 97.7% (image-wise) and 96.6% (object-wise) on Cornell, 94.6% on Jacquard, about 20 ms per prediction. The same table gives GG-CNN 73.0/69.0. Code: skumra/robotic-grasping — [arXiv 1909.04810](https://arxiv.org/pdf/1909.04810); [GitHub skumra/robotic-grasping](https://github.com/skumra/robotic-grasping)
- Source conflict on GG-CNN speed: 19 ms per the GR-ConvNet table vs 6 ms per an RSS summary — [arXiv 1909.04810](https://arxiv.org/pdf/1909.04810); [alphaxiv overview](https://alphaxiv.org/overview/1804.05172v2)

### Inferences
- **Panda out of the box:** GraspGen (explicit), VGN (explicit real Panda setup; GIGA and Edge Grasp inherit the VGN simulation gripper, unverified), Contact-GraspNet, 6-DOF GraspNet and M2T2 (ACRONYM / NVIDIA lineage; Panda very likely, unverified). AnyGrasp and GraspNet-baseline use a generic parallel-jaw model; clip width to 80 mm. GG-CNN and GR-ConvNet are planar 4-DoF (x, y, θ, width), so only top-down grasps are possible, mapping to palm-face = "on". That is too restrictive for handle or edge grasps on a pan or cutting board.
- **Recommended shortlist for CoppeliaSim:**
  1. GraspGen (Panda, object-centric, CoppeliaSim segmentation masks replace SAM2, active in 2026).
  2. Contact-GraspNet (scene-level, explicit contact and width, best for face binning).
  3. VGN or GIGA (TSDF from the sim camera, Panda, BSD/MIT licenses, permissive for any use).
- VGN and GIGA need a TSDF volume within a 30 cm cube. A frying pan with a long handle or a wooden tray may exceed it. This is a practical constraint to check.
- Thin flat objects (cutting board, tray rims): ACRONYM / ShapeNet-trained detectors often favour edge pinches. These map naturally to member = "edge", fingers on on/under faces.

### Gaps
- M2T2, GIGA and Edge Grasp licenses and gripper models were not read from LICENSE files. The M2T2 license is unknown (NVIDIA-style non-commercial is likely, unverified).
- No published, verified study on running these detectors on CoppeliaSim vision-sensor depth specifically.
- Other 2024–2026 parallel-jaw "foundation" models (e.g., ZeroGrasp, EquiGraspFlow, GraspSplats, FoundationGrasp) returned no usable results. They are uncovered here.
- 6-DOF GraspNet code (NVlabs/6dof-graspnet TF, jsll/pytorch_6dof-graspnet) license and status were not verified.

## Q4. Which outputs make it easy to map grasps to "which bounding-box faces contact palm and fingers"?

### Takeaway
Any detector that outputs a full SE(3) gripper frame plus an opening width allows deterministic face binning: approach axis to palm face, closing axis to the two finger faces, and width to a feasibility check. Contact-GraspNet (explicit contact point + width), AnyGrasp/GraspNet (rotation, translation, width, depth) and M2T2 (contact points) are the most convenient. GraspGen gives poses and scores, so the contacts must be recovered by ray casting.

### Cited Findings
- Contact-GraspNet: contact point + 4-DoF per point (approach, baseline, width) — [arXiv 2103.14127 snippet](https://arxiv.org/pdf/2103.14127)
- AnyGrasp/graspnetAPI: `rotation_matrix`, `translation`, `width`, `height`, `depth`, `score` per grasp — [graspnetAPI docs](https://graspnetapi.readthedocs.io/en/latest/graspnetAPI.html)
- M2T2 predicts contact points and gripper targets — [GitHub NVlabs/M2T2](https://github.com/NVlabs/M2T2)
- GraspGen outputs 6-DoF poses + scores (no width described) — [GitHub NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- GG-CNN/GR-ConvNet: per-pixel quality, angle, width (planar) — [RSS 2018](https://m.roboticsproceedings.org/rss14/p21.html); [arXiv 1909.04810](https://arxiv.org/pdf/1909.04810)

### Inferences (proposed binning algorithm; my design, not from a source)
1. Transform the grasp frame into the parent object's box frame (CoppeliaSim gives ground-truth object pose, or use an OBB from the segmented cloud). Assign the member (body/handle/edge) whose sub-box contains the grasp centre, or the midpoint of the two contacts.
2. **Palm face** = the face whose outward normal n maximises −n·a, where a is the approach direction (gripper z). Approaching from above gives "on"; from the +x side gives "front", and so on. Use "inside" when the TCP lies within an interior cavity (mug interior).
3. **Finger faces** = the faces hit by the rays from the centre along ±closing axis b. Equivalently, take the two faces with normals most aligned with ±b. With Contact-GraspNet, use the predicted contact point plus contact + w·b directly.
4. **Feasibility**: the member's extent along b must be ≤ 0.08 m (Panda max opening). Use the predicted width as a cross-check.
5. Aggregate the scores per (member, palm, f1, f2) tuple (max or top-k mean) and threshold to emit canGraspFrom facts. Treat (f1, f2) as an unordered pair because the gripper is symmetric.
- Glass mug: a real depth camera would fail on transparent glass, but a CoppeliaSim depth sensor renders it as opaque geometry. This difference matters only for real transfer.

### Gaps
- No source found that implements exactly this bounding-box face binning.

## Q5. Prior work converting grasp detections into symbolic/discrete grasp predicates or grasp types for task planning

### Takeaway
I found closely related but not identical work. AGFP-Net predicts the feasibility of a discrete set of grasp types alongside action feasibility and feeds it to a symbolic planner as a heuristic. PS-CNN maps objects to primitive classes with parametrised grasp families. Standard TAMP uses predicates such as AtGrasp(o, g) with continuous g. None of the work found uses face-level palm/finger predicates like UTAMP-2's canGraspFrom.

### Cited Findings
- AGFP-Net ("Simultaneous Action and Grasp Feasibility Prediction for ...", IEEE): a multi-task net predicting action feasibility and feasibility of a set of grasp types, used as a heuristic for the symbolic planner — [IEEE Xplore 10341257](https://ieeexplore.ieee.org/abstract/document/10341257/)
- TAMP formulations with grasp predicates such as AtGrasp(o, g) and learned samplers for grasp poses — [Learning Symbolic Operators for TAMP](https://arxiv.org/html/2103.00589); [Guided Search for TAMP using learned heuristics](https://people.eecs.berkeley.edu/~pabbeel/papers/2016-ICRA-tamp-learning.pdf)
- VLM-generated constraints for open-world TAMP (OWL-TAMP) — [arXiv 2411.08253](https://arxiv.org/html/2411.08253v3)
- PS-CNN: each primitive class has parametrised grasp families ranked by priority, a discrete "grasp type per primitive" mapping — [arXiv 1909.08508](https://arxiv.org/abs/1909.08508)

### Inferences
- The binning approach in Q4 is a reasonable novel contribution relative to these works. AGFP-Net is the nearest baseline to cite for "discrete grasp-type feasibility for a symbolic planner".

### Gaps
- Searches did not surface work on handle / edge / wrap grasp-type classifiers feeding a PDDL planner (e.g., grasp-taxonomy-based or affordance-part based such as AffordanceNet-style "grasp" parts). This needs a targeted follow-up.
- AGFP-Net authors and venue details were not retrieved.

## Q6. Superquadric-based grasping and learning from procedural/primitive shapes with transfer to real objects; EGAD

### Takeaway
Superquadric (SQ) grasping goes from single-SQ fitting with a humanoid hand (Vezzani 2017) and clutter (Makhal 2018) to multi-SQ probabilistic recovery (Liu et al., CVPR 2022, EMS) and learning-free antipodal grasps from "hidden superquadrics" (Wu et al., RSS 2023). Learning on procedural/primitive shapes transfers to real objects in several studies (Tobin et al. domain-randomised objects at >90% sim success on unseen real meshes; PS-CNN at 94.2% real success; Bousmalis et al. where procedural shapes matched or beat ShapeNet). EGAD is a set of over 2,000 evolved procedural objects plus 49 3D-printable evaluation objects (BSD-3), meant for training and benchmarking grasp detectors.

### Cited Findings
- Vezzani, Pattacini, Natale, "A grasping approach based on superquadric models", ICRA 2017: superquadrics model both the object and the graspable volume of a humanoid (iCub) hand, for real-time collision-free grasping of unknown objects — [ResearchGate](https://www.researchgate.net/publication/318697655_A_grasping_approach_based_on_superquadric_models)
- Makhal, Thomas, Gracia, "Grasping Unknown Objects in Clutter by Superquadric Representation", IEEE IRC 2018, pp. 292–299, arXiv 1710.02121 — [arXiv 1710.02121](https://arxiv.org/pdf/1710.02121)
- Liu, Wu, Ruan, Chirikjian, "Robust and Accurate Superquadric Recovery: a Probabilistic Approach", CVPR 2022 (oral). Uses a Gaussian-uniform mixture for noise and outliers, MLE solved by Expectation-Maximization-Switching (EMS), and an extension to multi-SQ fitting of complex objects. Code: EMS-superquadric_fitting (MATLAB; a Python port is believed to exist, unverified) — [CVF open access](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Robust_and_Accurate_Superquadric_Recovery_A_Probabilistic_Approach_CVPR_2022_paper.pdf); [Papers with Code](https://www.paperswithcode.com/paper/robust-and-accurate-superquadric-recovery-a)
- Wu, Liu, Liu, Chirikjian, "Learning-Free Grasping of Unknown Objects Using Hidden Superquadrics", RSS XIX 2023 (arXiv 2305.06591). A two-stage method recovers multiple SQs at different positions inside the object from its point cloud and synthesises antipodal grasps per SQ. It was evaluated on isolated and packed scenes with "competitive" performance vs SOTA (numbers not retrieved) — [RSS proceedings](https://www.roboticsproceedings.org/rss19/p042.html); [arXiv 2305.06591](https://arxiv.org/pdf/2305.06591)
- Tobin et al., "Domain Randomization and Generative Models for Robotic Grasping" (arXiv 1710.06425): trains on millions of unrealistic procedurally generated objects, with >90% simulated success on unseen realistic objects and reported transfer to the real world — [arXiv 1710.06425](https://arxiv.org/pdf/1710.06425)
- Bousmalis et al., "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping" (arXiv 1709.07857): random procedural shapes helped real grasping, realistic 3D models were not needed, and procedural objects slightly beat ShapeNet when combined with 10% real data — [arXiv 1709.07857](https://arxiv.org/pdf/1709.07857)
- PS-CNN (Lin et al., ICRA 2020): segments monocular depth into 6 primitive-shape classes, trained purely on synthetic simulation data, then applies primitive-specific grasp families. Reports 94.2% real success on static objects, and 93.5% success with 100% completion in light clutter. A known limitation is that sim and real must share camera setup — [arXiv 1909.08508](https://arxiv.org/abs/1909.08508); [arXiv 2201.00956](https://arxiv.org/abs/2201.00956)
- PS-CNN follow-up (multi-modal primitive scene abstraction): adds cones and hollow variants and uses ICP for 6-DoF grasp parameters. Reports 95% success on unseen objects in the CoppeliaSim simulator — [IEEE 10678767 / ResearchGate](https://www.researchgate.net/publication/383967956_Multi-Modal_Robust_Geometry_Primitive_Shape_Scene_Abstraction_for_Grasp_Detection)
- EGAD: Morrison, Corke, Leitner, "EGAD! An Evolved Grasping Analysis Dataset for Diversity and Reproducibility in Robotic Manipulation", IEEE RA-L 2020 (arXiv 2003.01314). Over 2,000 evolutionarily generated objects spanning shape complexity and grasp difficulty, for training and evaluating visual grasp detectors, plus 49 diverse 3D-printable evaluation objects for reproducible cross-lab benchmarking. BSD-3 license. Code: github.com/dougsm/egad — [arXiv 2003.01314](https://arxiv.org/pdf/2003.01314); [QUT Research Data Finder](https://researchdatafinder.qut.edu.au/display/n18178); [GitHub dougsm/egad](https://github.com/dougsm/egad)

### Inferences
- For UTAMP-2's box-member abstraction, SQ recovery (EMS multi-SQ) is a natural bridge. Each recovered SQ gives an oriented primitive with dimensions, which can serve directly as a member sub-box (body/handle/edge). Antipodal grasps from SQs (Hidden Superquadrics) are analytically tied to SQ axes, so palm/finger face assignment is trivial.
- A hybrid is attractive: run a learned detector (GraspGen or Contact-GraspNet) for robustness, and use SQ or OBB member decomposition for symbolic binning.
- A procedural-primitive training set (boxes, cylinders, handles, slabs, with EGAD-style diversity) generated in CoppeliaSim could provide labelled canGraspFrom examples, consistent with the transfer evidence above. CoppeliaSim has already been used for primitive-based grasp evaluation (the PS-CNN follow-up).
- EGAD is not a superquadric dataset. It is procedurally evolved meshes intended for grasp-detector training and evaluation; its 49 printable objects could serve as a real-transfer test set.

### Gaps
- EGAD generation details (MAP-Elites / CPPN?) and the gripper used for difficulty metrics were not confirmed (sources say only "evolutionary algorithms").
- Success rates for Makhal 2018 and Hidden Superquadrics 2023 were not retrieved.
- Whether Tobin et al. appeared at IROS 2018 is unverified.
- No source found that learns grasps on superquadrics specifically and then transfers to real objects. Learned SQ-based grasp work (e.g., later iCub works by Vezzani et al. 2018 combining SQ with learning) was not retrieved.
