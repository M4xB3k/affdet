# Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics
**Authors**: Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, Ken Goldberg
**Year / Venue**: 2017 / RSS 2017 (arXiv page: "To appear at Robotics: Science and Systems 2017")
**Cite key**: mahler_2017_dexnet2
**Source read**: https://arxiv.org/abs/1703.09312 (abstract page) and https://arxiv.org/pdf/1703.09312 (full text, read via local text extraction). The arXiv id is correct.

## In one paragraph
Dex-Net 2.0 asks: can we learn to pick up objects from a depth image using only synthetic data? The authors build a dataset of 6.7 million synthetic depth images, each paired with a planar parallel-jaw grasp and a robust analytic quality score computed on the 3D mesh. They train a Grasp Quality CNN (GQ-CNN) that takes a depth image crop aligned to the grasp and predicts the probability that the grasp succeeds. At test time they sample many antipodal grasp candidates from the depth image, score each with the GQ-CNN, and execute the best one. The labels come from an analytic metric, not from physics simulation. On a real ABB YuMi robot the planner reached 93% success on 8 known adversarial objects at 0.8 s planning time.

## Input and output
- Input: a depth image (2.5D point cloud) of a single object on a table, from an overhead camera. The image is transformed so that it is centred on the grasp centre pixel and the grasp axis lies along the middle row. The dataset section calls this a 32 x 32 grasp image. A second scalar input is the gripper depth z from the camera.
- Output: one number, the predicted robustness Q in [0, 1] for that candidate grasp.
- Grasp parameterisation: u = (p, phi) with centre p = (x, y, z) and one angle phi in the table plane. So it is a planar grasp, 4 degrees of freedom (the gripper is perpendicular to the table, closing along an axis parallel to the table). In the network the candidate is written u = (i, j, phi, z): pixel position, angle, depth.
- Each object has up to 100 labelled grasps. Candidate grasps at test time are sampled by an antipodal rule on the depth-image gradients, then ranked by the GQ-CNN.

## Architecture
- The Fig. 4 caption says: four convolutional layers in pairs, separated by ReLU nonlinearities, then three fully connected layers, plus a separate input layer for z. The network has approximately 18 million parameters.
- The first-layer filters shown in the paper are 7 x 7 convolution filters (stated in the experiments text). Exact filter counts, strides and fully connected widths for the other layers are in the figure, not in the text: not verified in the paper text.
- Image rotation to the grasp axis is done by a known image transform, so the network does not have to learn rotation invariance.
- Training: SGD with momentum, cross-entropy loss, weight initialisation from a zero-mean Gaussian. Augmentation: flips about both axes and 180 degree rotation (all give an equivalent grasp), plus freshly sampled depth noise (multiplicative Gamma plus Gaussian-process noise) at each batch.

## How grasp labels are made
- Source: analytic metric on meshes, computed by Monte-Carlo sampling. No physics simulator is used for the labels.
- Metric: the robust epsilon quality EQ (a force-closure style quality averaged over uncertainty in object pose, gripper pose and friction coefficient). A grasp is labelled positive if EQ > delta with delta = 0.002 AND the gripper does not collide with the object or table.
- Friction: the friction coefficient is drawn from a truncated Gaussian N(0.5, 0.1) truncated to [0, 1] (appendix). Antipodal candidates are sampled with a fixed mu = 0.6 friction cone.
- Objects: 1,500 meshes (1,371 synthetic from 3DNet plus 129 laser scans from KIT). Each is rescaled to fit a 5.0 cm gripper opening and given a mass of 1.0 kg centred in the bounding box, because some meshes are not closed. Stable poses on the table are computed and used.
- Size: 6.7 million datapoints, about 21.2% positive. Train/validation split 80-20 image-wise.
- Real-robot "success" (evaluation, not training labels): the gripper still holds the object after lifting, transporting and shaking it.

## Evaluation and headline results
- Held-out synthetic validation accuracy of the GQ-CNN on the full dataset: 85.7% (about 1.3 million datapoints).
- Known objects (8 adversarial 3D-printed objects, 80 trials): the GQ-CNN planner reaches 93% success and 94% precision, planning in 0.8 s (Table III caption: the 93% model is GQ-L-Adv, pretrained on Dex-Net and fine-tuned on the 8 objects). Point-cloud registration reaches 95% but is about 3 times slower.
- Novel household objects (10 objects, 50 trials): the GQ planner has the highest success rate among compared methods, 80% success with 100% precision (zero false positives over 29 grasps classified as robust).
- 40 novel objects (100 trials, using the cross-entropy-method planner): 94% success and 99% precision (68 successes out of 69 grasps classified as robust), 2.5 s average planning time.
- Training on 400 physical grasp outcomes alone (GQ-Adv-Phys) gives 80%, which supports the value of large synthetic data.

## How it differs from our setup
- Grasp definition: Dex-Net's label is "analytic quality + collision free". Lifting is not simulated. Lifting and shaking appear only in the real-robot evaluation. Ours is a physical lift test (clamp, lift 50 cm, hold) with a separate "level" label. So Dex-Net's label is closest to our analytic baseline, not to our simulation label.
- Geometry: Dex-Net is also top-down (gripper perpendicular to the table), one yaw angle per candidate. This is the closest published match to our top-down pinch on a grid of positions and yaw angles.
- Output: Dex-Net scores one grasp candidate at a time and takes the argmax. We predict a per-point heatmap on the point cloud ("some yaw works here"). Dex-Net's grasp image is centred on the grasp, which is effectively a local patch classifier. Ours is a global per-point model (max-pool context).
- Input: Dex-Net uses a single-view depth image, and its dataset uses partial synthetic depth with noise. We use a point cloud with normals; our reconstruction experiments (RGB-D view 0.86, single-image 0.82) are the analogue.
- Mass: Dex-Net fixes mass 1.0 kg at the bounding-box centre, a crude choice. Our analytic baseline needs real mass and centre of mass; this is the same weakness.
- Worth copying: report a calibration or "precision above 0.5" view (Dex-Net's precision and robust grasp rate), and keep the success label, the collision label and the planning constraint as separate things. Dex-Net defines S = (quality high) AND (collision free). That is exactly the kind of decomposition the supervisors asked for: separate clamp (quality), lift (physical lift), and level (tilt) labels.

## Likely supervisor questions
1. Why does your label need a physics simulator when Dex-Net used an analytic metric? Answer: analytic metrics assume rigid contacts and a known centre of mass and do not capture slip, tilt or lift. Our simulation label includes lifting and levelness, and Dex-Net itself only checks lifting and shaking on the real robot.
2. Is your task the same as Dex-Net's? Answer: it is closest in geometry (top-down, planar position plus yaw), but Dex-Net scores one grasp at a time with a local image, while we output a dense per-point heatmap with a global feature.
3. Why a heatmap and not the best grasp? Answer: a heatmap keeps all feasible positions for the downstream planner. Dex-Net also keeps a score over many candidates and then takes the maximum.

## Not verified
- Exact channel counts, kernel sizes (apart from 7 x 7 in the first layer) and fully connected widths of the GQ-CNN: shown only in a figure, not in the extracted text.
- Whether the gripper width used for the main dataset objects equals 5.0 cm in all experiments: stated for the dataset, not checked elsewhere.
- Success rate of the plain (non-CEM) planner on the 40-object set: only the CEM result is given in the text I read.

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (our 0.86 and 0.82). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@inproceedings{mahler2017dexnet2,
  title     = {Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics},
  author    = {Mahler, Jeffrey and Liang, Jacky and Niyaz, Sherdil and Laskey, Michael and Doan, Richard and Liu, Xinyu and Aparicio Ojea, Juan and Goldberg, Ken},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2017},
  note      = {arXiv:1703.09312}
}
```
