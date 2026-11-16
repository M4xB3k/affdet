# ACRONYM: A Large-Scale Grasp Dataset Based on Simulation
**Authors**: Clemens Eppner, Arsalan Mousavian, Dieter Fox
**Year / Venue**: arXiv Nov 2020 (arXiv:2011.09584); ICRA 2021 per the task brief. The arXiv page and the PDF do not name a venue. Secondary confirmation: the reference list of Contact-GraspNet (arXiv:2103.14127) cites it as IEEE International Conference on Robotics and Automation (ICRA), 2021. This is evidence from a citing paper, not from ACRONYM itself.
**Cite key**: eppner_2021_acronym
**Source read**: https://arxiv.org/abs/2011.09584 (abstract page) and https://arxiv.org/pdf/2011.09584 (full text, read via local text extraction). The arXiv id is correct.

## In one paragraph
ACRONYM is a dataset of 17.7 million parallel-jaw grasps for 8,872 ShapeNetSem objects from 262 categories, where every grasp is labelled by running a physics simulation (FleX): close the fingers, shake, and check whether the object is still held. The goal is to give learning-based grasp planners much more shape variety and dense grasps per object (2,000 per object) than earlier datasets, while using physics rather than an analytic metric. The authors also provide a way to build cluttered scenes on support surfaces by reusing the single-object labels and marking grasps that collide with scene geometry as failures. Re-training two existing planners (6-DOF GraspNet and GPD) on ACRONYM improves their performance by 2 to 3 times on unseen categories.

## Input and output
- The dataset provides no images. It provides grasps per object mesh, plus code (pyrender based) to render depth images, segmentation masks and point clouds.
- Grasp parameterisation: g = (g_e, g_i), g_e in SE(3) the external degrees of freedom (6) and g_i in R^n the internal DoF of the hand. For the parallel-jaw gripper n = 1, using the Franka Panda gripper model. Each grasp is really a pre-grasp (gripper open with an aperture of 8 cm) that may become a stable grasp when the fingers close.
- Label per grasp: binary success from the simulator (stored with the grasp pose). Average success per object varies (Fig. 3).

## Architecture
Not applicable: it is a dataset. As baselines they re-train 6-DOF GraspNet (both the c-VAE and a c-IMLE variant) and GPD with their original architectures and identical capacity, changing only the training data.

## How grasp labels are made
- Objects: ShapeNetSem meshes, with multi-component meshes excluded and the rest made watertight. Uniform density is assumed (centre of mass equals the geometric centre). All objects have the same friction coefficient 1.0 and density 150 kg/m^3. Scale: the longest side of the axis-aligned bounding box is drawn uniformly between 6 cm and 35 cm.
- Candidate sampling: antipodal sampling. A random surface point plus a line inside a cone around the surface normal; the intersection with the mesh gives the second contact; the gripper pose is the midpoint with a uniformly sampled rotation about that line. Rejection: grasps are discarded if hand and object collide or if the volume between the fingers contains no object. 2,000 grasps per object that pass the test.
- Simulation (FleX): objects, palm and fingers are rigid bodies; there is no gravity. The gripper is an unconstrained position-controlled body. The fingers close with a velocity controller until a force threshold is reached or the hand is fully closed (the force value is not given in the text I read). Then a shaking motion: the hand moves up and down along its approach direction, then rotates about a line parallel to the prismatic joint axes of the fingers. Success is recorded if the object is still in contact with both fingers afterwards.
- Check against gravity: shaking without gravity has 91% precision and 93% recall against labels from a simulation that includes gravity.
- Totals: 17.744 million simulated grasps, 59.21% succeed (about 10.5 million).
- Real-world validity (from other papers, as cited here): Danielczuk et al. executed 2,625 real grasps and found FleX to have an average precision of 86%, the highest among compared models. Kappler et al. found simulation-based metrics more predictive of real success than analytic ones.

## Evaluation and headline results
- Metric: success-coverage curves (the area under the curve), on held-out objects rendered from a fixed random viewpoint. Categories sampled uniformly.
- Training 6-DOF GraspNet (c-VAE and c-IMLE) and GPD on ACRONYM instead of the smaller earlier dataset (206 objects, 6 categories, 7.07 M grasps in the paper's table) improves their performance on unseen categories by a factor of 2 to 3 (stated in the text; the curves are in figures).
- On held-out instances of the five categories of the earlier dataset the ACRONYM-trained models are also better, even though ACRONYM-trained models see no boxes or cylinders.
- All methods perform clearly worse on ACRONYM than on the older dataset, which the authors take as a sign that it is a harder, longer-lived benchmark.
- GPD's failures mostly come from occlusion boundaries, because it only processes points between the fingers.

## How it differs from our setup
- Closest to our label pipeline: it is the large-scale precedent for simulation-labelled grasps. We can cite it to justify "physics simulation instead of analytic metrics". The authors argue analytic measures do not always transfer to the real world.
- The ACRONYM success label is "still touching both fingers after shaking, with no gravity". It is a clamp-and-retain test, not a lift against gravity, and it does not test tilt. Ours (held: followed the lift with slip under 5 cm; level: tilt under 30 degrees) is stricter and has a gravity lift. The authors' own comparison (91% precision, 93% recall) shows shaking in zero gravity almost matches gravity labels, but that is for retention, not tilt.
- Candidate generation: ACRONYM samples antipodal 6-DoF candidates (2,000 per object), so positives are concentrated on geometrically plausible grasps and 59% succeed. Ours is an exhaustive top-down grid of positions and 6 yaws, with no antipodal prefilter. Our positive rate is therefore a different quantity, and our map includes obviously bad positions as negatives, which is part of what a heatmap needs.
- Scale: 8,872 objects and 262 categories versus our 185 objects. Our held-out object split is small by comparison; the paper's lesson is that variety (many categories) matters more than model capacity.
- Parameters: they fix friction 1.0, density 150 kg/m^3 and uniform density. We should record our friction, density and force settings in the same way, because supervisors will ask.
- Gripper: Panda gripper with 8 cm aperture, which is the same opening as ours (8 cm).
- Worth copying for the three-way separation: report separately (a) pre-grasp feasibility (collision-free, non-empty closing volume: the geometric filter), (b) clamp success (retained during shaking) and (c) lift and level. Their pipeline already separates the geometric filter (rejection sampling) from the physics label, and the scene version adds a third filter (collision with scene geometry).

## Likely supervisor questions
1. Why is your label not just ACRONYM's? Answer: ACRONYM labels 6-DoF antipodal pre-grasps with a zero-gravity shake. We need top-down pinch outcomes under gravity on a dense grid, plus a tilt criterion, so we simulate our own labels in MuJoCo, using the same idea of physics-based labels.
2. How do you know simulated labels transfer to the real world? Answer: ACRONYM cites 86% average precision for FleX against 2,625 real grasps. We have not validated MuJoCo on a real robot, and we say so; the supervisors accepted simulation as ground truth for this thesis.
3. Why only 185 objects? Answer: it is a limitation and, per ACRONYM, the main lever for generalisation is object variety. The results on held-out objects should be read with that in mind.

## Not verified
- The venue (ICRA 2021) in ACRONYM's own text: not stated; confirmed only through the reference list of Contact-GraspNet.
- The finger closing force threshold and the finger velocity used in FleX.
- The exact numbers of the 2 to 3 fold AUC improvement for each model (given only in figure legends).
- Figures for the real-world validity of FleX come from other papers as cited in this paper; I did not read those papers.

## Verification (second reader, 2026-11-16)
Checked against the extracted full text (arXiv 2011.09584): 17.7 M grasps, 8,872 objects, 262 categories, 2,000 grasps per
object, friction 1.0 and density 150 kg/m^3 for all objects, size 6-35 cm, Panda aperture 8 cm, closing until "force
threshold is reached or the hand is fully closed" (no value given), success = still "in contact with both fingers" after
shaking, gravity check 91 % precision / 93 % recall, 59.21 % success, 2,625 real grasps and 86 % AP (cited from Danielczuk
et al.), earlier dataset 206 objects / 7.07 M grasps. All match. Note for our defence: our pad friction 1.0 matches theirs;
our objects use friction 0.6 and MuJoCo takes the larger of the two geoms, so the contact friction is 1.0 as well.

## BibTeX
```bibtex
@inproceedings{eppner2021acronym,
  title     = {ACRONYM: A Large-Scale Grasp Dataset Based on Simulation},
  author    = {Eppner, Clemens and Mousavian, Arsalan and Fox, Dieter},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2021},
  note      = {arXiv:2011.09584; venue confirmed only via the reference list of arXiv:2103.14127}
}
```
