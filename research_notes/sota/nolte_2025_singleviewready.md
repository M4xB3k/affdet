# Is Single-View Mesh Reconstruction Ready for Robotics?
**Authors**: Frederik Nolte, Andreas Geiger, Bernhard Schoelkopf, Ingmar Posner (Oxford Robotics Institute; University of Tuebingen; Max Planck Institute for Intelligent Systems)
**Year / Venue**: 2025 (v1 23 May 2025, v2 11 Aug 2025, v3 5 Sep 2026). arXiv comments field says accepted at BMVC 2026. 49 pages. The PDF I read is v3.
**Cite key**: nolte_2025_singleviewready
**Source read**: https://arxiv.org/abs/2505.17966 (abstract page, fetched). Full text: https://arxiv.org/pdf/2505.17966 (v3, downloaded, converted with pdftotext; introduction, desiderata, results, discussion and the metrics and depth appendices read). Found by web search for task 7. Other search hits (Triplane Grasping, DreamGrasp, Shape Completion with Prediction of Uncertain Regions, Single-View Shape Completion for Robotic Grasping in Clutter) were seen only as search snippets and were not read.

## In one paragraph
The paper asks whether today's single-image mesh reconstruction models are good enough to build a physics "digital twin" of a tabletop scene for robot manipulation. The authors define five robotics requirements: surface accuracy (5 mm Chamfer target), no collisions between reconstructed objects, stable resting pose, robustness to occlusion, and speed (under 2 s per scene). They test thirteen models on real robotics datasets (YCB-Video and Aria Digital Twin). Most models fail most requirements. They also run a grasp test: grasps computed on the reconstructed mesh are tried on the true mesh in simulation. Only about half transfer. They also relate Chamfer error to grasp success object by object. This is the closest paper I found to "how much does reconstruction error cost a downstream grasp".

## Input and output
- Input to the reconstruction models: one RGB view of an object (ground-truth masks are supplied so that segmentation does not matter). Each model gets its own recommended crop and resize.
- Output evaluated: a complete mesh per object (or per scene for scene-level models).
- Alignment before measuring error: scale is estimated by the median ratio of principal-component standard deviations between ground truth and reconstruction; translation and rotation by ICP from 512 evenly spread initial rotations. So the authors estimate scale and pose from the ground truth mesh. This is close to our own "optimistic" alignment (we give size and pose even more directly).
- Scale and pose of the models themselves: the paper treats missing scale and pose as something to be fixed by alignment, and notes that scene-level models do not preserve relative object poses.

## Architecture
Not a model paper. Models evaluated (13): object-level SF3D, InstantMesh, One-2-3-45, LGM, Michelangelo, ZeroShape, Real3D, DSO, TRELLIS.2; scene-level MIDI, Gen3DSR, PhysGen3D, SAM3D. Note: TripoSR, LRM and the original TRELLIS are not in the list (checked by searching the text). TRELLIS.2 is a different, later model.

## Training data and loss
Not applicable. Evaluation data: 177 objects from YCB-Video and 136 objects from the Aria Digital Twin dataset (hand-picked frames); 28 YCB scenes for scene-level models. Grasp test on YCB-Video: antipodal grasp sampling for a Franka Panda parallel gripper. A grasp from the reconstruction counts as transferred if it does not collide with the true mesh, the gripper and surface normals differ by at most 22.5 degrees, and the object stays in the gripper after a shake in simulation.

## Headline results and known failure cases
- Reconstruction accuracy: on YCB-Video most models give about 5 mm Chamfer distance; on Aria the median error is about double. The authors say models struggle most with non-canonical views (from the top or bottom of the object).
- Grasp transfer: only about half of the grasp poses computed on a reconstruction can be applied to the true mesh (Figure 3, per-model rates are shown only as a plot).
- Error versus task: per-object Pearson correlation between Chamfer distance and grasp transfer success is r = -0.55 (-0.52 without objects that had no valid grasp). A threshold effect: Chamfer error above about 1 to 2 cm almost always means grasp failure. Below about 1 cm, outcomes depend on things global Chamfer does not capture: normals at the contact points, watertightness, edges and thin structures. Their example: InstantMesh and ZeroShape have similar Chamfer on YCB but InstantMesh transfers worse.
- Occlusion: reconstruction error at occluded parts is 40 to 105 percent worse than at visible parts.
- Stability and collisions are further, independent failure modes.
- Limits the authors state: Chamfer is a global metric that misses local contact quality; the relation between error and success depends on the task (a cylinder grasp is more forgiving than a pen pinch); only mesh outputs are studied (Gaussian outputs are left out). My own reading of the setup: grasp success is judged in simulation on YCB and household objects, so it is a proxy for real grasping.

## How it relates to our camera route
- It supports three of our own findings and one decision. (1) Reconstruction error does cost grasp performance, but non-linearly: small global errors can still fail, large ones nearly always fail. This matches our drop from 0.86 (RGB-D view) to 0.82 (TripoSR). (2) Thin structures and local surface detail are where global metrics fail, which fits our worst cases (trays, woks, plates). (3) Our alignment (true size, true pose) is at least as generous as theirs, so our TripoSR numbers are optimistic.
- Decision support: in the depth appendix the authors say depth "can resolve location/scale ambiguities" and that RGB-D should be used more explicitly. They also report that a model that needs depth (Gen3DSR) failed when real RGB-D holes were given. That supports using the lab RGB-D camera, with attention to missing depth values.
- Difference from our test: they transfer grasp poses and measure collision and shake in simulation; we train heatmap networks and measure AUC against simulation labels. So their r = -0.55 is a reference point, not a calibration for us. A direct analogue would be to plot our per-object AUC loss against Chamfer error of the reconstruction, per object.
- TripoSR is not tested here, so we should not quote this paper as evidence about TripoSR specifically.

## Likely supervisor questions
1. Is there evidence from other work that reconstruction error hurts grasping? Answer: yes. Nolte et al. (2025) report a moderate negative correlation (r = -0.55) between Chamfer error and grasp transfer success, with failure almost certain above 1 to 2 cm and unpredictable below.
2. Why do you prefer RGB-D over single-image reconstruction? Answer: single-image models give about 5 mm to 1 cm error on common objects and fail at occluded regions. The same paper recommends depth to resolve scale and location. A measured partial cloud keeps true size.
3. Does low Chamfer error guarantee a good heatmap? Answer: no. The paper shows low-error meshes can still fail for local reasons such as contact normals and thin structures. We should report heatmap AUC per object type and not rely on a single global shape metric.

## Not verified
- The per-model grasp transfer percentages (only plotted in Figure 3).
- Whether models run on cropped crops (their recommended preprocessing) match our rendered white-background setting.
- The paper has no TripoSR, LRM or original TRELLIS results.
- Venue taken from the arXiv comments field only.
- I did not read the other candidate papers from the search (Triplane Grasping, DreamGrasp, Shape Completion with Prediction of Uncertain Regions, Single-View Shape Completion for Robotic Grasping in Clutter). They may also contain relevant measurements.

## BibTeX
```bibtex
@inproceedings{nolte2025singleview,
  title     = {Is Single-View Mesh Reconstruction Ready for Robotics?},
  author    = {Nolte, Frederik and Geiger, Andreas and Sch{\"o}lkopf, Bernhard and Posner, Ingmar},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026},
  note      = {arXiv:2505.17966}
}
```
