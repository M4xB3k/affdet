# Inter-object affordances: contact heatmaps and simulation labels

Research note after the supervisor meeting of 14 Oct 2026, with a working MuJoCo prototype.

## 1. What was asked

- **Inter-object affordances**, not unary face labels matched afterwards.
  - The model looks at both objects and decides whether object 1 can be supported by object 2 (`isP2PSupportive`).
  - Proof of concept: can grasp and can support. Part 2: can lift and can slide.
- **Exact labels.** They must come from physics, not from VLM guesses.
- **Contact regions as heatmaps.** Which pixels of each object take part in the interaction. A heatmap is more precise and less restrictive than a box face. The box-face fact UTAMP-2 needs can still be read off it.
- **Simulation for labelling.**
  - For support: does B really stay on A?
  - Later, for grasp and lift: does the object stay in the gripper, and how does it swing?
  - The simulation produces the label; the network only sees images.

## 2. Short answers

| Question | Answer |
|---|---|
| Can we simulate support with what we have? | Yes. The MuJoCo prototype in `affdet/sim/` already does it. |
| Is it reliable? | After calibration it gives all 28 primitive cases with known answers right. It agrees with the static labeller on 44 of 48 shaker faces; the 4 differences are borderline objects. |
| Is it pairwise, as asked? | Yes. Same mug, two partners: a board placed on it stays, a shaker placed on it falls into the cup. Matching per-object face labels cannot express this. |
| Can contacts become pixel heatmaps? | Yes. Contact regions from the settled simulation are rendered as visible and amodal (hidden) masks. |
| The robot cannot see the bottom of a shaker. | Use an amodal heatmap: the label marks where the hidden contact surface projects into the image, as in amodal segmentation. The network learns it from shape, as people infer a hidden flat base. |
| Do we have everything? | Nearly. Meshes, sizes, rough masses and collision shapes are covered. Missing: friction per material, a better mass model for hollow objects, a gripper model for grasp and lift, and cleaned member boxes. See §5. |

## 3. Heatmaps: what exists and what fits us

### Closest prior work

- **O2O-Afford** (Mo et al., CoRL 2021) is the closest match.
  - Input: object A as a point cloud and a partial scan of scene object B. Output: a point-wise affordance heatmap on B.
  - Tasks: placement, fitting, pushing and stacking.
  - Labels come from simulated interactions in SAPIEN on ShapeNet models, with no human annotation.
  - It is the template for "simulate the interaction, label the contact points, learn a pairwise heatmap" ([arXiv 2106.15087](https://arxiv.org/abs/2106.15087), [project page](https://cs.stanford.edu/~kaichun/o2oafford), [PMLR](https://proceedings.mlr.press/v164/mo22b.html)).
  - Differences from us: it works on point clouds, not RGB plus DINO features. It predicts on one object of the pair, and it does not output planner predicates.
- **Where2Act** (Mo et al., ICCV 2021) predicts a per-pixel actionability map from RGB-D, also trained from simulated interactions in SAPIEN. It shows that per-pixel maps can be learned from simulation outcomes ([arXiv 2101.02692](https://arxiv.org/abs/2101.02692)).
- **Placement work** with simulated stability labels:
  - Placeit! (2025): evolutionary search for placements that settle under gravity, with contact-quality heatmaps ([arXiv 2510.09267](https://arxiv.org/html/2510.09267)).
  - Seeing the Unseen (2024): placement masks for objects not yet in the image ([arXiv 2401.07770](https://arxiv.org/abs/2401.07770)).
  - Learning to Regrasp by Learning to Place (2021): a stability classifier for placements ([arXiv 2109.08817](https://arxiv.org/abs/2109.08817)).
- **Grasp contact data:**
  - GraspGen (NVIDIA) provides simulated Franka grasps on Objaverse LVIS objects. That is a ready source of finger contact regions for the grasp heatmap ([GitHub](https://github.com/NVlabs/GraspGen)).
  - Contact-GraspNet labels grasps by executing them in a physics simulator ([arXiv 2103.14127](https://arxiv.org/abs/2103.14127)).
  - ContactGen represents grasps as contact maps ([arXiv 2310.03740](https://arxiv.org/abs/2310.03740)).
- **Hidden regions:** no affordance paper predicts hidden contact surfaces (the underside) from one view. The nearest is amodal segmentation and completion:
  - [Amodal panoptic segmentation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Mohan_Amodal_Panoptic_Segmentation_CVPR_2022_paper.pdf).
  - [Contact-aware amodal completion, ICCV 2025](https://arxiv.org/html/2508.00427v1).

  An amodal contact heatmap would be new.

The gap: no published model takes RGB of two objects plus metric size and returns a support score with contact heatmaps on both objects, trained on simulated stability. That is our contribution.

### Proposed output and how it handles occlusion

**Input.** One scene image, with the objects standing apart as the robot sees them. We take a crop for A and a crop for B, plus each box size in metres. Two crops from one image suffice; we do not need two separate photos.

**Output per ordered pair (B on A):**

1. **support score:** B stays on A. This is the `isP2PSupportive` fact.
2. **heatmap on A's crop:** where B would touch A. For a mug that is the rim, which is visible from above.
3. **heatmap on B's crop:** where B touches A. For a shaker that is its base, which is hidden.

**Two label types per heatmap:**

- **visible:** the contact surface as seen.
- **amodal:** the contact surface projected through the object.

The amodal map is the useful one. The shaker's base is never seen, but its footprint projects to a well-defined image region, and the network can learn it from shape cues, as amodal segmentation does.

**Back to UTAMP-2 facts.** Sum each heatmap's heat per box face of the member box. The hottest face of A and of B gives `isP2PSupportive faceA A faceB B`. The heat centroid gives the placement point, replacing the paper's "always the centroid" assumption.

**Grasp uses the same format.** A heatmap of finger contacts from successful simulated grasps (§4.4), summed per face, gives `canGraspFrom`.

## 4. Simulation

### 4.1 Which simulator

| | MuJoCo | PyBullet | CoppeliaSim | Isaac Sim |
|---|---|---|---|---|
| Install | `pip install mujoco` (3.16 works here) | no Python 3.13 wheel; build fails here | desktop app; UTAMP-2 uses it | GPU, heavy |
| Contacts | soft, accurate, rolling friction (condim 6) | fine | Bullet, ODE, Newton or MuJoCo engines | PhysX; unstable at high mass ratios in one study ([CRISP](https://arxiv.org/pdf/2609.21761)) |
| Franka gripper | MuJoCo Menagerie | yes | in the UTAMP-2 scene | yes |
| Role | label generation (fast, headless, CPU) | — | final check on the 5 benchmark objects in their own scene | not needed |

MuJoCo is generally rated strong for contact-rich manipulation ([guide](https://roboticscenter.ai/rl-environments/mujoco-vs-isaac-sim)). A dense-stack study reports jitter and drift with many stacked objects ([GS-Playground](https://arxiv.org/pdf/2604.25459)). We stack only two objects, and calibration covers this.

### 4.2 What the prototype does (`affdet/sim/mujoco_support.py`)

- **Collision shapes.**
  - A table only ever touches an object's convex hull, so the hull is exact for "rests on a face".
  - Object-on-object contact needs the cavities (mug, pan, tray), so those use a CoACD convex decomposition. That takes 5–90 s per object once, then it is cached ([obj2mjcf uses the same tool](https://github.com/kevinzakka/obj2mjcf)).
- **Mass.** Spread uniformly over the decomposed volume. For a thin-walled mug the decomposition follows the walls, so its centre of mass is roughly right.
- **Face test (`face_holds`).**
  1. Drop the object once, untilted, to find its natural rest pose. Tapered bodies rest a few degrees off their box face.
  2. Tilt it 8° in 8 directions and release each time.
  3. It holds only if every run returns to that rest pose (within 3°) and has stopped moving in the last second.

  This is the dynamic version of the static tip-angle rule. A flat face rocks back. A round side has no restoring torque, so it stays tilted or rolls away.
- **Calibration.**
  - Box, 64-sided cylinder, hexagonal and 12-sided prisms, domed cap, tall cylinder and cone: 28 of 28 faces correct (`affdet/tests/test_sim_support.py` covers 5 of them).
  - Three bugs found and fixed on the way:
    - front and back faces were swapped;
    - drift was measured at the body origin, which swings on tall objects;
    - there was no rolling friction, so a tilted mill circled on its rim like a spinning coin.
- **Real shakers.** 12 objects × 4 faces, 35 s in total. It agrees with the static labeller on 44 of 48. The 4 differences are all borderline:
  - a tall 4.5 cm mill nudged by 8°;
  - two small flat caps used upside down;
  - one ridged side.

  The real question there is how large a disturbance a support must survive. That is a parameter to agree on, for example the robot's placement error.
- **Stack test (`stack_test`).** A stands on the table; B is set centred on A's top (UTAMP-2's placement rule), released, and simulated for 2 s. B passes if it did not drop, tip or slide off. `reports/figures/sim_stacks.png`:

  | Pair | Result |
  |---|---|
  | shaker on board | stays |
  | shaker on mug | falls into the mug (drops 9 cm) |
  | board on mug | stays (41 contacts along the rim) |
  | mug on shaker | falls (drops 6 cm) |
  | mug on board | stays |

- **Contact heatmaps (`contact_region_heat`, `heatmap_render.py`).**
  - In the settled pose, every point of A's surface within 3 mm of B is "contact". MuJoCo reports only about 3 solver points for flat-on-flat contact, so proximity is used instead.
  - These regions are rendered from a robot-like camera as visible (red) and amodal (orange) masks.
  - `reports/figures/sim_contact_heatmaps.png` shows four cases:
    - the shaker footprint on the board;
    - the shaker's hidden base;
    - the mug rim;
    - the hidden ring under the board.

### 4.3 Label generation plan for support

For each ordered pair (A, B) of curated objects:

1. A's upward face: usually `on`, for trays and boards also `in`.
2. B's downward face: every face where B rests on the table.
3. Placement: centred plus offsets of a few millimetres, to stand in for placement error.
4. Simulate with mass and friction drawn from their ranges.
5. The label is the fraction of runs that stay, plus the contact heatmaps from the stable runs.

Cost: about 1 s per stack test. 319 objects in 5 categories give about 100k ordered pairs, so subsample: all cross-category pairs plus a sample within each category. It runs on CPU and parallelises trivially.

### 4.4 Grasp and lift (part 2, but the same machinery)

1. Put a Franka hand (MuJoCo Menagerie) at sampled antipodal grasps, or at GraspGen grasps.
2. Close it, lift and shake.
3. Record whether the object stays, how far it swings, and the finger contact patches.

The finger patches are the grasp heatmap. Success after lifting with a realistic mass gives `canLift` per member, which covers the paper's pan-held-by-the-rim case.

## 5. Do we have everything for simulation?

| Needed | Status |
|---|---|
| Meshes | ✓ 46 shakers labelled; 71 mugs, 64 boards, 60 pans, 78 trays curated. 13 still need an up-axis fix. |
| Metric size | ✓ ObjaTHOR height, scaled by us. Coarse (5 cm steps), so randomise ±10–20 %. |
| Mass | ~ ObjaTHOR value, 0.5 kg steps. Fine for single-object rest tests, where mass does not matter. For pairs only the mass ratio matters, so sample a range. Better: VLM material × volume. |
| Mass distribution | ~ Uniform over the decomposed volume. Hollow sealed objects (shakers) are solid in the sim. Use the empty/full bounds from our earlier discussion if a label is sensitive. |
| Collision shapes | ✓ Convex hull for table tests; CoACD for concave objects (cached). |
| Friction | ✗ Defaults (0.6) now. Needs a per-material table (wood, ceramic, glass, metal, plastic), with materials from the VLM, plus randomisation. |
| Contact softness | ✓ MuJoCo defaults, validated on primitives. |
| Gripper | ✗ Not yet. The Franka hand from MuJoCo Menagerie is the plan. |
| Members (handle, rim) | ✗ Needed for grasp facts. Heatmaps reduce the need, because the contact region marks the part. |
| Realistic renders of separated objects | ~ pyrender now; Blender for training images (the thesis already has a Blender path). |
| Compute | ✓ CPU only; about 1 s per stack test, about 3 s per face test. |

## 6. Open questions for the supervisors

1. **Disturbance:** how large a disturbance must a support survive? This sets the 8° tilt and the placement offsets.
2. **Heatmap form:** is an amodal heatmap acceptable for hidden contacts, or do they want visible-only plus a per-face summary?
3. **Pair scope:** should the pairs cover all 5 categories crossed, or only the benchmark relations?
4. **Validation:** check the simulated support facts against UTAMP-2-GT on the 5 benchmark objects in their CoppeliaSim scene. This needs their `.ttt` file and hand-written fact table.

## 7. Next steps

1. **Friction:** a per-material friction table, and randomising mass, friction and size in the tests.
2. **Pair labels:** run support labels on all shakers and boards first, then mugs, pans and trays (needs cached decompositions).
3. **Training images:** render the separated objects in scenes with the amodal and visible heatmap labels.
4. **Gripper:** add the Franka hand and run grasp and lift sims (part 2).
5. **Model:** a pairwise head over DINOv2 patch features of both crops, which outputs a score and one heatmap per crop.
