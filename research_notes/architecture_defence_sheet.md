# Architecture defence sheet

Written 2026-11-16 from the code, not from memory. Code references: `M` = `p2p-heatmaps/p2pheat`, `S` = `affdet/sim` in this worktree.
Result numbers are from `p2p-heatmaps/reports/grasp_model_2026-11-04.md` and were not re-run today.

## 1. The thirty-second version

"The input is a point cloud of one object, in metres. Each point looks at its 16 nearest points three times in a row, so it learns the local shape around it. Then I take the maximum over all points to get one vector that describes the whole object, and give that vector back to every point. A small network then gives every point a number between 0 and 1: the chance that a top-down pinch at this point picks the object up. The labels come from a physics simulation that tries the pinch at every position on a grid."

## 2. The pipeline, step by step

| # | Step | Exact setting | Where |
|---|---|---|---|
| 1 | Object | mesh, convex parts for collision | `S/objects.py`, `S/registry.py` |
| 2 | Gripper | two box pads, 20 x 8 x 25 mm, opening 8 cm, each pad 5 kg and welded to a moving target | `S/grasp_map.py:27`, `:33` |
| 3 | Grasp test | pads come down from above, close until both feel 30 N or 1.2 s have passed, wait 0.1 s, lift 50 cm in 1.5 s, hold 0.4 s | `S/grasp_map.py:28-32`, `:171-172` |
| 4 | Friction | pad 1.0. MuJoCo takes the larger value of the two touching surfaces, so 1.0 is what acts | `S/grasp_map.py:37-41` |
| 5 | Grid | every position on a grid over the object, 6 gripper angles (yaws) each | `S/grasp_map.py:217-240` |
| 6 | Outcome per test | `held`: not on the table, rose more than 5 cm, slipped less than 5 cm. `swing_deg`: how far the object rotated, measured once at the end | `S/grasp_map.py:194-199` |
| 7 | Map to points | the grid value is read at each point's (x, y) by bilinear interpolation. Points not on the first surface seen from above get label 0 and are marked not valid | `M/grasp.py:40-44`, `:55-62`, `:124` |
| 8 | Four labels per point | `held` = share of the 6 yaws that hold. `held_any` = at least one yaw holds. `level30`, `level30_any` = the same, with rotation under 30 degrees as well | `M/grasp.py:34`, `:120-123` |
| 9 | Sampling | training: 2,048 random points (default). Test: every n-th point, up to 4,096 (default) | `M/grasp.py:179-182`, `:380` |
| 10 | Augmentation | random turn about the vertical axis. With `--partial 0.7`, in 70 % of samples only the points visible from a random camera (20 to 55 degrees up) are kept, and normals are estimated from them | `M/grasp.py:171-178`, `:184-187` |
| 11 | Model | section 3 | `M/model.py`, `M/grasp.py:195-208` |
| 12 | Loss | binary cross-entropy per point, summed over the 4 labels. Points that are not valid count with weight 0.1 | `M/grasp.py:272-274` |
| 13 | Optimiser | AdamW, learning rate 1e-3 with one-cycle schedule, weight decay 1e-4, batch 8, 120 epochs in the reported runs | `M/grasp.py:264-265`, `:378-379` |
| 14 | Split | 5 folds over objects. Each category is spread over the folds. A test object is never seen in training | `M/data.py:18-30` |
| 15 | Score | per object: AUC (does the model rank good points above bad ones) and top hit (is the single best point really good). Then the mean over objects | `M/metrics.py:15-32` |

No centring and no scaling: the coordinates go in as stored, in metres (`M/grasp.py:162-190`). So the model sees real size. It is not invariant to moving the object.

## 3. The model, layer by layer

Grasp model, one batch of 8 objects with 2,048 points. Parameter counts measured by running the code on 2026-11-16.

| Layer | What it does | Output shape | Parameters |
|---|---|---|---|
| Input | xyz in metres + normal (zeros in the camera variant) | (8, 2048, 6) | 0 |
| kNN | for each point, the 16 nearest points by xyz. Computed once | (8, 2048, 16) indices | 0 |
| EdgeConv 1 | for each neighbour j of point i: `[f_i, f_j - f_i]` -> Linear(12, 64) -> LayerNorm -> GELU -> Linear(64, 64). Then max over the 16 neighbours | (8, 2048, 64) | 5,120 |
| EdgeConv 2 | same, on the output of layer 1 | (8, 2048, 64) | 12,544 |
| EdgeConv 3 | same, wider | (8, 2048, 128) | 33,280 |
| Concat | outputs of the three layers side by side | (8, 2048, 256) | 0 |
| Global | max over the 2,048 points -> Linear(256, 256) -> copied to every point | (8, 2048, 256) | 65,792 |
| Fuse | concat local + global (512) -> Linear -> LayerNorm -> GELU -> Linear | (8, 2048, 128) | 82,432 |
| Head | Linear(128, 128) -> GELU -> Linear(128, 4) -> sigmoid | (8, 2048, 4) | 17,028 |
| **Total** | | | **216,196** |

Pair (support) model: one such encoder for the top object and one for the base (2 x 199,168), then 2 rounds of cross-attention in both directions (4 heads, width 128; 2 x 264,960), then two heads (16,770 + 16,641). Total 961,667. Outputs: where the top touches the base, where the top's centre can go, which part of the top carries it. `M/model.py:65-80`.

Things to say without hesitation:

- **There is no flatten.** The "one vector" step is a max over points (`M/model.py:49`). The output keeps one value per point.
- **There is no downsampling.** Every layer works on all points.
- **How far one point sees.** Three layers on the same 16-neighbour graph: 3 steps along the graph. Everything further away reaches it only through the global vector.
- **Why `[f_i, f_j - f_i]`.** `f_j - f_i` describes the local shape independent of where it is. `f_i` keeps where the point is. With only the difference the layer would not know position.
- **Why max.** The order of points and neighbours carries no meaning, and max does not depend on order.
- **Why the global vector.** Whether a point is a good pinch depends on the whole object (its width, where its mass is), not only on the patch around the point.
- **What cross-attention does (pair model).** Each point of one object looks at all points of the other and takes in a weighted mix of their features. That is how the base learns how big the top object is.

## 4. Where each part comes from

| Our part | Paper | What the paper says | Checked |
|---|---|---|---|
| Max over all points, copied back to every point | PointNet (Qi et al. 2017) | segmentation concatenates the 1,024-wide global feature onto each point (n x 1088) | yes, in the paper text |
| EdgeConv with `[x_i, x_j - x_i]`, max over neighbours | DGCNN (Wang et al. 2019) | this is the edge function the paper adopts (Eq. 7) | yes |
| Fixed neighbour graph on xyz | DGCNN, Table 4 | fixed graph 92.2 %, dynamic graph 92.9 % on ModelNet40 classification. So the dynamic graph is worth about 0.7 points, on a different task | yes |
| Per-point success map learned from simulation | Where2Act, O2O-Afford (Mo et al. 2021) | both use a PointNet++ backbone at width 128 and sampled trials in SAPIEN | first reader only |
| Two objects exchanging information | O2O-Afford | a bounding-box baseline is almost as good on placement (F 80.9 against 81.4) | first reader only |
| Per-point grasp success | Contact-GraspNet (Sundermeyer et al. 2021) | each observed point predicts a contact score and a 4-DoF grasp | first reader only |
| Physics labels for grasps | ACRONYM (Eppner et al. 2021) | FleX simulator, friction 1.0, 8 cm opening, test = still in the fingers after a shake with gravity off | first reader only |

"First reader only" means one agent read the paper and nobody checked the note. Do not quote exact numbers from those rows until checked.

## 5. How we compare with the field

| | Ours | Typical in the literature |
|---|---|---|
| Backbone | 3 EdgeConv layers, 0.2 M parameters, no hierarchy | PointNet++ (hierarchical). Newer: Point Transformer V3 (46.2 M parameters, built for scenes) |
| Labels | every position on a grid, simulated (dense) | random sampled trials (sparse), or analytic rules on the mesh |
| What "grasp" means | clamp + lift 50 cm + stay level | close and hold, sometimes a shake. No lift, no level condition in the label |
| Grasp output | where a top-down pinch works | a full 6-DoF gripper pose |
| Scene | one object on a table | cluttered scenes, real depth images |
| Real robot | none yet | Dex-Net 2.0, 6-DOF GraspNet, AnyGrasp all report real success rates |
| Metric | per-point AUC, top hit | F-score, average precision, real success rate. Not comparable with ours |

## 6. Weak points to name yourself

1. **k = 16 at two densities.** Training uses 2,048 points and testing 4,096, with the same k. So at test time each point's neighbourhood is physically smaller. DGCNN doubled k when it doubled the points. Measured 2026-11-16 on the real training items (data32r, partial views 0.7, 2,048 points; median distance to the 16th neighbour, 3 objects per category): shaker 4.8 mm, mug 7.8, bowl 11.6, wok 13.1, plate 13.3, pan 13.5, cutting board 19.9 mm. At the 4,096 test points: 4.1, 7.6, 9.1, 10.8, 10.2, 11.3, 17.6 mm (2-25 % smaller). So the main problem is the 4x spread across objects; the train/test shrink is smaller. Fix being tested: a fixed ball (`--radius 0.015` and `0.010`, at most 64 points: `--k-max 64`) in `p2pheat/grasp.py`, plus a PointNet row (`--k-max 1`: no neighbourhood at all) to show whether local geometry helps. Caveat: on shakers more than 64 points lie within 15 mm, so there the cap makes it kNN with k = 64.
2. **The label is "pickable", not "grasp".** Split into clamp, lift, level is now built in the simulator (2026-11-16) and tried on 5 objects (`research_notes/grasp_stages_2026-11-16.md`); the 185-object run is still to do. Two things found on the way: the old 0.1 s wait lifted from a 300 N squeeze (outcomes barely changed; new runs use 0.5 s), and the old level label counted a spin about the vertical as tilt (bowl level 23 % -> 43 %, mug 70 % -> 85 % with the tilt of the up axis from rest). The reported model was trained on the old level label.
3. **The simulator does not record whether the pads reached 30 N.** The lift also starts after a 1.2 s timeout (`S/grasp_map.py:171`).
4. **"Level" is measured once at the end and counts spin about the vertical too** (`S/grasp_map.py:194`).
5. **Thresholds are assumptions:** 30 N, 5 cm slip, 30 degrees, 8 mm pad clearance. Force and pad size were swept and matter little. The 30 degree limit matters a lot.
6. **The recipe (loss weight 0.1, 120 epochs) was chosen on the same folds it is reported on.**
7. **Camera results are optimistic.** The TripoSR route is given the true size and pose. No real photo yet.
8. **Flat objects.** 95 % of plate cells and 99 % of board cells are blocked for a top-down pinch before any physics runs. That is a limit of the gripper approach, not of the model.
9. **The analytic rule beats the model on "held"** (AUC 0.921 against 0.879). The model is better on "level" (0.848 against 0.804) and is the only one that works from a camera, because the rule needs the full mesh and mass.
10. **To confirm on the cluster:** the point counts of the reported runs. The defaults are 2,048 and 4,096, an older note says 8,192. Read `config.json` of the run before quoting.

## 7. Questions to practise

1. What goes into the network? What comes out?
2. Why metres and not normalised coordinates?
3. What is a neighbour here, and how many?
4. What does one EdgeConv layer compute? Why the difference `f_j - f_i`?
5. Why a max and not a mean?
6. Where is the whole object taken into account?
7. Is there a flatten or a pooling that loses the per-point output?
8. How many parameters? Is that small? (PointNet++ and Point Transformer V3 for scale.)
9. Why not PointNet++ or a transformer? (Honest answer: not tested; 185 objects; DGCNN shows the fixed graph is 0.7 points behind on classification.)
10. Where do the labels come from, exactly? What does one test do, second by second?
11. What is the difference between `held` and `held_any`?
12. Why binary cross-entropy when `held` is a share between 0 and 1?
13. What does "valid point" mean and why weight 0.1?
14. How do you know the model generalises to new objects?
15. What is the baseline, and when does it win?
16. What would change with a real camera?
17. How does your "grasp" differ from Dex-Net or ACRONYM?
18. For the pair model: what does cross-attention add, and did it help? (Only on one of three maps, +0.02 AUC.)
