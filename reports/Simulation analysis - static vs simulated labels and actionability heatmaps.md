# Simulation analysis: static vs simulated labels, and actionability heatmaps

> **Updated 2026-10-15.** The shaker face study (section 2) stands. The placement and grasp maps in sections 3 and later were made with earlier versions of the simulator; the corrected, full-resolution results are in `Simulation analysis part 3 - overnight run, scale-up on the cluster.md` (grasp pads, release rule, stability definition, 276 pair maps).

Follow-up to `Inter-object affordances - heatmaps and simulation.md`. Everything here was re-run from scratch on
the branch `sim-heatmap-analysis`; numbers come from the scripts listed in §7. Nothing was checked against a real
robot or the UTAMP-2 scene.

## 1. Short answers

| Question | Answer |
|---|---|
| Did the "4 outliers of 48" reproduce? | No. The report never says which 12 shakers it used. On all 46 shakers × 6 faces the original prototype disagreed with the static labeller on 25 of 276 faces. |
| Why did static and sim disagree? | Partly because they asked different questions (sim tilted 8°, static requires 5° and a 5 mm margin). After aligning them, 35 of 276 faces still disagree (§2). The leftovers have identifiable causes, mostly a *defect in the static labeller* (it only sees flat facets), not in the simulator. |
| Was the simulator itself right? | No, it had three real bugs, all fixed (§3): bodies made of several convex parts sank 9 mm into the table, the stack test measured creep of the whole stack, and objects were released from the rim height instead of lowered onto the surface. |
| Can we make *action* heatmaps (not touch maps)? | Yes, for two actions. **Placement:** sweep the release point over A, simulate each, keep where B stays (§4.1). **Grasp:** sweep a parallel-jaw pinch over the object, lift, keep where it stays level (§4.2). Both give a map over A's/the object's surface and can be rendered to image pixels (§4.3). |
| Is the COM effect real in the sim? | Yes. On a uniform 20 cm bar the swing grows steadily with the pinch distance from the COM: 0, 8, 16, 19, 24, 30, 38° at 0, 2, 4, 5, 6, 8, 9.5 cm. It never drops the bar (the pinch holds, the bar pivots). An analytic torque model puts its "level" limit at 5.1 cm, which in the sim is about 19° of swing; the model agrees with the simulated mug maps without ever over-promising (§4.2). |
| Does it work for frying pans? | **Only analytically.** The simulated gripper cannot lift a 1.2 kg pan (0 and 2 % of grasp points succeed, §4.2). The analytic model gives the expected answer: pinch the rim radially, the closing line through the COM; the handle fits but the pan swings. |

## 2. Shaker study: static vs simulated labels

### 2.1 What was made consistent

| Inconsistency | Before | After |
|---|---|---|
| Disturbance | sim tilted 8°, static needs a 5° tip angle | one constant, `DISTURBANCE_DEG = Params.min_tip_deg = 5°` (a test pins this) |
| "Near the face" | sim dropped the object exactly on the box face; static accepts a rest on any facet within 20° | sim tries the exact face, then the largest hull facets within 20° (`face_holds`, up to 4 candidates) |
| Margin | static requires the COM 5 mm inside the contact patch; sim had no such check | sim reports the COM margin of the *settled pose* (vertices within 0.5 mm of the table); reported next to the tilt test |
| Sway | 2 mm centre-of-mass motion in the last second failed a tall mill that only sways | 5 mm |
| Settle angle | 3° (a 64-gon cylinder, whose facets are 5.6° wide, sits 2.8° off, so it counted as flat) | 2° |
| Calibration | 5 primitive cases in the test file (the report cites 28) | 19 tests: box on all 6 faces, cylinder, cone, faceted cylinder; plus a bar-grasp and an analytic-model test |

### 2.2 Result on all 46 shakers × 6 faces (276 faces)

Static labels recomputed with `min_tip_deg = T` so both tests ask "does the face survive a T° disturbance?".
The sim verdict is "passes the tilt test" (column *tilt only*) or "passes the tilt test and the settled COM margin is
at least 5 mm" (column *+ 5 mm margin*, the physical counterpart of the static rule). Agreement = faces where static
and sim give the same yes/no.

| T | tilt only: agree | static-only | sim-only | + 5 mm margin: agree | static-only | sim-only |
|---|---|---|---|---|---|---|
| 3° | 209 | 6 | 61 | 244 | 12 | 20 |
| 5° | 228 | 9 | 39 | **241** | 15 | 20 |
| 8° | 233 | 11 | 32 | 238 | 17 | 21 |
| 12° | 237 | 13 | 26 | 236 | 19 | 21 |

At the aligned setting (T = 5°, + 5 mm margin) the two agree on 241 of 276 faces (87 %). Per-face data:
`docs/objaverse/salt_shaker_sim_vs_static.csv`.

### 2.3 The 35 remaining disagreements

| Faces | Group | What the pictures show |
|---|---|---|
| 20 | **Sim holds, static says no.** Static margin below 5 mm (often negative), yet the settled sim pose has a COM margin of 7–15 mm. | The object rests on a *wider patch than the flat facets*: a cap bulge plus a body corner (3-point support), as in `settled_poses.png` rows 1–2. The static labeller builds its support polygon only from flat triangles, so it misses these contacts. This looks like a defect of the **static** labeller. |
| 9 | **Static holds, sim fails the tilt test.** | Rounded or domed contact: after the tilt the object settles on another facet (7 faces, e.g. `95ac8f05` cap-down, `settled_poses.png` row 3: a dome touching at a spot) or topples (2 faces: `cf8640d4`, `e83afd6d`, cap-down; `cf8640d4` rests on a 104 mm² patch). The sim looks right for the dome. |
| 6 | **Static holds, sim passes the tilt but the settled margin is below 5 mm.** | Near-round, review-flagged bodies (`9ef6c018`, `38f7cb1a`, `e07a892b`). Marginal either way. |

The four faces discussed earlier, at the aligned setting:

| Face | Static | Sim (T = 5°) | Reading |
|---|---|---|---|
| `88067fa6` under (tall mill) | holds | **holds** (16.7 mm margin) | was a threshold artefact (2 mm sway limit), gone |
| `12a6366c` on (cap) | holds (tip 7.3°) | **holds** | tipped only because the old 8° tilt exceeded its 7.3° tip angle, gone |
| `95ac8f05` on (cylinder, cap down) | holds | fails | rounded cap; the sim looks right |
| `5ba5959e` left (ridged) | holds, review-flagged | fails (2 mm margin) | marginal, near-round |

### 2.4 Recommendations

1. **Adopt the settled-pose margin in the static labeller.** Replace "support polygon from flat facets" by "vertices
   within about 0.5 mm of the resting plane" (what `settled_margin` does). That removes the 20-face group without
   running dynamics.
2. **Treat the disturbance as a label parameter**, not one yes/no: the sweep gives a robustness curve per face
   (5 / 8 / 12° columns above). The supervisor question "how big a nudge" then becomes "which column to train on".
3. Keep the dynamic test for what statics cannot see: domed contacts and toppling from a small patch (the 9-face group).

## 3. Simulator bugs found and fixed

| Bug | How it showed | Fix |
|---|---|---|
| **Multi-part bodies sink.** A CoACD body (board, mug, pan) sank 8.8 mm into the table at the default contact softness (single convex hulls: 0.1 mm). | A shaker on a board "dropped" 13 mm; `sink mm` test over solref/timestep. | `timestep 0.001`, `solref 0.006` (0.7 mm). Primitive tests still pass. The earlier cloud numbers used the soft default. |
| **Stack metrics included creep.** Absolute drift and tilt of the top object picked up the whole stack creeping about 5 mm in 2 s. | Shaker on a flat board "drifted" 1–5 cm. | Drift and tilt are measured relative to the base. |
| **Dropped from the rim height.** The top object fell into any concave base (cup, tray) and was marked as failing the drop test. | Tray interior all "unstable". | Released 2 mm above the surface under its footprint (`release="surface"`), like a robot lowering it until contact; the map also records how deep the support surface is (`on` vs `in`). |
| **Mocap pads cannot lift.** Teleporting mocap fingers have zero velocity for contacts, so they squeeze but never drag an object with friction. | Even a plain 3 cm box could not be lifted. | Pads are dynamic bodies welded to mocap targets; each closes until its own contact force reaches 30 N. |

## 4. Actionability heatmaps: what we can produce and how

### 4.1 Placement ("where can B be put down on A?")

`affdet/sim/placement_map.py`. For every offset on a grid over A's footprint, B is lowered onto A there and the
stack simulated for 2 s. B counts as staying if it keeps touching A, tips less than 10°, slides less than 3 cm
relative to A, and does not end lower than the surface under it requires. Output per cell: stays on top, stays
inside (cup/tray), or fails; plus a robustness value, the distance in mm to the nearest failing placement. The
argmax of that is the suggested placement point. Cost: about one second per placement, 800–1000 placements per map,
1–3 minutes per map on 8 cores.

| Pair | Stable area | UTAMP-2 default (bbox centre) | Most robust point | Reading |
|---|---|---|---|---|
| shaker on board 41×28 cm | 86 % | stable, 110 mm from failing | centre, 110 mm | the board is forgiving everywhere except near its edge and the slot |
| shaker on mug 14×10×12 cm | 16 % (44 rim cells, 51 inside) | stable but **5 mm** from failing | 2 cm off-centre toward the cup centre, 14 mm | the default works only because the bbox centre happens to be near the cup floor; the map shows how little margin it has. Rim cells are the shaker bridging the thin rim |
| shaker on tray 55×39×3 cm | 70 % (inside) | stable, 165 mm | 165 mm | placement is on the tray floor, not on the rim |

Figures: `reports/figures/sim_analysis/shaker_on_{board,mug,tray}.png` (left: kind map, right: robustness).

This is the closest to what O2O-Afford does ("where on B can A be placed"), with labels from the same kind of
simulated interaction, but the input is a mesh/image, not a point cloud, and the output is meant to feed
`isP2PSupportive` by summing heat per box face. It is not a learned model yet: these maps are the *labels*.

### 4.2 Grasp ("where can the object be picked up and held?")

`affdet/sim/grasp_map.py` (simulation) and `affdet/sim/grasp_analytic.py` (analytic). A parallel-jaw gripper
(two 2×2.5 cm pads, 8 cm opening, 30 N each, friction 1.0, torsional friction 0.01 m) closes at (x, y) with a given
yaw, lifts the object 50 cm and holds. It succeeds when the object is clear of the table and has not slipped in the
grip. `swing_deg` is how far the object rotated, which is the COM effect: a pan pinched far from its COM pivots about
the pinch until the COM hangs below it.

**Calibration on a uniform bar (20×3×2 cm, 1.2 kg), pinch across the width:**

| Pinch distance from COM | 0 | 2 cm | 4 cm | 5 cm | 6 cm | 8 cm | 9.5 cm |
|---|---|---|---|---|---|---|---|
| Swing (sim) | 0° | 8° | 16° | 19° | 24° | 30° | 38° |
| Held clear of the table, no slip | yes | yes | yes | yes | yes | yes | yes |
| Analytic: weight torque / pad torque capacity | 0 | 0.4 | 0.8 | 1.0 | 1.2 | 1.6 | 1.9 |

The pads never lose the bar here; it pivots about the pinch by an amount that grows with the distance (the pad
torsional friction is finite, not a hard limit). So the "level" test below is a threshold on that growth, not a
physical cliff.

**Analytic model:** a pinch holds the object level iff it fits (the chord along the closing direction is below the
opening) and `m g d <= tau_max = 2 mu_t F` (0.6 Nm), where d is the horizontal distance of the COM from the closing
line. Limit for the 1.2 kg bar: 5.1 cm, where the simulated swing is about 19°. The simulator's own success threshold (swing below 30°) is reached at 8 cm, so the analytic "level" set is a subset of the simulated successes; that is why the mug comparison below has no analytic-only cells.

**Analytic vs simulation on the mug** (165 grasp points × 6 yaws, any-yaw): both yes 50, both no 72, sim only 43,
analytic only 0. So the analytic map is conservative (it never promises a grasp the sim cannot do); the 43
sim-only cells are edge pinches where the pads close on part of the rim rather than around the full width.
(`grasp_analytic_vs_sim_mug.json`)

**Frying pans (analytic maps, `grasp_analytic_frying_pan_*.png`):** for the 1.2 kg pan the whole **rim wall** is
the best pinch (191 of 1204 grid cells hold it level; the best yaw is the one whose closing line passes through the
COM, so d is about 0). The **handle** fits but is amber: the COM is 15–25 cm along the handle, so the pan swings
(38 cells). The bowl floor is dark: nothing to pinch. The COM itself is over the floor, so "grasp where the COM is"
is not available on a pan; "close the pads along the line through the COM" is.

**Limit: the simulated gripper cannot lift the pans** (`grasp_ba9407ad.png`; 0 % and 2 % of grasp points succeed).
Handle grasps get the pan up for a moment (it hangs vertically in the video frames) and it slips out of the 2 cm
flat pads; slower lifts never get it off the table. I traced the force (contacts hold 30–60 N) but not the final
cause. Likely: flat pads on a rounded handle, and a hull approximation of the handle. So pan labels from the
simulator are not usable yet; the analytic model is, with the caveat that it is validated on a bar and a mug, not
on a pan. A better gripper model (the Franka hand from MuJoCo Menagerie) is the next step.

### 4.3 From a map to pixels

`scripts/sim_map_render.py` copies map values onto the mesh and renders it from a robot-like camera
(`pixel_heatmaps.png`: pan grasp classes, shaker-on-mug robustness, shaker-on-board robustness). Only
upward-facing surfaces get heat; for hidden contact surfaces (the underside of a shaker) the amodal rendering
from `heatmap_render.py` applies unchanged. Per-pixel labels for training therefore fall out of any map of this kind.

## 5. How this differs from the touching heatmap, and what to build

| | Touching heatmap (first prototype) | Actionability heatmap (this work) |
|---|---|---|
| Label means | surfaces that touch once a *fixed centred placement* has settled | "if I act *here*, does it work?" |
| Varies over | nothing (one trial) | the action parameter: release point, pinch point and yaw |
| Answers the planner | where the contact is | where to place, where to grasp, how robust |
| Cost | 1 simulation | ~1000 simulations per object pair (minutes) |

Plan to produce training labels: (1) placement map per pair, sum per box face for `isP2PSupportive`, use the
robustness argmax as the placement point; (2) grasp map per object from the analytic model (fast), with the
simulator as the check on objects where the gripper works; (3) render both to image space with visible and amodal
variants. The model then learns the maps; the simulator is only the labeller.

## 6. Open issues

* **Pan grasp in simulation** (§4.2); needs a real Franka hand model and a better handle collision shape.
* **Mass distribution.** Mass is uniform over the collision hulls. For a pan this puts a solid handle's volume on
  par with the thin body; the real COM sits lower in the bowl. The COM drives the grasp maps, so a per-material or
  per-part density is needed before pan labels mean anything.
* **Friction** is one value per object (0.6 / pad 1.0); a per-material table is still missing.
* **Static labeller** has the facet-only polygon defect of §2.3 (the fix is described, not applied).
* **Creep.** Soft contacts let a stack creep ~5 mm per 2 s; the 3 cm slide limit is generous for that reason and
  could hide very slow slipping.
* **Only one shaker, three base objects and two pans** were mapped; the pipeline is per-object and parallel, but no
  claim is made about category-level behaviour.
* Where the literature comparison names O2O-Afford and Where2Act, it relies on the existing report's summary
  and sources and was not re-read here.

## 7. Reproduce

```
python scripts/label_faces.py docs/objaverse/salt_shaker_set.csv GLBS OUT --no-render
python scripts/sim_shaker_sweep.py docs/objaverse/salt_shaker_set.csv GLBS sweep.json      # 4 tilts, ~10 min
python scripts/sim_shaker_report.py docs/objaverse/salt_shaker_set.csv GLBS sweep.json cmp.json
python scripts/sim_settled_pose_figure.py docs/objaverse/salt_shaker_set.csv GLBS fig.png 1f4e8a33:front ...
python scripts/sim_placement_maps.py picks.json GLBS docs/objaverse/salt_shaker_set.csv OUT
python scripts/sim_grasp_maps.py picks.json GLBS OUT --role frying_pan --index 0 1 --step 0.025
python scripts/sim_grasp_analytic_report.py picks.json GLBS WORK OUT
python scripts/sim_map_figures.py picks.json GLBS PLACEMENT GRASP OUT
python scripts/sim_map_render.py picks.json GLBS PLACEMENT GRASP OUT
```

`picks.json` lists the first three kept objects per role from `docs/objaverse/other_roles_set.csv` (mug 77d41aba,
board 5b21f622, tray 3f3aeaa2, pans ba9407ad and ad932414). GLBs are downloaded from Hugging Face
(`allenai/objaverse`). Tests: `affdet/tests/test_sim_support.py` (19 cases, about 20 s).
