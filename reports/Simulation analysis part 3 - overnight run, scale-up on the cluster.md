# Simulation analysis, part 3: scaling to the cluster, 22 objects, bugs the scale-up exposed

Parts 1 and 2: `Simulation analysis - static vs simulated labels and actionability heatmaps.md`, `Simulation analysis part 2 - pan grasp, mass model and pair support maps.md`. This part covers one overnight run (log with timestamps: `overnight_log.md`).

Scope is unchanged: the proof of concept covers isP2PSupportive and canGrasp, each label a heatmap. canSlide is a prototype. The simulator is the label source.

## 1. What now exists

| | count | notes |
|---|---|---|
| Objects | 22 | 4 shakers, 3 mugs, 3 frying pans, 3 trays, 3 cutting boards, 3 bowls, 1 plate, 2 woks |
| isP2PSupportive pair maps | 276 | 16 tops (shakers, mugs, pans, bowls, plate, woks) on 18 bases, 32 grid cells along the base's long side, full sweeps |
| Simulations behind them | 539,000 | 7.9 job-hours; about 1 hour of wall time on the cluster |
| canGrasp maps | 22 | one per object, grid scaled to the object's size |
| canSlide maps | 22 | 12 push directions at 4 heights each |
| Contact-region labels | 274 | the part of A that touches the part of B at the most robust placement |
| Label files | 552 + 22 + 22 | per-vertex arrays, 113 MB without contacts; format in `docs/sim_labels.md` |

The laptop needed more than an hour for 60 pairs at a coarse grid. The IFI cluster did 276 pairs at twice the resolution in about an hour, using one 16-core job per top object. One job ran at 4.4 s per 100 simulations against 19.7 s on the laptop, which was under memory pressure at the time. Code runs from a clone of your fork (`~/affdet-sim`), data under `/scratch/maksym.bek/sim/`, wrappers in `scripts/slurm_sim_*.sh`.

Results are saved outside git in `data/sim_results/2026-10-15/` (gitignored): raw maps, labels and the cluster logs' summaries.

## 2. Bugs the scale-up exposed

Every one of these changed numbers. None of them showed in the 46-shaker study or the first pair maps.

1. **Grasp pads crushed the object.** The closing logic only advanced the pads, so how hard they squeezed was set by how far they overshot. The hold force was 57 N with 0.2 kg pads and about 930 N with 5 kg pads, against the 30 N intended. Pads now back off and regulate to the target. With light pads and a regulated force, 0 of 10 pan-handle grasps held (the pad sank into the handle); with 5 kg pads, 9 of 10 hold. The bar calibration became a cliff: level up to 6 cm from the centre of mass, 21° swing at 8 cm, 52° at 9.5 cm. The earlier smooth ramp was an artefact.
2. **A single contact step decided "touching".** One step can report no contact while the top rests on the base. That left holes inside stable regions. Contacts are now counted over the last second.
3. **A solver crash killed a whole sweep.** MuJoCo ran out of its default working memory on a scene with 219 contacts. The arena is 256 MB now, and a failed solve counts as a failed placement.
4. **The overlap-at-release rule was a sensitive guess.** Placements that began inside the base were rejected above 3 mm of overlap, and moving that limit to 6 mm changed 144 of 711 face maps by more than 5 points. It is replaced by what a robot does: the top is held above the point and lowered straight down until it first touches anything (bisection on its height). Results moved for 22 of 72 pairs compared on identical grids: pans now nest into other pans (39% to 89% of positions) and wide objects now rest across a mug's rim (3 to 4% up to 26 to 44%), which the old probes missed. The old overlap limit no longer has any effect.
5. **Two stability definitions.** A single placement test used 0.3 rad/s and 1 cm limits, the stored maps 1.5 rad/s and 3 cm. For 14 of 274 best placements the two disagreed. One shared function now defines "stays" for both, with a test that every cell of a small map equals the single test there.
6. **Videos did not show what the labels computed.** The clip builder still used the old release while the labels used the new one. It now calls the same lowering routine and warns if its final pose differs from the label run's by more than 5 mm.
7. **The map on B painted where A's centre can be, not where A touches B.** A plate resting across a mug's rim lit up the mug's floor. Section 5 describes the extra support channel that fixes this.

## 3. isP2PSupportive: the 16 x 18 matrix

Share of B's grid that can carry the top with some face (mean over the pairs in the cell), and the mean best margin in mm. Full table with every top: `figures/overnight/matrix_v4.md`.

| top \ base | bowl | board | pan | mug | plate | tray | wok |
|---|---|---|---|---|---|---|---|
| bowl (round) | 3 % | 82 % | 17 % | 1 % | 91 % | 38 % | 3 % |
| pan | 30 % | 99 % | 55 % | 22 % | 100 % | 78 % | 20 % |
| mug | 14 % | 89 % | 30 % | 40 % | 93 % | 66 % | 9 % |
| round shaker | 12 % | 91 % | 29 % | 11 % | 12 % | 70 % | 2 % |
| square shaker | 30 % | 100 % | 47 % | 66 % | 94 % | 85 % | 29 % |
| wok (handled) | 5 % | 97 % | 38 % | 40 % | 100 % | 65 % | 10 % |

Sanity checks that hold: nothing stands on a bowl or wok base (they are round), almost nothing round sits on a mug, everything fits on a flat plate or board, and a square shaker (six resting faces) is supported far more widely than a round one (one resting face). The margins are large on boards and trays (10 to 15 cm) and small on mugs (1 to 4 cm).

Figures: `figures/overnight/video_*.mp4` (six pairs, four placements each, 40% speed), `gallery_hex_shaker_faces_by_base.png`, and for the same mug painted for eight tops `figures/overnight/gallery_mug_*.png`.

## 4. How much do the choices matter

- **Tilt limit (10°).** Moving it to 5° lowers the mean stable share from 35.4% to 29.7% and changes 178 of 711 face maps by more than 5 points. 15° raises it to 38.2% (123 maps). This is the one threshold the labels are sensitive to. The export therefore carries the margin at 5° and 15° as extra channels.
- **Slide limit (3 cm).** 1 cm lowers the share by 3.6 points (68 maps move); 5 cm changes nothing.
- **Friction.** 23 pairs at friction 0.4, 0.6 and 0.8: none moves by more than 10 points (most by 0 to 2, worst 7). Support labels barely depend on friction, as expected for horizontal surfaces. Grasp labels were not tested for it.
- **Coarse-to-fine sweeps** (`placement_map_adaptive`): saves 39 to 60% of the simulations at factor 2 with 96.6 to 98.4% of cells unchanged against a fresh full sweep, and the same best margin in every case. They miss narrow stable regions: on a pan-on-tray map the stable set came out as a thin plus-shape with speckled borders. They are not used for the final labels.
- **Reproducibility.** @@REPRO@@

## 4b. Audit: are the verdicts physically right?

36 placements were drawn at random across the 276 maps (half the map calls stable, half failing although the top touched the base) and re-run
(`scripts/sim_label_audit.py`, figures `figures/overnight/audit_*.png`, table `audit.json`). A fresh run reproduced the stored verdict in all 36.
Looking at every final frame: 34 are physically what one would expect. Pans nest in pans, a pan balances across a mug's rim, a ball bowl
domes over a mug, mugs and shakers lying on trays or in bowls roll and are marked failing, and anything leaning on a rim or half over an edge is
marked failing. Two tiles (the same pair drawn twice, a round shaker standing with its base across the mug's 5 mm rim ring, margin 5 mm) are
fragile rather than wrong: the simulator keeps them for 2 s, but they are one grid cell from failing and a real placement there would not hold.
Their score is low (5 / 20 = 0.25), which is what the score is for. One further tile (plate on a tray) could not be judged from the camera angle.
The audit judges the pose in which the top was released: a mug lying in a bowl is stable in its new pose but fails as "stands upright where put down".

## 5. Which part of B carries A

The map on B (`margin_mm`) answers "where can A's centre be". For a label of which part of B does the carrying that is the wrong question: the plate that rests across a mug's rim lights up the mug's floor. The export therefore has a second pair of channels, `support_mm` and `support_score`: for every vertex, the best margin among all stable placements whose contact points lie within 1 cm of it. Contact points are MuJoCo's own, collected over the last second and stored per sweep cell (object frames).

A second label holds the contact region of the single most robust placement per pair (`contact`, `contact_gap_mm`): a pan balanced on a mug touches the mug along most of its rim and the pan along a ring on its floor; a bowl on a mug touches the mug's rim with its foot ring. Figures: `figures/overnight/contact_*.png`.

## 6. canGrasp on all objects

Held means lifted 50 cm and not slipping in the pads. Level means it also hangs within 30° of level. Closing is regulated to 30 N.

| object | cells | held | level | pad fits (model) | model vs simulation |
|---|---|---|---|---|---|
| square shaker | 100 | 81 | 81 | 81 | agrees (81 of 81) |
| round shaker | 144 | 92 | 92 | 90 | agrees |
| mug 0 / 1 / 2 | 192 / 208 / 221 | 43 / 52 / 82 | all of them | 36 / 44 / 67 | no over-promise; simulator also holds 7 to 15 cells the model misses |
| pan 1.2 kg | 176 | 27 | 4 | 20 | no over-promise; level 4 cells the model rejects |
| pan 0.8 kg | 170 | 26 | 1 | 19 | one over-promise |
| bowl 0 / 1 | 289 / 289 | 138 / 124 | 49 / 60 | 107 / 115 | model over-promises level in 63 / 44 cells (curved shell) |
| tray 0 / 1 / 2 | 204 / 224 / 176 | 38 / 42 / 20 | 0 | 44 / 42 / 25 | model promises 3 to 11 cells the simulator does not lift |
| cutting boards | 176 | 0 to 1 | 0 | 4 to 6 | the model finds fits the pads cannot reach (97% of cells are blocked) |
| plate | 256 | 0 | 0 | 0 | agrees |

A flat board or plate lying on a table cannot be picked up by a top-down parallel gripper, which the simulator reproduces. Pans and woks are lifted at the rim and handle but hang at an angle (level in 0 to 2%): the weight pitches them about the pinch. The analytic model (roll and pitch torque limits) is a conservative filter for most objects and wrong for curved shells (bowls) and flat objects (boards, trays).

Figure: `figures/overnight/gallery_grasp.png`.

## 7. canSlide (prototype)

A flat pusher pushes an object that has settled on the table, horizontally toward its centre, at 12 azimuths and 4 heights, 12 cm at 5 cm/s. It slides if it follows the pusher (80% or more of the travel) without tipping more than 10° or turning more than 25°. Tall narrow objects slide when pushed low and tip when pushed high (round shaker: 100% at 15% of its height, 0% from 40%); boards, trays and the plate slide from every side and height; round-bottomed bowls and one wok roll or tip. A square shaker slides only when pushed along a face normal (4 of 12 azimuths), otherwise it spins. Video: `figures/overnight/slide_tall_shaker.mp4`.

## 8. Limits and open items

- **No independent reference.** There are no pair-level labels for these objects, so what validates the maps is the physics checks above, the agreement with the shaker face study (24 of 24 faces), and visual review of the videos.
- **Coarse grids on large objects.** 32 cells along the long side gives 1 to 2 cm cells on boards and trays; margins are quantised to that.
- **Faces that do not hold on a flat table are not swept**, so a round cap resting inside an opening is missed.
- **Mass** is spread by surface area, not by material; friction is 0.6 everywhere.
- **Bowls** are the weakest case for the analytic grasp model, and round-bottomed bowls show up as "slides" or "tips" depending on how they settle.
- **The label scale** (score = margin / 20 mm, capped) is a choice; `margin_mm` is stored.
- **canLift** is the lift phase of the grasp test (`held`); **canSlide** needs a second push direction convention (inward push only) before it can be called a label.

## 9. Suggested next steps

1. Look at the 276 maps at full resolution for outliers, starting from the pairs whose most robust placement did not stay stable on re-run (14 of 274).
2. Decide the score definition and the tilt limit with your supervisor; both are cheap to change from the stored raw metrics.
3. Add objects from other classes (bottles, plates, cups, boxes) to the matrix; the pipeline runs without changes (specs in `affdet/sim/registry.py`).
4. Train a first model on `labels/` to see whether the heatmaps are learnable from images.
