# Simulation analysis, part 2: pan grasp, mass model and pair support maps

Scope: the proof of concept covers two predicates, **isP2PSupportive** (A placed on B) and **canGrasp**. Each label is a heatmap. canLift is the lift phase of the grasp test, and canSlide comes later. The simulator is the label source. Part 1 is `Simulation analysis - static vs simulated labels and actionability heatmaps.md`.

## 1. Pan grasp: found and fixed

**Cause.** The gripper pads were spawned at table height. On a frying pan that is inside the 13 mm high floor hull, so one pad read about 500 N of contact before the grasp started and the weld to its target fought the geometry. Part 1 therefore reported "the gripper cannot lift the pans (0 to 2 %)", and its 56 % mug figure was inflated by the same artefact.

**Fix** (`affdet/sim/grasp_map.py`).
- Pads now start at a height computed from rays cast under both pad footprints.
- If a pad cannot reach the pinched surface, the cell is marked `blocked` and counts as a failure. That happens, for example, next to a taller wall.

**Result after the fix.**

| object | cells | lifted and held | held with swing < 30° |
|---|---|---|---|
| pan 1.2 kg | 308 | 43 (14 %) | 0 |
| pan 0.8 kg | 273 | 33 (12 %) | 0 |
| mug 0.3 kg | 165 | 35 (21 %) | 35 |

The pans are lifted at rim and handle cells, but they always hang 30° to 90° from level.

**Why they swing.** A pinch has two torque limits, and part 1's model had only the first.
- *Roll* about the closing line: m g d against 2 μ_t F. This one was already in the model.
- *Pitch* about the horizontal axis perpendicular to the closing line, when the centre of mass (COM) is offset along the closing direction by e. A radial rim pinch of a pan has e equal to the whole radius. The pads can only resist it with a couple of their normal forces: m g e against F · L.
- L was calibrated on 272 simulated grasps (two pans and a mug). When m g e was above F × 1.25 cm, the median swing was 42° to 60° in every ratio bin. When it was below F × 0.6 cm, the median swing was 10°. L is set to 1.25 cm, half the pad height; only two cells fell between the two thresholds, so the exact value is not pinned down.
- Part 1's claim that "the rim holds the pan level" was wrong. The rim can be lifted but not held level.

**Analytic model vs simulation** (`scripts/sim_grasp_compare.py`, zero over-promises everywhere):

| object | stable: both / sim only / analytic only | held vs "pad fits": both / sim only / analytic only |
|---|---|---|
| pan 1.2 kg | 0 / 0 / 0 | 36 / 7 / 11 |
| pan 0.8 kg | 0 / 0 / 0 | 25 / 8 / 11 |
| mug | 29 / 6 / 0 | 29 / 6 / 0 |

The 22 analytic-only pan cells are all on the handle at yaws that are not perpendicular to it: the pads slip off. I tested contact-normal angle as a cause and it did not predict which cells hold (62 to 96 % held in every angle bin), so I removed it. This is still open.

Figures: `figures/sim_analysis_2/grasp_pan_1p2kg.png`, `grasp_pan_0p8kg.png` and `grasp_mug.png`. Black dots are cells the simulated gripper held, and ringed dots held with swing under 30°.

## 2. Mass model

`load_body(mass_model="auto")` splits the mass over the convex parts by surface area (thin shells) or by volume (solid objects), instead of uniformly over hull volume.

- COM shifts against the old model are small: 5 mm for the 1.2 kg pan, up to 15 mm in x for the 0.8 kg pan and 5 mm in z for the mug. Part 1 overstated "pan COM unreliable" as an error source.
- The real unknown is per-material density, for example a plastic handle on a steel body. The GLB material slots are unnamed and objaverse metadata lists materials per object, so this stays open.

## 3. Pair maps for isP2PSupportive (`affdet/sim/pair_map.py`)

A pair label has a heatmap on each object. The sweep puts A on B with each face of A down and records where it stays.

- **On B:** at each grid point, the best robustness (mm to the nearest failing placement) over A's faces.
- **On A:** the best robustness of each face, painted on that face's resting patch.
- **Prefilter:** a face is only swept if it holds on a flat table by the shaker study's verdict (rest, survive a 5° tilt, COM 5 mm inside the patch). This is the same function as in part 1.

**Check against the shaker study.** Four shakers (round, tall, square, hexagonal) on 12 bases give 66 pairs in total. The faces that work in the pair maps match the study on all 24 of 24 shaker faces. Two defects were found and fixed first: the pair prefilter let a rolling round shaker through (front face), and it rejected the hexagonal shaker's four side faces because it settles on a neighbouring facet. Before the fix the agreement was 19 of 24.

**Third defect: overlap at release.** A video of the hex shaker on the mug showed a lying shaker (9 cm long) released across a mug with an inner diameter under 9 cm, so it started overlapping the wall by up to 33 mm. The solver pushed it out in arbitrary directions and the map still counted it as stable. `stack_test` now rejects any placement where B overlaps A by more than 3 mm at release (`release_penetration_m`). Stored maps were re-checked without re-running the physics (`scripts/sim_recheck_release.py`, zero-step check of the stable cells only): 29 of the 66 pair maps changed, from 1 % fewer stable cells for most shakers on trays and pans to 50 to 70 % fewer for shakers on mugs and 30 to 74 % fewer for pans on trays. The three part-1 shaker placement maps are unchanged apart from one mug cell (95 to 94). The 24 face agreements are unchanged.

For the hex shaker lying on the mug, 12 front, 8 back, 17 left and 9 right cells survive. The best front-face placement rests across the rim: its contact points are all at rim height (12.1 cm) and it ends 3° tilted, with a 9 mm margin.

**Cost.** The first pass took 4,662 s for 66 pairs on 8 CPU workers, 71 s per pair on average and 253 s at most. The grid had 16 cells along the base's long side. Re-running with the corrected faces reused every sweep that still applied.

| top | base | pairs | B points that can carry A | best margin (mm) | faces that work |
|---|---|---|---|---|---|
| tall shaker | board | 3 | 85 % | 145 | under, on |
| tall shaker | pan | 3 | 27 % | 152 | under, on |
| tall shaker | mug | 3 | 20 % | 16 | under, on |
| tall shaker | tray | 3 | 56 % | 170 | under, on |
| round shaker | board | 3 | 82 % | 145 | under |
| round shaker | pan | 3 | 25 % | 144 | under |
| round shaker | mug | 3 | 14 % | 18 | under |
| round shaker | tray | 3 | 56 % | 183 | under |
| square shaker | board | 3 | 99 % | 145 | all six |
| square shaker | pan | 3 | 44 % | 144 | all six |
| square shaker | mug | 3 | 55 % | 18 | all six |
| square shaker | tray | 3 | 74 % | 170 | all six |
| hex shaker | board | 3 | 98 % | 145 | under and four sides |
| hex shaker | pan | 3 | 43 % | 144 | under and four sides |
| hex shaker | mug | 3 | 37 % | 16 | under and four sides |
| hex shaker | tray | 3 | 74 % | 170 | under and four sides |
| mug | board | 9 | 90 % | 174 | under, on, front, back |
| pan | tray | 9 | 43 % | 136 | under, on |

Per-pair records: `docs/objaverse/pair_support_maps.jsonl`. Figures: `figures/sim_analysis_2/pair_*.png`. In each, the left panel is the map on B, the middle panel is the per-face robustness on A, and the two right panels are A rendered with those values painted on.

## 4. Limits

- **Coarse grids.** 16 cells along the long side means 2 to 5 cm steps. The best-margin numbers are quantised to the step, so treat them as ranks, not millimetres.
- **No reference labels.** No pair-level labels exist for these objects, so the only validation of the A-side maps is the 24 face agreements with the shaker study. The B-side maps are validated only by the part 1 placement checks.
- **Concave supports.** The prefilter assumes a face that does not hold on a flat table cannot hold on B. A round cap resting inside a mug opening could break that.
- **Boxy shakers.** The A-side map is uninformative when every face works: the whole shaker lights up.
- **Material density.** Open, see section 2.
- **canLift and canSlide.** Not started.
