# Simulation speed pilot (2026-10-15)

Question: can the pair maps be made for much less compute at the same quality? Reference = the full run that made the stored labels
(2 s per placement, every cell of a 32-cell sweep). Data in `docs/objaverse/`: `speed_pilot_stack_test.json`, `speed_pilot_reuse.json`
(cluster, 8 pairs, 120 random sweep cells each = 960 cells), `grid_pilot_v5.json` and `symmetry_pilot_v5.json` (offline, all 539,201 stored cells).
Scripts: `scripts/sim_speed_pilot.py`, `sim_grid_pilot.py`, `sim_symmetry_pilot.py`, `sim_profile_one.py`.

Where the time goes (one placement, 0.4-0.6 s on a cluster core): stepping the physics 79-88 %, building and compiling the model about 15 %.

| Shortcut | Verdicts equal to the full run | CPU time vs full | Verdict |
|---|---|---|---|
| Stop once B has clearly left A (drift > 6 cm or tilt > 30 deg) | 960 / 960 | 0.75 | use; saves most where most cells fail (shaker/bowl on mug 0.35-0.41), nothing on pan-on-tray |
| Stop at drift 4.5 cm / tilt 20 deg | 960 / 960 | 0.71 | same, slightly more |
| Skip placements where B is not over A (it lands on the table) | 953 / 960 | 0.96 | do not use: 7 false fails (a wok rests on the table touching the pan) |
| Run 1.5 s / 1.25 s instead of 2 s | 953 / 952 | 0.64 / 0.61 | not worth the risk |
| Reuse the compiled model across a sweep | 959 / 960 | 0.885 | use; one verdict flips in a tumbling cell (raw tilt differs there, the position rounding in the XML differs by 5 um) |
| Abort + reuse | 959 / 960 | 0.64 | |
| Simulate every 2nd cell + refine where corners disagree or next to a stable cell | 99.3 % of cells | 0.52 of the cells | loses 3,634 of 189,200 stable cells (1.9 %), best margin exact |
| Same without the "next to a stable cell" refinement | 99.2 % | 0.37 of the cells | best margin 95th percentile off by 4.9 mm |
| Every 3rd cell + refine next to stable cells | 99.0 % | 0.43 of the cells | loses 2.7 % of stable cells |
| Symmetry (simulate half, mirror, check on 5 % of the other half) | | | not worth it: only 153-258 of 698 face maps pass the check, with 3-4 % error, saving about 48 % on those |

Symmetry: mean agreement between a map and its mirror image is 78-89 %. Round bases do not help much because the tops are not
symmetric (handles, polygons, faces on their side); only round tops on round bases (bowl on plate) are, and those maps are almost all
or none stable already.

Combined estimate (not run end to end): refinement 0.52 x abort+reuse 0.64 = about 0.33 of the full cost, with the cell loss above.
Lossless only: abort + reuse = 0.64.

## End to end on the cluster (12 pairs, full 32-cell sweeps)

Run: `--adaptive 2 --grow 1 --abort --reuse-model` against the same code without options, on the same node (job 1729217, 16 cores).
Data: `docs/objaverse/speed_pilot_end_to_end.json`, script `scripts/sim_fast_vs_full.py`.
Pairs: shakers (3) on mug / tray / board, pans on tray and board, mug on board and tray, bowl on mug, plate on bowl, wok on pan, tray on board, bowl on tray.

| | fast | full | ratio |
|---|---|---|---|
| simulations | 11,735 | 19,720 | 0.595 |
| wall time | 387 s | 764 s | 0.51 |
| cells with the same verdict | 19,623 of 19,720 | | 99.5 % |
| stable cells lost / invented | 79 / 18 (of 9,574 stable) | | 0.8 % lost |
| best B-side margin | identical in all 12 pairs | | |
| best margin of one face | one face off by 11 mm (pan 0 on tray 0) | | |

The full run reproduced the stored labels exactly on the 11 pairs found in the stored set (0 cells differing).

The measured cost is 0.51, not the 0.33 estimated from the separate pieces. The adaptive sweep simulated 59.5 % of the cells instead of
52 % (more cells are refined on real sweeps than in the replay), and abort and model reuse save less once pool start-up and the two
passes of the adaptive sweep are included. Per pair, the saving is large where most cells fail or the stable set is compact (shaker on
mug 0.22, mug on tray 0.32, bowl on mug 0.39) and small where nearly all cells are stable (pan on board, tray on board 0.74).
The adaptive sweep stores the contact points of an inherited cell from its nearest simulated corner, an approximation for the support labels.
