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
