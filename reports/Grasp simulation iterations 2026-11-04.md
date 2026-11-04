# Grasp simulation: settings, level rule, grid (2026-11-04)

Goal for 2026-11-10: canGrasp heatmaps from the simulator, for every object of the support data, and a model trained on them.
Code: `affdet/sim/grasp_map.py` (settings now parameters, `grasp_params`), `scripts/sim_grasp_maps.py`. Cluster:
`/scratch/maksym.bek/grasp/{iter1,iter2,full32}`. Figures and tables: `reports/figures/grasp_2026-11-04/`.

The simulated gripper: two flat pads, top-down, closing horizontally, 8 cm opening, force-regulated, lift 50 cm, hold 0.4 s,
6 yaws (0 to 150 deg). Per (cell, yaw): held = lifted and not slipping more than 5 cm; level = held and turned less than 30 deg.

## 1. Settings barely change the maps (iteration 1)

10 objects (2 shakers, 2 mugs, 2 pans, bowl, tray, board, wok), 10-cell grid, 6 settings, one cluster task each (9 to 45 s).

| setting | what changes |
|---|---|
| base | 30 N, pads 2 x 2.5 cm, pad friction 1.0 / 0.01 / 0.001 (sliding / torsional m / rolling m) |
| f70 | 70 N (Franka hand, continuous) |
| panda | 70 N, Menagerie Panda fingertip pad 1.7 x 1.7 cm, Menagerie friction 1.0 / 0.005 / 0.0001 |
| mu05 | pad sliding friction 0.5 (with pad priority, so the pad's value is the contact's) |
| tors005 | torsional friction 0.005 m |
| hold2 | hold 2 s instead of 0.4 s |

Mean success share over the objects: base 0.236, f70 0.242, panda 0.240, mu05 0.216, tors005 0.236, hold2 (9 objects) 0.150*.
Per object (`iter1_settings.md`): all settings within 0.01 except friction 0.5 on the round shaker (0.96 -> 0.82) and
70 N on frying pan 0 (0.01 -> 0.07). The settings do reach the physics (measured hold force 32 N vs 74 N; swing angles change by
up to tens of degrees; some held/dropped outcomes flip), but the success map is set by geometry and the level rule.

\* hold2 lost the square shaker to a native crash (`free(): invalid size`, worker pool hung, cancelled); its mean is over 9 objects.

MuJoCo note: with equal geom priority a contact uses the element-wise maximum of the two frictions, so the pads' 1.0 is the
effective grasp friction (not the objects' 0.6). Lowering it needs `pad_priority=1` (tested).

## 2. The level rule is the sensitive choice

Same base maps, success re-scored with other level limits (no new simulation; swing is stored):

| object | < 15 deg | < 30 deg (current) | < 60 deg | held only | median swing when held |
|---|---|---|---|---|---|
| shakers | 0.96-1.00 | 0.96-1.00 | same | same | 4 deg |
| mugs | 0.09-0.12 | 0.15-0.18 | 0.18-0.19 | 0.18-0.19 | 11-13 deg |
| bowl 0 | 0.03 | 0.05 | 0.32 | 0.32 | 42 deg |
| pans | 0 | 0-0.01 | 0-0.11 | 0.13-0.19 | 53-78 deg |
| tray 0, wok 0 | 0 | 0 | 0.03-0.10 | 0.11-0.17 | 57-66 deg |
| board | 0 | 0 | 0 | 0 | - |

A top-down pinch resists a pan's weight only with the pads' torsional friction, so pans, trays and woks swing. This is physics,
not a bug, but it means the current gripper cannot pick these level: a handle grasp with the fingers above and below the handle
(UTAMP-2's handle grasp) is not simulated.

## 3. Grid resolution (iteration 2)

16, 24, 32 cells along the longer side: success shares change little (mugs 0.15-0.17, pans 0-0.01, bowl 0.08), held cells on
pans 15 % -> 18 % (thin handles need fine cells). A 32-cell map takes 25 s to 5 min (bowls slowest), so the full run uses 32.

## 4. Full run and labels

- 185 objects (all objects of the support data: 39 shakers, 29 mugs, 27 pans, 25 bowls, 24 boards, 23 trays, 16 plates, 2 woks),
  32 cells, base settings: array 1773518, `full32/maps/`.
- Grid and model point clouds share one frame (checked by overlay, `frame_check.png`, and by the rebuilt mesh: median 0.0 mm).
- Which points carry a grasp label: the first surface a vertical ray from above meets (exact on the mesh). The normal rule
  (normal z > 0.5) misses the inside of single-sided shells: bowl 0 has 71 points facing up by normal, 6,233 seen from above.
  The same normal rule is used for the support B-map mask in `label_export.py`; worth checking for bowls and plates.
- Label channels: held / level30 as the share of yaws, and `_any` (some yaw works). A rim is pinched by only 1 or 2 of 6 yaws, so
  the share marks rims negative; the planner picks the yaw, so `_any` is the main label.

Known limits: one gripper type (top-down pinch); no yaw finer than 30 deg; grid stops up to one cell short of the far edges;
friction and mass model as for support.
