# Are the "no grasp" results on plates and cutting boards justified? (audit, 2026-11-10)

Question: the 185-object grasp maps show almost no graspable cells on cutting boards (3 successful cells in 86,442) and plates (268 in
81,900). Is that the physics, or an artefact of the simulator? Data: the stored 32-cell maps (`full32`), 16 plates and 24 boards, every
on-object cell and the 6 yaws. Scripts: `scripts/audit_blocked.py`, `audit_physics.py`, `audit_trace.py`, `audit_videos.py`; raw output in
`reports/figures/grasp_audit_2026-11-10/`. Videos: `figures/grasp_videos_2026-11-10/audit_notheld.mp4`, `audit_plateok.mp4`.

## What decides the outcome: the "blocked" rule, not the physics

`grasp_test` returns "blocked" before any physics when the pads cannot start low enough: with `z_top` the surface under the grasp centre and
`floor` the highest surface under either pad footprint, the pads start at `z0 = max(3 mm, z_top - 3 cm, floor + 1 mm)` and the cell is blocked
if `z0 > z_top - 8 mm`. This decides 95.2 % of plate cells and 99.4 % of board cells.

## 1. Is the rule implemented correctly? Yes

I re-derived the verdict from scratch (height fields at 2 mm, rays from above, the same 18 footprint points) on the simulator's collision
hulls. It reproduces the stored flag for 98.3 % of cells or more per object (plates 95.4 % blocked vs 95.2 % stored; boards 99.4 % vs 99.4 %).

## 2. Is it a hull artefact (convex decomposition fills dishes)? Almost never

Same re-derivation on the true visual mesh instead of the hulls: plates 94.9 % blocked (hulls 95.4 %), boards 99.4 % (99.4 %). Cells that only the
hulls block: 1.25 % (plates), 0.24 % (boards); cells that only the mesh blocks: 0.7 % and 0.18 %. Hulls inflate the top by 3.7 mm on average on
plates (2.4 mm boards), up to 47 mm at single points, but that does not move the verdict.

## 3. Is it the 8 mm threshold? Partly for plates, not for boards

Blocked share on the mesh with the clearance rule set to 3 / 5 / 8 / 12 mm: plates 85.1 / 89.9 / 94.9 / 97.8 %; boards 99.3 / 99.3 / 99.4 / 99.4 %.
In 75.8 % of plate cells and 98.8 % of board cells a pad footprint touches a surface at least as high as the pinched one, so no threshold helps:
two pads that close horizontally on a plate flat on the table, or on the middle or edge of a board, cannot both reach below the pinched surface.
A further 45.2 % of plate cells and 6.6 % of board cells are thinner than 11 mm at the pinch, below the lowest height the pads can start.

## 4. Would a narrower pre-grasp open anything? No

The pads always start at +-4.4 cm (8 cm opening). Reopening with a half-opening of 0.5 / 1 / 1.5 / 2.5 cm unblocks 0 / 1 / 154 / 481 plate cells
(0.6 % at most) and 0 / 0 / 16 / 9 board cells. Physics on a sample of the unblocked cells: 34 plate cells at 2.5 cm, 9 held, 1 level;
9 board cells, 0 held (all still blocked once the real pad height is used).

## 5. The unblocked minority: physics (replay matches the stored outcome in 269 of 269 cells)

Sample of 5 cells per object and stratum. Not held (plates 60, boards 105 replays): only 35 % (plates) and 17 % (boards) never lift the object
(max rise under 5 mm); 55 % and 60 % lift it 0.5 to 5 cm and lose it; 10 % and 23 % lift it 5 cm or more and then drop it. The held-but-not-level
cells swing 56 to 77 degrees (plates 7 and 11, board 11). Plates with successes (7, 14, 19, 22, 23, 25) are not separated from plates without by dish
depth (0.4 to 2.3 cm vs 0.1 to 2.1 cm); I did not find the separating property. Hypothesis from the videos, not tested: flat pads on a sloped thin
rim squeeze the plate out like a seed.

## Verdict

- Within the modelled gripper (top-down, two flat pads closing horizontally, 8 cm opening) the "no grasp" result on boards is geometry, not an
  artefact: 98.8 % of cells are blocked by geometric necessity and no tested threshold or opening changes it. Cutting boards are not
  graspable by this gripper except by a narrow handle or end.
- Plates: the blocked share is 85 to 98 % depending on the clearance rule, so "mostly not graspable" holds for any plausible value; the exact
  number (95 %) depends on the 8 mm rule. The unblocked remainder fails mainly by slipping out during the lift; that part I only inspected on a
  sample of replays and videos.
- What is not tested: other grasps (vertical fingers over a rim, pinch at a table edge, handle hole, suction), soft pads, a different mass
  (plates 0.5 kg, boards 1.5 kg are fixed values). The labels are valid for "top-down pinch" only.
