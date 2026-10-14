# Simulation labels: format

Labels for the proof-of-concept predicates come from the MuJoCo simulator (`affdet/sim/`), exported by
`affdet/sim/label_export.py` as one `.npz` per object and role. Each file holds the object's mesh and per-vertex
heatmap channels, in the object's own upright metric frame (z up, base at z = 0).

| array | shape | meaning |
|---|---|---|
| `vertices`, `faces` | (N, 3), (M, 3) | the mesh, subdivided to 6 mm edges |
| `valid` | (N,) bool | vertices that can carry the label (see below); labels are 0 elsewhere |
| `score` | (N,) in [0, 1] | `clip(margin_mm / 20, 0, 1)` |
| `margin_mm` | (N,) | millimetres a placement may be off before it fails |
| `meta` | json | schema version, cap, object names, thresholds, grid step |

## isP2PSupportive (A placed on B): two files

- **B file (the supporting object).** `valid` = upward-facing vertices. `margin_mm` is the best, over all faces of A,
  of the distance from that point to the nearest failing placement. `best_face` is the face of A that achieves it
  (index into `meta.faces`: under, on, front, back, left, right; -1 where nothing holds).
- **A file (the object put down).** `valid` = vertices on the resting patch of a face that holds on a flat table.
  `margin_mm` is that face's best margin over all positions on B. `face` is the face index.

A placement stays if, after 2 s of physics, A has not tipped more than 10 degrees relative to B, has not slid
more than 3 cm, has not fallen further than B's surface under it requires, is not still tumbling, still touches B,
and did not overlap B by more than 3 mm when released.

## canGrasp: one file per object

`valid` = upward-facing vertices (a top-down parallel-jaw gripper). `held_frac` is the share of gripper yaws at that
point that lift the object 0.5 m and keep it (no slip in the pads); `stable_frac` additionally requires it to hang
within 30 degrees of level.

## What a score means

`score` 1 means the placement tolerates 2 cm of error, which is about a robot's positioning accuracy; below that the
score is linear. The cap is a labelling choice, not a physical constant: `margin_mm` is stored so another cap can be
applied. Margins are quantised to the map's grid step (`meta.step`, 1 to 5 cm for most pairs).

## Known limits

Faces that do not hold on a flat table are not swept, so a round cap resting inside an opening is missed. Mass is
spread over parts by surface area, not by material. See the part-2 report for the open items.
