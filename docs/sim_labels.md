# Simulation labels: format

Labels for the proof-of-concept predicates come from the MuJoCo simulator (`affdet/sim/`), exported by
`scripts/export_labels.py` (functions in `affdet/sim/label_export.py`). Layout of the output directory:

- `mesh/<object>.<role>.npz`: the object's subdivided mesh (`vertices` float32, `faces` int32) in its own upright metric frame
  (z up, base at z = 0). One per object and role, shared by all label files that name it in `meta.mesh`.
- `pair/<top>_on_<base>.A.npz` and `.B.npz`, `grasp/<object>.npz`, `slide/<object>.npz`: per-vertex channels (float16 or int8)
  aligned with the mesh named in `meta.mesh`.
- `index.json`: one line of summary statistics per label file.

| array | shape | meaning |
|---|---|---|
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

`margin_mm_tilt5` and `margin_mm_tilt15` (same shape as `margin_mm`) repeat the label with the tilt limit at 5 and 15 degrees instead of 10:
the limit is the threshold the maps are most sensitive to (about a quarter of the face maps move by more than 5 points of stable share
between 10 and 5 degrees, see the part-3 report), so a model can be trained on all three or on the spread.

A placement stays if, after 2 s of physics, A has not tipped more than 10 degrees relative to B, has not slid
more than 3 cm, has not fallen further than B's surface under it requires, is not still tumbling, still touches B,
and the object was released by lowering it straight down from above until it first touched anything (so there is no overlap at release).

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

## canSlide: one file per object

`valid` = side vertices (surface normal mostly horizontal). `slide_score` is 1 where a flat pusher pushing horizontally at that
point (inward, along the line to the object's centre) slides the object 12 cm without tipping it more than 10 degrees or turning
it more than 25 degrees; 0 otherwise. `moved_frac`, `tilt_deg` and `yaw_deg` carry the measured outcome. The map is sampled at
12 azimuths and 4 heights and painted on the mesh by nearest sample. The object settles for 1 s before the push.
