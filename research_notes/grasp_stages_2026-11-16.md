# Clamp / lift / level on 5 objects (2026-11-16)

Objects: shaker86450daa, mug0, frying_pan0, bowl0, plate7 (meshes, CoACD cache and picks_ext.json copied from the cluster;
the same files the full32 maps used). Grid 16 cells along the longer side, 6 yaws, default grasp settings except the wait
between "force reached" and the lift (settle). Data: `affdet/data/sim_results/2026-11-16_stages/` in the main checkout
(git-ignored): `cells16_settle0.1`, `cells16_settle0.5` (final), two `*_tiltFromLiftStart` folders (superseded, see 4).
Figure: `reports/figures/grasp_stages_2026-11-16/stages_settle0.5.png`. Full table: `.../stage_table.md`
(`scripts/grasp_stage_table.py`). Thresholds: `affdet/sim/grasp_stages.py` THRESHOLDS, defaults.
Everything here is 5 of 185 objects at 16 cells. It shows what to change before the 185-object run; it is not a result
for the whole set.

## Results (share of grasp tests on the object that are not blocked; settle 0.5 unless stated)

| object | old held (0.1 / 0.5 / full32) | old level: held & swing < 30 | new: held & tilt from rest < 30 | clamp | lift | level (nested) |
|---|---|---|---|---|---|---|
| shaker | 100 / 100 / 100 | 100 | 100 | 100 | 100 | 100 |
| mug0 | 85 / 85 / 87 | 70 | 85 | 88 | 83 | 83 |
| frying_pan0 | 93 / 93 / 90 | 4 | 6 | 98 | 92 | 6 |
| bowl0 | 100 / 100 / 99 | 23 | 43 | 97 | 97 | 42 |
| plate7 | 71 / 72 / 70 | 9 | 9 | 90 | 67 | 8 |

## What it shows
1. **The wait before the lift.** With the old 0.1 s the pads still press with about 300 N (target 30 N) when the lift
   starts; the clamp stage is then 0 % by construction. Held and old level barely change between 0.1 and 0.5 s (within 3
   points), and the 16-cell maps agree with full32. So the squeeze did not change who gets lifted on these 5 objects.
   0.3 s is enough for the shaker and the mug, but not for a thin curved wall: traced step by step on 2 bowl cells, the
   force swings 0-60 N for the first 0.3 s and comes to rest (30-34.5 N, no pad motion) after about 0.4-0.5 s. With 0.5 s
   the bowl's clamp share goes from 77 % (0.3 s) to 97 %. **Use settle 0.5 for the 185-object run.**
2. **The old level label counted spin as tilt.** `swing_deg` is the whole rotation from rest. When the pads close at an
   angle, they turn a mug or bar about the vertical to line up with them (traced: mug 37-39 deg of rotation, 2-3 deg of
   it tilt). It stays upright, but the old label called it not level. The new `tilt_max_deg` is the angle of the up axis
   from its resting direction, the maximum over lift and hold. Effect: bowl 23 -> 43 %, mug 70 -> 85 %, pan 4 -> 6 %,
   plate and shaker unchanged. The trained grasp model learned the old level30 label; `p2pheat/grasp.py build` now uses the
   tilt whenever a map has it.
3. **Level is still the main filter** for pans, bowls and plates (not level: pan 87 %, bowl 55 %, plate 59 % of tests).
4. **A tilt during closing counts.** A first version measured tilt from the lift start; the bowl traces showed the bowl
   tilting 18-23 deg while the pads close (pinching one wall lifts that side), which that version missed. Fixed: tilt is
   measured from the pose before closing (test `test_level_is_the_tilt_from_the_resting_pose_not_the_spin`).
5. **What the clamp stage catches.** mug0: real loss of contact at the mug's edge (both pads 0 N through the whole window;
   the mug squeezes out); 12 % of tests at 0.5 s, and some of these still lift by wedging, which the old label counted as
   success. plate7: 22 % of tests clamp but fail the lift; in all of them the plate stays on the table while the pads rise
   (slip 500 mm), and the old rule fails them too.

## Decisions for the user
- Settle 0.5 for the 185-object run (recommended above). Old results still reproduce with the default 0.1.
- The level label for the model: tilt from rest (recommended) instead of the old swing. This changes the level channels of
  the next training data; the old numbers stay as they are, with a note.
- Whether a clamp that fails but still lifts (mug edge wedging) is a negative for the grasp channel. The nested definition
  says yes.
