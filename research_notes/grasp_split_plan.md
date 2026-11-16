Plan written by a planning agent on 2026-11-16 (read-only design, nothing implemented).

# Plan: split "pickable" into CLAMP / LIFT / LEVEL in the grasp-map simulator

Read-only design; nothing was changed. Paths under `W = /home/mascarpone/max_uibk/uibk-courses/Semester_6/affdet/.claude/worktrees/sim-analysis` and `P = /home/mascarpone/max_uibk/uibk-courses/Semester_6/p2p-heatmaps`.

## 0. Three findings that change the brief

1. **The "2 s hold drops success to 0.150" finding is an averaging artefact, not physics.** In `W/reports/figures/grasp_2026-11-04/iter1_settings.md` the hold2 column equals base on every object except frying_pan0 (0.01 -> 0.00). hold2 is missing `shaker86450daa` (base 1.00, lost to the native crash), so its mean is over 9 objects. Base over the same 9 objects is 1.35 / 9 = 0.150. The real open problem from hold2 is the crash (`free(): invalid size`, pool hung), see Risks.
2. **Whether the pads ever reached the target force is not recorded today.** The lift starts on "both pads >= F" or on the 1.2 s timeout (`grasp_map.py:171`), and the result does not say which. This is the main missing field for CLAMP.
3. **"Level" today is not "tilt during lift and hold".** `swing_deg` is the full rotation angle (includes spin about the vertical), measured once at the end, relative to the pose before closing (`grasp_map.py:159, 194`). A swing that damps out before the end, or a spin about z, is scored differently from the supervisors' definition.

## 1. What is recorded today

`grasp_test` (`W/affdet/sim/grasp_map.py:99-202`) returns, per (cell, yaw):

| field | line | meaning |
|---|---|---|
| `x, y, yaw` | 200 | grasp centre and yaw |
| `on_object` | 136, 151, 201 | a vertical ray hits the object at (x, y) (133-136) |
| `blocked` | 149-151 | pads cannot start low enough: `z0 > z_top - 0.008`; only present when True; no physics run |
| `held` | 199 | finite, not touching the table at the end, `lifted > 0.05`, `slip < 0.05` |
| `lifted_m` | 193 | COM rise, end only |
| `swing_deg` | 194 | full rotation angle, end only, vs. pre-close pose |
| `slip_m` | 185, 195-196 | movement of the pinch point in the object frame, end only, reference taken at lift start |
| `success` | 201 | `held and swing < max_swing_deg` |
| `hold_force_n` | 174, 202 | mean of the two pads, hold phase only |
| `com0`, `params` | 158, 202 | start COM; settings used |

`grasp_map` (217-240) stacks these into `[ny, nx, nyaw]` arrays: `success, held, swing_deg, blocked, lifted_m, slip_m, on_object, hold_force_n`, plus `score, best_yaw_idx, xs, ys, yaws, step, com, params`. `scripts/sim_grasp_maps.py:50-51` adds `extents, spec, tag`.

Missing for the split:
- force reached or timeout; time of clamp; each pad's force separately; force at clamp time (only the hold mean exists)
- any measurement in the settle window (0.1 s, line 172): relative motion object/pads, pad closure still changing, gap between pads
- object displacement and rotation caused by closing
- maxima over time (tilt, slip) and per phase (lift vs hold); time of failure
- tilt of the up axis as opposed to full rotation
- the continuous quantity behind `blocked` (`z_top - z0`), and the hard-coded 0.008 / 0.003 / 0.001 (lines 137, 148, 149) are not parameters
- off-object and blocked cells return `swing_deg = slip_m = 0.0`, indistinguishable from a measured zero

Stale docstrings to fix in passing: line 5 says 35 cm (it is 0.50, line 31); line 102 says 5 cm/s (it is 0.10, line 163).

## 2. Stage definitions

Timeline of the one run (unchanged): close from t = 0; `t_c` = both pads >= F or timeout; settle window `Wc = [t_c, t_c + settle]`; lift `[t_s, t_s + lift_time]` with `t_s = t_c + settle`; hold `[.., + hold]`. Reference pose for stages 2 and 3 is the object pose at `t_s` (after clamping), not the pre-close pose.

### Stage 1 CLAMP (measured in Wc, object still on the table, pads at z0)

| condition | recorded quantity | proposed default | status |
|---|---|---|---|
| reachable | `clearance_m = z_top - z0` | >= 8 mm (today's rule) | assumption; audit shows plates move 85-98 % for 3-12 mm |
| force reached | `force_reached`, `t_clamp_s` | reached before `close_max` | definition |
| force kept on each pad | `f_l_min, f_r_min, f_l_mean, f_r_mean` over Wc | min >= 0.8 F on both | assumption, sweep 0.5-0.9 (regulator band is 1.0-1.15 F, lines 79-83) |
| something is between the pads | `gap_m = 2*open_half - sl - sr`, `at_travel` | gap >= 2 mm, neither pad at travel limit | 2 mm matches `grasp_analytic.py:131` |
| pads stopped closing | `closure_drift_m` = change of `sl + sr` over Wc | < 1 mm | assumption, sweep |
| object does not move in the grip | `clamp_creep_m` (pinch point in object frame, over Wc), `clamp_rot_deg` (object rotation over Wc), `v_end, w_end` at end of Wc | < 1 mm, < 1 deg, < 1 cm/s, < 0.3 rad/s | assumption; the speed values are the residual jitter noted at `placement_map.py:154` |
| recorded, not in the default verdict | `close_shift_m`, `close_rot_deg`, `close_tilt_deg` (pre-close pose to `t_s`), `pad_contact_l/r` (contact centroid in object frame, 3 floats each), `pad_force_angle_l/r_deg` (net pad force vs. closing direction) | none | a gripper that centres an object is fine; the angle is the antipodal measure if the supervisors want "geometry" stricter |

`settle` becomes a parameter. Default 0.1 s keeps old results reproducible; sweep 0.1 / 0.3 / 0.5 s on the 5 objects, since 100 steps is short for "stably".

### Stage 2 LIFT

Recorded: `slip_lift_max_m`, `slip_hold_max_m`, `slip_m` (end, as today), `lifted_m` (end), `lifted_max_m`, `on_table_end`, `finite`, `t_off_table_s`, `t_contact_lost_s` (first time after `t_s` that either pad force < 1 N; NaN if never).

Default: `finite and not on_table_end and lifted_m > lift - 0.05 and max(slip_lift_max, slip_hold_max) < 0.05`.

Assumptions: 5 cm slip (supervisors' number, keep); using the maximum instead of the end value; requiring the full rise instead of today's 5 cm. Slip is invariant to rotation about the pinch (it is measured in the object frame), so it does not double-count swing.

### Stage 3 LEVEL

Recorded: `tilt_lift_max_deg`, `tilt_hold_max_deg`, `tilt_end_deg` (angle of the object's up axis vs. its pose at `t_s`), and the same three for the full rotation `rot_*`. Legacy `swing_deg` stays exactly as is.

Default: `max(tilt_lift_max, tilt_hold_max) < 30`. Assumptions: 30 deg (report section 2 shows this is the sensitive threshold: bowl 0.05 at 30, 0.32 at 60); up-axis vs. full rotation; max vs. end. All three are switchable offline.

### Nesting

Store the three raw verdicts independently and derive nested labels offline: `clamp`, `lift = clamp & lift_raw`, `level = lift & level_raw`. Also store `fail_stage` (int8): 0 off object, 1 blocked, 2 force not reached, 3 clamp unstable, 4 lift failed, 5 not level, 6 success. The cross-table "clamp failed but lifted" (object scooped or wedged) is a diagnostic to report, not hide. Legacy `held` and `success` are kept unconditioned on clamp.

### Re-thresholding without re-simulation

Two layers:
- **Scalars** above, dense `[ny, nx, nyaw]` float32, NaN where no physics ran. Enough for any threshold on maxima.
- **Trace** (on by default, `trace_hz=50`): from `t_c` to the end, fixed length `(settle + lift_time + hold) * 50` = 100 samples at defaults, channels `tilt, rot, slip, obj_rise - pad_rise, f_l, f_r, on_table`, float16, stored as `trace [n_sim, T, 7]` plus `trace_index [ny, nx, nyaw]` (-1 = none). This gives time of failure for any threshold and any phase window. Worst case 32 x 32 x 6 x 100 x 7 x 2 B = 8.6 MB per object before compression; flat objects have almost none.

Limit to state clearly: the `blocked` clearance can be re-thresholded only in the stricter direction offline. Loosening it (3 mm) needs those cells simulated, so make `min_clearance` a parameter.

## 3. One run or three

One run. The phases are already sequential in the same loop (`grasp_map.py:168-190`); the stages are read at the phase boundaries. Three runs would repeat close + settle twice and lift once, roughly 2.2x the stepping, for no extra information (replays are deterministic: 269 of 269 in the audit).

Cost of the change:
- per-step recording is a few 3x3 products; negligible against the two Python contact loops in `_pad_force` (86-96, called twice per step at 170). Merge them into one pass that also returns table contact: about neutral, possibly faster.
- `settle` 0.1 -> 0.3 s would add 0.2 s to a 2.1-3.3 s run (6-10 %). Not needed at default.
- optional `fast` abort (off by default, same convention as `mujoco_support._simulate`, lines 178-190): stop once the object is back on the table and the pads are above it. Saves most of lift + hold on dropped cells.

Expected: 185 objects stay at about 25 s to 5 min each, same slurm array as `full32`.

## 4. Code changes

### `W/affdet/sim/grasp_map.py`
- `DEFAULTS` (42-44): add `settle=0.1, min_clearance=0.008, min_pad_z=0.003, trace_hz=50`. `grasp_params` (47-54) needs no change. Replace literals at 137, 148-149, 172.
- New `_contacts(m, data, bl, br, oid)`: one pass, returns `f_l, f_r`, force vectors, contact centroids, `on_table`. Replaces `_pad_force` calls at 170 and the end-only test at 197-198. Keep `_pad_force` as a thin wrapper (used by scripts).
- New `_empty(x, y, yaw, **kw)`: the result for off-object and blocked cells, all continuous fields NaN, legacy fields as today (0.0). Replaces the two literal dicts at 135-136 and 150-151; blocked adds `clearance_m`, `z_top`, `floor`.
- `grasp_test` loop (168-190): keep control flow and step order identical. Add a small `_Recorder` updated after `mj_step` (182): accumulates Wc statistics, per-phase maxima, first-event times, decimated trace. Take the stage-2/3 reference pose at `t_s` next to `rel0` (183-185).
- Return (191-202): all legacy keys unchanged, plus the new scalars and `trace`.
- New pure function `stages(m, **thr) -> dict` with `THRESHOLDS` dict (module level, no mujoco import): returns `clamp, lift_raw, level_raw, lift, level, fail_stage`. On a v1 map (no `force_reached`) it returns `clamp=None`, `lift=held`, `level=held & (swing_deg < thr)`. Modelled on `placement_map.finalize` (`W/affdet/sim/placement_map.py:143-164`).
- `grasp_map` (217-240): stack new scalars; build `trace`/`trace_index`; call `stages` and store `clamp, lift, level, fail_stage, thresholds`; add `schema=2`. Existing keys and shapes unchanged, so every current reader keeps working.
- Replace `Pool.map` (224-225) with `concurrent.futures.ProcessPoolExecutor` so a dead worker raises `BrokenProcessPool` instead of hanging; write the map only when complete.

### `W/affdet/sim/grasp_analytic.py`
- `grasp_analytic` (90-143): add `lifts = fits & (2 * mu * force >= mass * g)` so each stage has a rule: `fits` -> clamp, `lifts` -> lift, `level` -> level. Add `mu` argument defaulting to `gm.PAD_FRICTION[0]`.

### `W/affdet/sim/label_export.py`
- `grasp_labels` (94-102): add `clamp_frac`, `lift_frac`, `level_frac` via `stages`; keep `held_frac`, `stable_frac`. Skip `clamp_frac` on v1 maps.
- `W/docs/sim_labels.md:72` and `W/scripts/export_labels.py:84`: add the new channels and counts. `SCHEMA_VERSION` stays 1 (additive; a test asserts 1).

### `mujoco_support.py`, `heatmap_render.py`
No change. `heatmap_render.render_heat` takes any per-vertex heat and can be reused for the per-stage figures.

### Scripts
- `W/scripts/sim_grasp_maps.py:35-37`: add `--settle --min-clearance --trace-hz`; line 55-57: print clamp / lift / level cell shares.
- `W/scripts/slurm_sim_grasp.sh`: no change (passes `"$@"`).
- New `W/scripts/sim_grasp_stages.py`: re-score maps with other thresholds, write the per-stage figure and table.
- Unaffected readers (use only legacy keys): `audit_physics.py, audit_trace.py, audit_blocked.py, sim_grasp_compare.py, sim_grasp_compare_settings.py, sim_grasp_gallery.py, sim_grasp_jump_scan.py, sim_grasp_video.py`.

### `P/p2pheat/grasp.py`
- `CHANNELS` (34): `("clamp_any", "held_any", "level30_any", "clamp", "held", "level30")`. Keep the names `held` (= lift) and `level30` so `scripts/grasp_final_table.py:15`, `grasp_table.py:16`, `plot_grasp_preds.py`, `export_grasp_viewer.py` keep working. Add `LEGACY_CHANNELS` = today's four.
- `build` (88-137): at 120-123 add `clamp`, `clamp_any` from the map's stored `clamp`; choose by `--stage-def legacy|nested` whether `held`/`level<T>` come from legacy fields (default, comparable with existing runs) or from nested stages with max-tilt. Add `--thr k=v` to recompute from raw fields with a local copy of `stages`; assert that recomputing at the stored thresholds reproduces the stored booleans. At 128 add rule channels `clamp_any: fits.any`, `clamp: fits.mean`. Write `channels`, `thresholds`, `schema` into the index (136). v1 maps: no clamp channel, index says so.
- `GraspData.__getitem__` (167-168): take the channel list from the data dir, not the global.
- `make_model` (195-208): `make_model(d, k, channels)`; head is already `Linear(d, len(CHANNELS))` (203), so it goes 4 -> 6 outputs. `train` (262) stores `channels` in `config.json`; `evalfold` (295) and `arms` (316) build the model with `cfg.get("channels", LEGACY_CHANNELS)` so existing 4-output checkpoints still load.
- `evaluate` (221-232), `arms` (336, 343), `format_summary` (361), loss (274): iterate the model's channels.
- Alternative not recommended as default: conditional heads (lift given clamp), because the masks get very small on pans, trays, plates.

## 5. Tests first

In `W/affdet/tests/test_sim_support.py` (existing grasp tests at 63-72, 104-119, 209-216, 304-336; bar and walled-slab fixtures reusable):
1. Recording does not change physics: bar at d = 0, 0.05, 0.095 gives the same `held, swing_deg, slip_m, lifted_m` as before (pin the current values first).
2. Bar at the COM: `force_reached`, both `f_*_min` within the band, `clamp_creep_m < 1 mm`, `gap_m` about 3 cm; clamp, lift, level all True.
3. Bar at 9.5 cm: clamp True, lift True, level False; `tilt_lift_max_deg >= tilt_end_deg`.
4. `pad_friction` low with `pad_priority=1` (as test 322-329): clamp True, lift False, `t_contact_lost_s` finite and inside the lift phase.
5. A 5 mm wide blade or a cell where the pads meet: `force_reached` False or `gap_m < 2 mm`; clamp False.
6. Blocked ledge (113-119): `blocked`, `clearance_m < 0.008`, continuous fields NaN, `fail_stage == 1`.
7. `stages` is pure: changing `max_tilt` from 30 to 60 flips level on a synthetic map without touching other keys; nested implies `level <= lift <= clamp`.
8. `stages` on a v1 dict (only `held`, `swing_deg`): `lift == held`, level as before, clamp None.
9. Trace: length `(settle + lift_time + hold) * trace_hz`; max of the trace's tilt equals the stored maxima within float16 error; the trace alone reproduces `stages` at default thresholds.
10. `grasp_map` on the bar: `schema == 2`, new arrays have shape `[ny, nx, nyaw]`, legacy keys unchanged, `fail_stage` consistent with the booleans.
11. A worker that raises makes `grasp_map` raise, not hang.
12. `grasp_labels`: `clamp_frac >= lift_frac >= level_frac` on a synthetic map; v1 map still gives `held_frac`, `stable_frac`.
13. `grasp_analytic`: `lifts` False for a heavy bar at low force while `fits` True.

New `P/tests/test_grasp.py` (there are no grasp tests in `P/tests/test_basic.py`):
14. `build` on a synthetic v2 map and a flat point cloud writes all six channels, `clamp_any >= held_any >= level30_any` pointwise.
15. `build` on a v1 map writes the four legacy channels and records the missing clamp.
16. `make_model(channels=...)` output keys and count; a 4-channel state dict loads with `LEGACY_CHANNELS`.
17. `--thr` recomputation equals stored booleans at stored thresholds.

## 6. Validation on 5 objects before the cluster run

Objects: `shaker86450daa` (easy, and the object that crashed in hold2), `mug0`, `frying_pan0` (held but swings), `bowl0` (slowest, median swing 42 deg), `plate7` (95 % blocked, has held-not-level cells). Optional cheap sixth: `cutting_board0`.

Steps:
1. Run at 16 cells locally with default settings. Compare legacy fields cell by cell with a run of the unmodified code at the same grid (the local maps in `/home/mascarpone/max_uibk/uibk-courses/Semester_6/affdet/data/sim_results/2026-10-15/grasp_maps/` are from an older simulator version, so use them only as a rough check). Expect identical `held, swing_deg, slip_m`.
2. Time per object old vs. new; file size.
3. Stage funnel per object: on-object -> not blocked -> force reached -> clamp -> lift -> level, with the "lifted without clamp" count.
4. Legacy vs. new verdicts: `held` vs. `lift`, `success` vs. `level` confusion counts, split by cause (max vs. end, tilt vs. rotation, clamp conditioning).
5. Offline sweeps from the stored fields: tilt 15/30/45/60; slip 2/5/10 cm; force floor 0.5/0.8/0.9 F; creep 0.5/1/2 mm. Simulated sweeps: `settle` 0.1/0.3/0.5; `min_clearance` 3/8 mm on the plate; `hold` 0.4/2.0 on the shaker to try to reproduce the crash.
6. Replay 5 cells per stage outcome with `scripts/sim_grasp_video.py` and check the verdict by eye.

Figures (`W/reports/figures/grasp_stages_<date>/`):
- per object, one row: top-down maps of clamp_any, lift_any, level_any, and the fail-stage map (categorical), same axes, object outline
- the same three on the mesh via `label_export.grasp_labels` and `heatmap_render.render_heat`
- funnel bar chart per category
- histograms of max tilt, max slip, min pad force with the threshold lines
- threshold-sensitivity curves (share of cells vs. threshold) per stage

Then the 185-object run with `scripts/slurm_sim_grasp.sh` into a new directory (`full32_stages`), old `full32` kept for comparison.

## 7. Risks

- **hold2**: the 0.150 is the missing shaker (section 0). The crash is real and unexplained; with `Pool.map` it hangs the task. Mitigation: executor that raises, per-object output written only when complete, rerun failed array tasks, try to reproduce on the shaker locally.
- **Flat objects**: 95.2 % of plate cells and 99.4 % of board cells are blocked before physics. The clamp map for these is almost all zero and the clamp channel has no positives on most of them (AUC undefined, `P/p2pheat/metrics.py:24-32`). This is a statement about the top-down pinch, not about the object; say so in every caption. Store `clearance_m` so the rule's effect is visible.
- **Clamp is tested with the object on the table**: the table carries the weight and can balance unequal pad forces, so "stable" here means no relative motion, not force closure under load. Thresholds on creep and speed are assumptions near the solver's jitter level.
- **Timeout cells**: today they are lifted anyway and can count as held. Under the nested definition they fail at clamp. The share is unknown until measured; it could move `held_any`.
- **Definition changes move numbers** (max vs. end, tilt vs. rotation, full rise vs. 5 cm). Keep legacy channels as default in `build` until the comparison in step 4 is reviewed.
- **Trace size** on the cluster scratch: check on the 5 objects; fall back to `trace_hz=20` or scalars only.
- **Existing checkpoints** have 4 outputs; handled by the stored channel list, but tables mixing old and new runs must compare only the shared channels.
- Still one gripper type; pans and trays fail LEVEL by physics (torsional pad friction), which the split makes visible but does not fix.

## 8. Sliding with the same approach

`W/affdet/sim/slide_map.py` already stores continuous outcomes (`moved_frac, tilt_deg, yaw_deg`, lines 100-108) but only at the end, with thresholds applied inside `slide_test` (`MOVED_MIN, TILT_MAX, YAW_MAX`, line 32). Reuse the same pattern: a `_Recorder` in the push loop (93-99) for max tilt, max yaw, pusher contact force, time of contact, time of contact loss and a decimated trace; NaN for off-object rays (line 76); a pure `stages(m, **thr)` giving CONTACT (pusher reaches and keeps force on the object), SLIDE (`moved_frac >= 0.8`), UPRIGHT (max tilt < 10 deg, yaw < 25 deg); the same `fail_stage` code and executor. `label_export.slide_labels` (105-123) and `scripts/slurm_sim_slide.sh` need only the new channel names. Share the recorder and the executor wrapper between the two modules rather than copying them.

### Critical Files for Implementation
- /home/mascarpone/max_uibk/uibk-courses/Semester_6/affdet/.claude/worktrees/sim-analysis/affdet/sim/grasp_map.py
- /home/mascarpone/max_uibk/uibk-courses/Semester_6/affdet/.claude/worktrees/sim-analysis/affdet/tests/test_sim_support.py
- /home/mascarpone/max_uibk/uibk-courses/Semester_6/p2p-heatmaps/p2pheat/grasp.py
- /home/mascarpone/max_uibk/uibk-courses/Semester_6/affdet/.claude/worktrees/sim-analysis/affdet/sim/label_export.py
- /home/mascarpone/max_uibk/uibk-courses/Semester_6/affdet/.claude/worktrees/sim-analysis/scripts/sim_grasp_maps.py
