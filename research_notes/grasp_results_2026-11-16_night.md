# Grasp: backbone rows and stage labels on all 185 objects (night of 2026-11-16)

Everything here ran on the IFI cluster on 2026-11-16 between 23:00 and 24:00. Tables as files:
`p2p-heatmaps/reports/grasp_2026-11-16/` (`backbone_table.md`, `stage_rows.md`, `old_on_stage_labels.md`) and
`reports/figures/grasp_stages_2026-11-16/stage_categories_185.md` in this worktree.
Code: p2p-heatmaps `master` (commits 6cf3a1d, 132ae5e and the one after), affdet `sim-heatmap-analysis` on the fork.
Cluster data: maps `/scratch/maksym.bek/grasp/full32s/maps`, labels `/scratch/maksym.bek/grasp/data32s`, runs
`/scratch/maksym.bek/grasp/runs/g_pnn_iw01_{r15,r10,pn,pp,k32}[_s1,_s2]` and `g_s_{pnn,pnn_c6,r15_c6}[_s1,_s2]`.

All model rows: the recipe of the reference run `g_pnn_iw01` (partial views 0.7, no normals, loss weight 0.1 for points off
the top surface, 120 epochs, batch 8, 5 folds over objects, 3 seeds, 2,048 training / 4,096 test points). AUC is per
held-out object on its top-surface points, then the mean over objects. "Best point" = share of objects whose highest-scored
point is really graspable. sd = standard deviation of the object mean over the 3 seeds.

## 1. Architecture rows (old labels, data32r)

Only the encoder changes.

| encoder | parameters | held_any AUC (sd) | best point | level30_any AUC (sd) | best point |
|---|---|---|---|---|---|
| EdgeConv, 16 nearest points (reference) | 216,196 | 0.872 (0.002) | 0.84 | 0.835 (0.000) | 0.76 |
| EdgeConv, 32 nearest points | 216,196 | 0.874 (0.002) | 0.81 | 0.840 (0.000) | 0.75 |
| EdgeConv, ball of 15 mm (at most 64 points) | 216,196 | **0.879** (0.001) | **0.85** | **0.842** (0.002) | **0.79** |
| EdgeConv, ball of 10 mm (at most 64 points) | 216,196 | 0.870 (0.001) | 0.79 | 0.836 (0.003) | 0.76 |
| PointNet (no neighbourhood) | 216,196 | 0.848 (0.001) | 0.73 | 0.814 (0.000) | 0.74 |
| PointNet++ style (hierarchy) | 497,220 | 0.860 (0.004) | 0.78 | 0.816 (0.003) | 0.73 |
| analytic rule (needs the full mesh and mass) | - | 0.921 | 0.85 | 0.804 | 0.73 |

n = 158 objects for held_any, 120 for level30_any (objects where both classes occur).

What it says:
1. **The local neighbourhood matters.** Without it (PointNet) held_any drops by 0.024 and the best point is right on 73 %
   of objects in place of 84 %. This is 12 times the seed spread.
2. **A fixed 15 mm ball is the best row, by a small margin:** +0.007 on both labels against the reference, 3 to 7 times the
   seed spread. The gain is concentrated on cutting boards (held_any 0.719 -> 0.781, level 0.567 -> 0.664) and frying pans
   (level 0.721 -> 0.760), the large objects where 16 neighbours reach furthest. It is the same size as the gap between
   partial views and full mesh input (0.872 against 0.879).
3. **10 mm is too small** (no better than the reference, best point worse). 32 nearest points is about the same as 16.
4. **The hierarchy did not help.** The PointNet++ style encoder has 2.3 times the parameters and is 0.012 (held) and 0.019
   (level) below the reference. Caveat: it is our own small version (2 levels with 512 and 128 centres, balls of 20 and
   50 mm, LayerNorm, no tuning of radii or widths, same 120 epochs), not the paper's network with its training recipe. Fair
   wording: "a hierarchical encoder with default settings did not beat the flat EdgeConv; not tuned".
5. None of the encoders closes the gap to the analytic rule on held_any (0.921). All of them beat it on level.

Per category (held_any / level30_any, mean over seeds):

| category | kNN 16 | ball 15 mm | PointNet | PointNet++ style | analytic rule |
|---|---|---|---|---|---|
| bowl | 0.949 / 0.868 | 0.952 / 0.871 | 0.947 / 0.865 | 0.945 / 0.862 | 0.983 / 0.660 |
| cutting_board | 0.719 / 0.567 | 0.781 / 0.664 | 0.683 / 0.671 | 0.697 / 0.270 | 0.936 / 0.500 |
| frying_pan | 0.912 / 0.721 | 0.908 / 0.760 | 0.851 / 0.618 | 0.908 / 0.747 | 0.899 / 0.545 |
| mug | 0.803 / 0.800 | 0.814 / 0.810 | 0.799 / 0.797 | 0.801 / 0.799 | 0.963 / 0.956 |
| plate | 0.879 / 0.861 | 0.871 / 0.866 | 0.853 / 0.806 | 0.891 / 0.867 | 0.710 / 0.728 |
| shaker | 0.868 / 0.868 | 0.871 / 0.871 | 0.858 / 0.858 | 0.833 / 0.833 | 0.920 / 0.920 |
| tray | 0.899 / 0.823 | 0.904 / 0.825 | 0.846 / 0.742 | 0.888 / 0.772 | 0.886 / 0.588 |
| wok | 0.965 / 0.836 | 0.968 / 0.833 | 0.954 / 0.835 | 0.959 / 0.775 | 0.989 / 0.705 |

The cutting-board level column rests on very few objects (almost no board position is level); do not read it.
Mugs are where the model loses most against the rule (0.80 against 0.96), with every encoder. That is the place to look
next, not the backbone.

## 2. Clamp / lift / level on all 185 objects

New maps: the same 185 objects, 32 cells, 6 yaws, 0.5 s wait before the lift, stages recorded. 177 objects have at least one
test that is not blocked (102,021 tests). Share of an object's tests, mean over the objects of a category:

| category | objects | blocked % | old held | old level | held (0.5 s) | level, old rule (0.5 s) | clamp | lift | level | lifted without clamp | held & tilt < 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bowl | 25 | 70 | 94 | 23 | 95 | 23 | 87 | 83 | 39 | 12 | 44 |
| cutting_board | 22 | 99 | 17 | 0 | 17 | 0 | 80 | 15 | 0 | 2 | 0 |
| frying_pan | 27 | 80 | 64 | 1 | 64 | 1 | 93 | 60 | 2 | 4 | 2 |
| mug | 29 | 78 | 87 | 79 | 88 | 79 | 82 | 79 | 79 | 9 | 88 |
| plate | 12 | 94 | 32 | 14 | 33 | 15 | 82 | 31 | 14 | 2 | 15 |
| shaker | 39 | 28 | 90 | 90 | 90 | 90 | 95 | 89 | 89 | 1 | 90 |
| tray | 21 | 89 | 51 | 12 | 52 | 12 | 82 | 44 | 17 | 8 | 22 |
| wok | 2 | 75 | 87 | 3 | 88 | 3 | 88 | 77 | 4 | 11 | 4 |
| all | 177 | 71 | 68 | 39 | 69 | 39 | 87 | 64 | 42 | 5 | 44 |

"old" = the full32 maps (0.1 s wait). "level, old rule" = held and whole rotation under 30 deg. clamp, lift, level are
nested (level needs lift, lift needs clamp). The last column drops the clamp condition.

What it says:
1. **The 300 N squeeze did not change the old results.** Held 68 -> 69 %, old level rule 39 -> 39 %, per category within
   1 point. The old maps stand. This confirms on 185 objects what the 5 local objects showed.
2. **Counting spin as tilt cost bowls, mugs and trays:** with the tilt of the up axis from rest, bowls go from 23 to 44 %
   level, mugs from 79 to 88 %, trays from 12 to 22 %. Pans, woks, plates and boards do not move: they really tip.
3. **Clamping is rarely the problem (87 %). Lifting and staying level are.** Boards clamp in 80 % of the tests that are
   not blocked and lift in 15 %; pans clamp in 93 %, lift in 60 % and stay level in 2 %.
4. **"Lifted without a clamp" is 5 % overall**, 12 % on bowls, 11 % on woks, 9 % on mugs. The old label counted these as
   success. Whether they are a negative is still the user's decision (the nested stages say yes).

## 3. Models on the stage labels (data32s)

| run | held_any | level30_any | held | level30 | clamp_any | clamp |
|---|---|---|---|---|---|---|
| old model (trained on old labels), scored on the new | 0.872 / 0.84 | 0.839 / 0.77 | 0.813 | 0.808 | - | - |
| old model with 15 mm ball, scored on the new | 0.880 / 0.85 | 0.847 / 0.79 | 0.826 | 0.810 | - | - |
| trained on the new labels, 4 channels (g_s_pnn) | 0.872 / 0.83 | 0.839 / 0.75 | 0.812 | 0.804 | - | - |
| ... plus clamp channels (g_s_pnn_c6) | 0.874 / 0.82 | 0.837 / 0.75 | 0.815 | 0.805 | 0.892 / 0.83 | 0.809 |
| ... plus clamp channels, 15 mm ball (g_s_r15_c6) | **0.883** / 0.85 | **0.848** / 0.78 | 0.826 | 0.813 | **0.898** / 0.88 | 0.821 |
| analytic rule on the new labels | 0.921 / 0.85 | 0.813 / 0.76 | 0.900 | 0.851 | 0.914 / 0.90 | 0.887 |
| height rule | 0.842 / 0.82 | 0.825 / 0.75 | 0.805 | 0.799 | 0.810 / 0.80 | 0.788 |

AUC / best point; 3 seeds each; seed sd at most 0.003. n = 158, 120, 133, 96, 176, 149 objects per column.

What it says:
1. **The label fix does not change the model's score.** The old model scored on the corrected level label gets 0.839; a
   model trained on the corrected label gets 0.839. So the reported numbers are not an artefact of the spin problem, and
   no retraining is needed to quote them. (The label is still the more correct one to describe.)
2. **Adding the clamp channels costs nothing** on the four old channels (differences within the seed spread), so one
   model can give clamp, hold and level maps.
3. **Clamp is the easy stage for the rule** (0.914; it is close to "do the pads fit"), and the model is behind (0.892 to
   0.898). By category the model wins on plates (0.914 against 0.661) and trays (0.936 against 0.879) and loses on mugs
   (0.785 against 0.929) and shakers.
4. The 15 mm ball is again the best model row, on every channel.

## 4. A faulty cluster node (gc16)

All 10 crashed simulation tasks and one crashed training fold ran on node gc16. The training crash was an index of
70368744179792 (2^46 + a small number) into an array of 4,854: one flipped bit, which points at bad memory on that node.
The new process pool turned these into errors; the old one would have hung.
What was done: the 10 crashed tasks were run again on other nodes. The 14 simulation tasks that had finished on gc16
were also run again elsewhere and compared: 13 agree (differences of 1e-16 in one field), frying_pan21 differs in 5 of 3,648
tests (which may also be the different CPU type). The maps folder now holds the reruns; the gc16 versions are in
`full32s/maps_gc16`. The labels in data32s were built before the swap, so frying_pan21 carries those 5 tests; too small
to matter, but a rebuild would make it exact. The 8 training folds that had run on gc16 were trained again on other nodes;
the tables above use the reruns.
To do for the user: report gc16 to the cluster admins, and add `--exclude=gc16` to job scripts until it is fixed.

## 5. What to say on Tuesday

- "I tested the neighbourhood. Local shape helps (PointNet is 0.024 worse). A fixed 15 mm ball is slightly better than 16
  nearest points (+0.007), mostly on large flat objects. A hierarchy with default settings did not help. So the backbone
  is not where the remaining error is."
- "I split the label into clamp, lift and level on all 185 objects. Clamping works in 87 % of the tests that are not
  blocked; the losses are at the lift (boards, plates) and at staying level (pans, woks, bowls)."
- "Two problems found in my own simulator and checked: lifting from a 300 N squeeze (no effect on the outcome) and counting
  spin as tilt (it did change the level label of bowls and mugs, but the model's score is the same on the corrected label)."

## 6. Not done
- No tuning of the PointNet++ style encoder; no pretrained backbone row.
- The camera arms (RGB-D view, TripoSR) were not re-run for the new rows.
- The recipe (loss weight, epochs) is still the one chosen on these same folds.
- 7 literature notes still unchecked: PoinTr, Minkowski, Depth Anything V2, LRM, TRELLIS, InstantMesh, single-view-ready. Checked tonight: Point Transformer, PTv3, AnyGrasp, TripoSR.
