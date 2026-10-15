# Overnight run log

Started 2026-10-15 00:00. Planned to run about 4 hours without input. Rules: push only to the fork (`mine`), never to
`pniaz20`; no merges; modest cluster use; PoC predicates are isP2PSupportive and canGrasp, canSlide only as a prototype.

## Plan
1. Finish the wide pair matrix (16 tops x 18 bases) on the IFI cluster and sanity-check every object type visually.
2. Quality loop: look for physics bugs and noise (re-measure run-to-run disagreement after the contact fix), fix, re-run.
3. Grasp: maps for the new objects, analytic comparison, canLift label from the lift phase.
4. New heatmap: canSlide prototype (where can the object be pushed so it slides without tipping).
5. Export all labels in the training format, write the report and update the artifact, checkpoint to the fork.

## Log
- 00:00 Wide pair matrix submitted (16 jobs, adaptive 2). Contact-count noise fixed (contacts counted over the last second).
- 00:01 canSlide prototype (affdet/sim/slide_map.py): tall narrow box topples when pushed high, flat box slides; tests + label export written.
- 00:02 Wrote scripts/export_labels.py (all maps -> per-vertex .npz + index) and scripts/pair_matrix_report.py.
- 00:10 Wide matrix v2 (16 cells, adaptive 2) done: 276 pairs, 65k sims, 1.4 job-h. Spot check: pan-on-tray map shows a thin plus-shaped region with speckles: likely adaptive artefact -> launching v3 = full sweeps, 32 cells along the long side.
- 00:12 canSlide: objects now settle for 1 s before the push, pusher height clamped (plates were pushed below the table). Re-running on all 22 objects; v3 full-resolution pair matrix running (16 jobs).
- 00:14 canSlide video (tall shaker: slides pushed at 15%/40%, topples at 65%/90%). Labels now blended smoothly between grid cells. 22 grasp maps (all objects, size-scaled grids) submitted.
- 00:15 Threshold sensitivity (711 face maps): tilt 5/15 deg shifts stable share by -4.4/+2.9 pts (20% of maps move >5 pts); release-overlap 3 mm rule is the most sensitive (6 mm: 144 maps move >5 pts). Replacing it by lowering the top from above until first contact (what a robot does); cancelled v3, will re-run as v4.
- 00:18 New release rule: the top is lowered straight down from above until it first touches anything (bisection), then released. Tests pass (31). Launching v4 = full sweeps, 32 cells, 16 tops x 17-18 bases.
- 00:30 Label export now stores each mesh once (mesh/) and float16 arrays per label file; docs/sim_labels.md updated incl. canSlide. Grasp vs analytic model on 20 objects: mugs/shakers agree, pans mostly, bowls weak (63 over-promised cells on bowl:0), boards: model says graspable, sim says no.
- 00:41 v4 job mug_1 crashed (MuJoCo arena out of memory with 219 contacts): arena enlarged to 256 MB, a failed solve now fails the placement; resubmitted (resumes).
- 00:42 Friction sensitivity (23 pairs, mu 0.4/0.6/0.8): no pair moves more than 10 pts (most 0-2 pts, worst 7 pts) -> support labels are insensitive to friction.
- 00:43 Release rule comparison on 72 identical pairs (v3 probe+overlap rule vs v4 lowering): 22 pairs change by >5 pts; pans nest in pans (39->89%), pans/bowls bridge a mug rim (3-4% -> 22-44%), previously missed because probes missed the rim. v4 is the more faithful rule.
- 00:55 Contact-region labels (which part of A touches which part of B at the most robust placement): works, e.g. bowl foot ring <-> mug rim. Cluster wrapper added, to be run on all pairs when v4 finishes.
- 01:13 v4 final matrix done: 276 pair maps, 539k sims, 7.9 job-hours (about 1 h wall on the cluster). Bulk label export started; contact-region jobs running on the cluster.
- 01:19 Results saved outside git: data/sim_results/2026-10-15/ (pair_maps_v4, grasp_maps, slide_maps, friction_runs, labels = per-vertex export of 276 pairs + 22 grasp + 22 slide, 113 MB) in the main checkout (gitignored). /home has only ~5 GB free.
- 01:22 Contact-region labels redone from MuJoCo's own contact points (visual-mesh distance missed thin rims because collision hulls are slightly larger); first cluster attempt was OOM-killed (distance queries on big meshes); re-submitting.
- 01:23 Found: video clips still used the old probe release while labels used the new lowering rule (clips showed bowl 'around' the mug). Clip builder now calls the same lower_until_touching() as stack_test and warns if its end pose differs from the label run's.
- 01:28 Found: stack_test used stricter stability limits (0.3 rad/s, 1 cm) than the stored maps (1.5 rad/s, 3 cm), so video labels/contact re-runs could disagree with maps for 14 of 274 best placements. One shared function placement_stable() now defines it for both; maps unchanged (same formula); test added.
- 01:30 The B-side map paints where A's centre can be, so a plate resting on a mug rim lit the mug floor. Added per-vertex SUPPORT labels from the contact points of all stable placements (best margin among the placements that touch each vertex). Needs contact points in every sweep cell: matrix re-run as v5.
- 01:35 Audit of 36 random placements: fresh run reproduces the map 36/36; visual review of final frames: 34 physically right, 2 fragile (round shaker across mug rim ring, margin 5 mm), 1 undecidable view. Grasp/hex/contact galleries made; part-3 report drafted.
- 01:43 Static geometry vs simulator on the 3 boards (48 pairs, 88,847 cells): flat faces agree on 84.9% (89.7% with a 3 deg tilt limit; the rest is boundary cells); on rim/ring faces the geometric rule has no support polygon (46%), where the simulator finds the real contacts.
- 01:44 Scale-out test: the other 35 single-object shakers as tops on the 18 bases (16 cells, full sweeps), 7 jobs x 5 shakers.
- 02:25 v5 (with per-cell contact points) done; v4 vs v5: all 539,201 cell verdicts identical in all 276 maps (different nodes). Rebuilding labels from v5 with support channels, tilt variants and contact regions.
- 02:28 Final galleries (support-channel mug gallery), refreshed pair figures, hex-shaker video with final rules; artifact updated (version 4); banners on parts 1 and 2 pointing to part 3.
- 02:37 Final labels built from v5 (support channels, tilt variants, contact regions merged for 274 pairs); saved to data/sim_results/2026-10-15/labels (+ pair_maps_v5). Zero solver failures/crashes in the final matrix; 2 pairs with no stable placement.
- 02:38 BUG found by validating 36 shakers against the face study: two shakers (a121f6c1, e07a892b) have up_axis=-x in the CSV and the registry ignored it, so their pair maps (36) were simulated lying down. Registry now passes the up axis (also to rendering); re-running those two.
- 02:39 Removed the invalid records of the two -x shakers from the scale-out groups and re-submitted them (fix job).
