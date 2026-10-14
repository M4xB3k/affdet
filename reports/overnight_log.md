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
