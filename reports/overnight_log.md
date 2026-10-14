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
