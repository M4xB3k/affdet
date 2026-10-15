# Start here (overnight run, 2026-10-15)

Branch `sim-heatmap-analysis`, pushed to your fork only (`M4xB3k/affdet`), nothing merged, nothing pushed to `pniaz20`.

**Read first**
0. In one line: 276 pair maps (16 tops x 18 bases) + 702 with all 39 shakers, grasp and slide maps for 22 objects, per-vertex labels, eight bugs found and fixed on the way, checks that the maps are reproducible (identical across nodes), physically right (audit), consistent with geometry on flat faces, consistent with the earlier face study (234 of 234 faces) and that the score means what it says (survival under position error).
1. `reports/Simulation analysis part 3 - overnight run, scale-up on the cluster.md`: what was run, the seven bugs the scale-up exposed, the matrix, the checks.
2. The artifact (same content with pictures): https://claude.ai/artifact/5QmGaeENDLFoKe9A6uSgKm
3. `reports/overnight_log.md`: every step with its time, including dead ends.

**Look at**: `reports/figures/overnight/` (videos `video_*.mp4`, `gallery_*.png`, `contact_*.png`, `audit_*.png`).

**Data (not in git)**: `data/sim_results/2026-10-15/` (`README.md` there): raw maps and the per-vertex labels (`labels/`, format in `docs/sim_labels.md`).
Cluster copies: `/scratch/maksym.bek/sim/` (no backup).

**Re-run anything**: wrappers `scripts/slurm_sim_pairs.sh`, `slurm_sim_grasp.sh`, `slurm_sim_slide.sh`, `slurm_sim_contacts.sh` (cluster, from the clone `~/affdet-sim`);
labels from maps: `scripts/export_labels.py` + `scripts/merge_contacts.py`. Tests: `pytest affdet/tests/test_sim_support.py` (about 1 minute).

**Decisions that are yours**
- The tilt limit (10 degrees) and the score scale (margin / 20 mm): the labels move with the first; both alternatives are exported.
- Whether to open a pull request (`https://github.com/M4xB3k/affdet/pull/new/sim-heatmap-analysis`); nothing was opened.
- Which objects to add next (the pipeline takes any spec in `affdet/sim/registry.py`).
