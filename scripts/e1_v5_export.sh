#!/bin/bash
# Pull the v5 ladder from the cluster ledger and regenerate every table that
# depends on it. Run from the repo root once all 15 v5 ladder runs are in.
#
#   bash scripts/e1_v5_export.sh
#
# Refuses to write if fewer than the expected seeds are present, because a
# partial mean over instance rungs is not evidence: each seed redraws which
# meshes are held out, and three of six draws describe a different experiment
# from six of six.
set -euo pipefail

SNAP=data/eval/e1_v5_ledger_snapshot.jsonl
REMOTE=ifi-cluster.uibk.ac.at
LEDGER=/scratch/maksym.bek/affdet/experiments/runs.jsonl

echo "== pulling v5 rows from the cluster ledger"
ssh -o BatchMode=yes "$REMOTE" "grep -E '\"name\": \"(e1[abc]-[a-z_]*-s[0-9]-v5|e9-v5-s[0-9])\"' $LEDGER" > "$SNAP"
echo "   $(wc -l < "$SNAP") rows -> $SNAP"

python3 - <<'PY'
import json, collections, sys
rows = [json.loads(l) for l in open("data/eval/e1_v5_ledger_snapshot.jsonl") if l.strip()]
# e9-v5-s* IS the scene rung: identical config to e1a-scene-s*-v5 in every
# field but `name`, so it is renamed here instead of being retrained.
seen = collections.Counter()
out = []
for r in rows:
    n = r["config"]["name"]
    if n.startswith("e9-v5-s"):
        r["config"]["name"] = f"e1a-scene-s{n[-1]}-v5"
    seen["-".join(r["config"]["name"].split("-")[:2])] += 1
    out.append(r)
with open("data/eval/e1_v5_ledger_snapshot.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

expected = {"e1a-scene": 3, "e1b-pair": 3, "e1c-actor_only": 6, "e1c-strict": 6}
short = {k: (seen.get(k, 0), v) for k, v in expected.items() if seen.get(k, 0) < v}
for k, v in sorted(seen.items()):
    print(f"   {k}: {v}")
if short:
    sys.exit(f"REFUSING TO EXPORT: rungs below seed count: {short}")
PY

echo "== regenerating tables"
python3 - <<'PY'
from affdet.experiment.result_tables import (
    write_ladder_tex, write_fidelity_tex, write_per_predicate_tex)
S, T = "data/eval/e1_v5_ledger_snapshot.jsonl", "tools/docs/thesis/tables/"
for p in (write_per_predicate_tex(S, T + "per_predicate.tex", rung="e1a-scene", corpus="v5"),
          write_ladder_tex(S, T + "ladder.tex", corpus="v5"),
          write_fidelity_tex(S, T + "fidelity.tex", corpus="v5")):
    print(f"   wrote {p}")
PY

echo "== rebuilding the thesis"
make -C tools/docs/thesis
echo "== done"
