#!/usr/bin/env bash
# Rebuild the exact 1v1 data behind MetapodV0 from the engine (none of it needs to be rebuilt to play):
#   1. data/dataset_aws: every ordered pair of the 156 ladder sets (data/sets_ladder.json) solved exactly - value, both
#      sides' equilibrium move mixes and the payoff of every move pair - on a grid of fresh-entry roots (HP, status,
#      Sleep Clause). This is the big one: about 12k pairs, 7 GB of rows; it was run on a fleet of cloud machines,
#      one `--shard K/N` per machine, each writing data/dataset_aws/shard-K. The per-pair solve times recorded in
#      its done.jsonl files add up to about 1,100 core-hours (and about 230 more for step 2).
#   2. data/dataset_pp: the pairs whose equilibrium is a long stall (data/stall_pairs.json) solved again with healing
#      moves' PP tracked (rules smogon-pp), replacing their rows.
#   3. data/matchup_table_v2.npz (+ _hp_left.npz): the lookup table the network's matchup inputs read (agent/matchups.py).
# The tablebase used in training (agent/tablebase.py) reads 1 and 2 directly.
#
#   THREADS=64 SHARD=0/1 bash scripts/build_tables.sh
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS=${THREADS:-$(getconf _NPROCESSORS_ONLN)}
SHARD=${SHARD:-0/1}
(cd engine && cargo build --release --bin meta)
META=engine/target/release/meta
k=${SHARD%/*}
"$META" dataset --rules smogon --sets data/sets_ladder.json --out "data/dataset_aws/shard-$k" --mid 500 --threads "$THREADS" --shard "$SHARD"
"$META" dataset --rules smogon-pp --sets data/sets_ladder.json --pairs data/stall_pairs.json --out "data/dataset_pp/pp_shard-$k" \
  --mid 500 --threads "$THREADS" --shard "$SHARD"
# After every shard has finished, with all of them copied to one machine (the healing-PP rows go last and win):
uv run python -m tools.build_matchup_table data/matchup_table_v2_full.npz data/dataset_aws/shard-* data/dataset_pp/*_shard-*
uv run python - <<'PY'
import numpy as np
z = np.load("data/matchup_table_v2_full.npz")
np.savez_compressed("data/matchup_table_v2.npz", **{k: z[k] for k in z.files if k != "hp_left"})
np.savez_compressed("data/matchup_table_v2_hp_left.npz", hp_left=z["hp_left"])
print("wrote data/matchup_table_v2.npz and data/matchup_table_v2_hp_left.npz")
PY
