#!/usr/bin/env bash
# MetapodV0 on Pokemon Showdown: spec2@200 + exact one-turn search (tau 0.02, --infer, K=8), one of the two searched
# teams per battle. The account comes from the environment, never from a file in this repository:
#
#   export SHOWDOWN_USERNAME=your_bot_account SHOWDOWN_PASSWORD='...'
#   N=5 bash scripts/play_showdown.sh                  # 5 Gen 1 OU ladder battles on play.pokemonshowdown.com
#   SERVER=local ACCEPT=yourname bash scripts/play_showdown.sh   # accept challenges on a local server (127.0.0.1:8000)
#
# Showdown's rules for bots apply: use an account that is clearly marked as a bot, and keep CONCURRENT low.
# Every battle is appended to $OUT/results.jsonl; replays (HTML) go to $OUT/replays and per-decision traces to $OUT/trace.
set -euo pipefail
cd "$(dirname "$0")/.."
SERVER=${SERVER:-showdown}
OUT=${OUT:-agent/runs/showdown}
if [ "$SERVER" != local ] && [ -z "${SHOWDOWN_USERNAME:-}" ]; then
  echo "set SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD first" >&2; exit 1
fi
MODE=(--ladder); [ -n "${ACCEPT:-}" ] && MODE=(--accept "$ACCEPT"); [ -n "${CHALLENGE:-}" ] && MODE=(--challenge "$CHALLENGE")
NAME=(); [ -n "${SHOWDOWN_USERNAME:-}" ] || NAME=(--name "${NAME_LOCAL:-metapodv0-local}")
mkdir -p "$OUT"
uv run --group showdown python -m agent.showdown_player \
  --server "$SERVER" "${NAME[@]}" "${MODE[@]}" --battles "${N:-10}" --concurrent "${CONCURRENT:-1}" --timer \
  --ckpt "${CKPT:-checkpoints/spec2_00200.pt}" --teams "${TEAMS:-data/search_teams_v1}" --search-tau 0.02 --infer \
  --device "${DEVICE:-cpu}" \
  --results "$OUT/results.jsonl" --save-replays "$OUT/replays" --trace "$OUT/trace" 2>&1 | tee -a "$OUT/showdown.log"
