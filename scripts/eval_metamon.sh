#!/usr/bin/env bash
# Paired evaluation of MetapodV0 against Metamon's Gen 1 OU bots (TaurosV0, Kakuna, TaurosEnsemble, KakunaEnsemble),
# all on one local Showdown server, the way the README's numbers were made (agent/next_eval_box.py; scoring:
# agent/next_eval_analyze.py). Metamon is not part of this repository; install it yourself:
#
#   git clone --recursive https://github.com/UT-Austin-RPL/metamon.git ../metamon   # we used commit 0a00a759
#   (cd ../metamon && uv venv --python 3.12 && uv pip install -e .)                  # Linux + CUDA recommended
#   (cd ../metamon/server/pokemon-showdown && npm install && node pokemon-showdown start 8000 --no-security &)
#   python scripts/patch_metamon_ensemble.py ../metamon        # only for TaurosEnsemble (an old checkpoint's key names)
#
# Then (two arms, live vs search off, 2 replicas x 25 battles per opponent; raise for real measurements):
#   METAMON_DIR=../metamon bash scripts/eval_metamon.sh
#   uv run python -m agent.next_eval_analyze agent/runs/eval_metamon/games.jsonl --ref live
#
# Every variable of the plan is documented at the top of agent/next_eval_box.py. Metamon's first run downloads its
# checkpoints from Hugging Face into METAMON_CACHE_DIR. Each Metamon bot is a 50-140M-parameter model: give it a GPU
# (GPU_SLOTS pairs at once) or use "TaurosV0@cpu".
set -euo pipefail
cd "$(dirname "$0")/.."
export METAMON_DIR=${METAMON_DIR:-../metamon}
[ -d "$METAMON_DIR/metamon" ] || { echo "METAMON_DIR=$METAMON_DIR is not a metamon clone" >&2; exit 1; }
curl -fs http://127.0.0.1:8000/ > /dev/null || { echo "no Showdown server on 127.0.0.1:8000 (start Metamon's, see above)" >&2; exit 1; }
export ARMS=${ARMS:-"live nosearch"}
export ARM_live=${ARM_live:-"--ckpt checkpoints/spec2_00200.pt --teams data/search_teams_v1 --search-tau 0.02 --infer"}
export ARM_nosearch=${ARM_nosearch:-"--ckpt checkpoints/spec2_00200.pt --teams data/search_teams_v1"}
export OPPS=${OPPS:-"TaurosV0:2:25 Kakuna:2:25"}
export SLOTS=${SLOTS:-4} GPU_SLOTS=${GPU_SLOTS:-4} TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-240} PAIR_LIMIT_MIN=${PAIR_LIMIT_MIN:-120}
export ARM_DEVICE=${ARM_DEVICE:-cpu} OUT=${OUT:-agent/runs/eval_metamon}
uv sync --group showdown > /dev/null
uv run --group showdown python -m agent.next_eval_box
