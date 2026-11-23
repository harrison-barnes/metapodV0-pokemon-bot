#!/usr/bin/env bash
# The PPO step that made MetapodV0: spec2 = 200 iterations of actor-learner PPO starting from v7@300 (policy and
# privileged critic), with our side bringing the two searched teams (data/search_teams_v1) half the time. With "mix1"
# only side 0 ever brings them; every other team is a real player's team (data/human_teams_v1.json). 30% of the envs
# play a league of earlier snapshots (PFSP exponent 1.5), 10% play two frozen human-imitation networks, and every 4th
# update also trains on exact 1v1 endgames from the tablebase. These are the defaults of the original recipe; spec2's
# own training log was not kept, so treat them as the recipe, not a verified transcript of that run.
#
# Needs (none of it is in this repository, see README "Training"):
#   $CKPT_DIR/v7_300.pt, v7_300_critic.pt   starting policy and critic
#   $CKPT_DIR/v5d_00400.pt, v6_latest.pt     league members (v7_300 is one too)
#   $CKPT_DIR/bc_full.pt, bc_skill.pt        human-imitation opponents
#   data/human_teams_v1.json                 real players' teams (from Metamon's parsed replays)
#   data/dataset_aws, data/dataset_pp        the exact 1v1 dataset behind the tablebase (scripts/build_tables.sh)
# Hardware: written for one CUDA GPU (24-48 GB) and 16-64 CPU cores; actors are CPU processes running the Rust engine.
#
#   RUN=spec2_repro CORES=32 bash scripts/train.sh
#   DEVICE=cpu ITERS=2 ENVS=128 MINIBATCH=256 ACT=2 RUN=smoke bash scripts/train.sh     # a tiny CPU smoke run
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-spec2_repro}
CKPT_DIR=${CKPT_DIR:-checkpoints/lineage}
INIT=${INIT:-v7_300}
DEVICE=${DEVICE:-cuda}
CORES=${CORES:-$(getconf _NPROCESSORS_ONLN)}
ACT=${ACT:-$(( CORES * 3 / 4 ))}          # actor processes: leave a quarter of the cores to the learner
ITERS=${ITERS:-200}
ENVS=${ENVS:-2048}
MINIBATCH=${MINIBATCH:-4096}
for f in "$CKPT_DIR/$INIT.pt" "$CKPT_DIR/${INIT}_critic.pt" "$CKPT_DIR/v5d_00400.pt" "$CKPT_DIR/v6_latest.pt" \
         "$CKPT_DIR/bc_full.pt" "$CKPT_DIR/bc_skill.pt" data/human_teams_v1.json; do
  [ -e "$f" ] || { echo "missing $f (see README, Training)" >&2; exit 1; }
done
ls data/dataset_aws/schema.json > /dev/null 2>&1 || { echo "missing data/dataset_aws (scripts/build_tables.sh)" >&2; exit 1; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BF16=--bf16; [ "$DEVICE" = cuda ] || BF16=
mkdir -p "agent/runs/$RUN"
uv run python -m agent.actor_learner --name "$RUN" --device "$DEVICE" \
  --actor-envs 64 --actor-threads 1 --max-lag 4 --actors "$ACT" \
  --init "$CKPT_DIR/$INIT.pt" --init-critic "$CKPT_DIR/${INIT}_critic.pt" \
  --teams "${TEAM_KIND:-mix1}:${TEAM_DIR:-data/search_teams_v1}:${TEAM_FRAC:-0.5}" --pretrain 0 \
  --lr 3e-5 --lr-final 1.5e-5 --ent-coef 0.003 --ent-final 0.002 \
  --envs "$ENVS" --steps 128 --epochs 1 --minibatch "$MINIBATCH" --clip 0.2 --gamma 0.995 --lam 0.95 --vf-coef 0.5 --grad-clip 0.5 \
  --eval-games 256 --league-frac 0.3 --pfsp 1.5 --human "$CKPT_DIR/bc_full.pt" "$CKPT_DIR/bc_skill.pt" --human-frac 0.1 \
  --adv-filter 0.75 --adv-keep 0.2 --aux-coef 0.5 \
  --tb-batch 1024 --tb-coef 0.5 --tb-every 4 --tb-policy 0.25 --cf-every 16 --cf-envs 64 --cf-seeds 2 --q-coef 1.0 \
  --magnet-coef 0.05 --magnet-every 50 --qhead --senses $BF16 \
  --league "$CKPT_DIR/v5d_00400.pt" "$CKPT_DIR/v6_latest.pt" "$CKPT_DIR/$INIT.pt" \
  --eval-vs "$CKPT_DIR/$INIT.pt" --iters "$ITERS" --eval-every 50 --eval-vs-every 50 --eval-vs-games 1024 --snapshot-every 50 \
  2>&1 | tee "agent/runs/$RUN/train.log"
# Snapshots: agent/runs/$RUN/snapshots/00200.pt (+ _critic.pt) is the spec2@200 equivalent.
