#!/usr/bin/env bash
# A two-iteration CPU run of the PPO actor-learner, to check the training code end to end without any of the training
# inputs (scripts/train.sh lists them): the shipped checkpoint stands in for the starting point, the league and the
# human-imitation opponent, teams come from the ladder set pool, and the tablebase is off. About two minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
C=checkpoints/spec2_00200
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
uv run python -m agent.actor_learner --name "${RUN:-smoke_cpu}" --device cpu --actor-envs 32 --actor-threads 1 --max-lag 4 --actors 2 \
  --init $C.pt --init-critic ${C}_critic.pt --teams pool --pretrain 0 --lr 3e-5 --ent-coef 0.003 \
  --envs 64 --steps 16 --epochs 1 --minibatch 256 --eval-games 8 --league-frac 0.3 --pfsp 1.5 --human $C.pt --human-frac 0.1 \
  --adv-filter 0.75 --adv-keep 0.2 --aux-coef 0.5 --cf-every 1 --cf-envs 4 --cf-seeds 2 --q-coef 1.0 --magnet-coef 0.05 \
  --magnet-every 50 --qhead --senses --league $C.pt --eval-vs $C.pt --iters 2 --eval-every 50 --eval-vs-every 50 \
  --eval-vs-games 8 --snapshot-every 1
