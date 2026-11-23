"""Time per decision of ExactSearchPlayer configurations on single battles (as on Showdown: one battle per search).

Positions come from in-engine games between the plain policy and itself (sampled), stopped at assorted turns.
    PYTHONPATH=agent/ext_search2:. uv run python -m agent.next_search_bench --device cpu --positions 24 \
        --configs '{"k":8}' '{"k":32}' '{"k":8,"depth":2}'
"""
import argparse
import json
import time

import gen1py  # noqa: F401
import numpy as np
import torch

from agent.arena import Player
from agent.search_exact import ExactSearchPlayer
from agent.selfplay import Runner


def positions(ckpt, device, n, seed, teams=None):
    """n single-battle envs at assorted turns (side 0 has a real choice), from sampled policy self-play."""
    r = Runner(n, seed, device, teams=teams)
    p = Player(ckpt, device, sample=True)
    stop = r.rng.integers(1, 45, n)
    for t in range(60):
        team, battle, mask = r.state()
        acts = np.full((n, 2), 11, np.int64)
        go = (t < stop) | (mask[:, 0].sum(-1) <= 1)
        go &= r.env.results() < 0
        if not go.any():
            break
        i = np.flatnonzero(go)
        for s in (0, 1):
            acts[i, s] = p.actions(team[i], battle[i], mask[i], s, r.rng)
        # frozen battles: step needs a legal action, so they are copied out below instead of stepped
        keep = gen1py.VecEnv(n, seed=1)
        keep.copy_from(r.env, list(range(n)), list(range(n)), 1)
        legal = np.where(mask.any(-1), mask.argmax(-1), 11)
        acts[~go] = legal[~go]
        r.env.step(acts)
        stay = [int(x) for x in np.flatnonzero(~go)]
        if stay:
            r.env.copy_from(keep, stay, stay, 1)
    team, battle, mask = r.state()
    ok = np.flatnonzero((r.env.results() < 0) & (mask[:, 0].sum(-1) > 1))
    return r.env, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/spec2_00200.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=['{"k":8}'])
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    env, ok = positions(a.ckpt, a.device, a.positions * 2, 5)
    ok = ok[:a.positions]
    team, mask = env.team_state().astype(np.int16), env.legal_mask()
    print(f"{len(ok)} positions, device {a.device}, torch threads {torch.get_num_threads()}")
    for c in a.configs:
        kw = {"tau": 0.02, "infer": True, **json.loads(c)}
        p = ExactSearchPlayer(a.ckpt, a.device, **kw)
        ts, lv = [], []
        for rep, g in enumerate([ok[0]] + list(ok)):          # the first call warms up
            p.leaves = 0
            t0 = time.perf_counter()
            p._search(team[[g]], mask[[g]], 0, env, [int(g)])
            if rep:
                ts.append(time.perf_counter() - t0); lv.append(p.leaves)
        ts = np.array(ts)
        print(f"{c:60s} mean {ts.mean():6.3f}s  median {np.median(ts):6.3f}s  max {ts.max():6.3f}s  leaves {np.mean(lv):8.0f}", flush=True)


if __name__ == "__main__":
    main()
