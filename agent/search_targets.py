"""Search targets for policy distillation inside PPO (docs/RESEARCH_GAME_AI.md, plan step 6).

Expert iteration failed here (xit_a, 2026-11-02) because the critic was retrained on search self-play:
it memorised the few thousand games and took on the search's optimism. This keeps only the part that
works: the *policy* is pulled toward the exact search's mix (agent/search_exact.py) on a small random
share of positions, while the critic keeps learning from PPO returns only. The search's root value is
never a training target.

A worker process plays policy-vs-policy games in our engine (both sides sample the current policy, so
the positions are the policy's own) and, with probability --distill-frac per decision with a real
choice, runs the search on that position and records (team, battle bytes, legal mask, side, search
mix). The move actually played is the policy's, not the search's (KataGo's playout-cap randomisation:
cheap moves keep the state distribution, expensive ones make the targets). Batches go to the learner,
which adds distill_coef * cross-entropy(search mix, policy) to each minibatch (agent/train_v4.update).

The search uses the checkpoint pair the learner refreshes every --distill-refresh iterations
(RUN/search.pt, RUN/search_critic.pt), so its targets improve with the policy and critic.

Smoke test (a few searches, then prints the rate):
    uv run python -m agent.search_targets agent/runs/v5b/snapshots/00400.pt --games 16 --steps 40
"""
import argparse
import os
import queue
import time

import gen1py  # noqa: F401
import numpy as np
import torch

FIELDS = ("team", "battle", "mask", "side", "pi")


def worker(rank, ckpt, device, games, frac, batch, seed, teams_kind, out, stop, search_kw=None, steps=None, gpu_frac=None):
    """Plays `games` battles forever (or `steps` turns), sending dicts of FIELDS arrays (`batch` rows) to `out`.
    gpu_frac caps this process's share of a CUDA device (the learner shares it)."""
    torch.set_num_threads(2)
    if gpu_frac and str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(gpu_frac)
    from agent.search_exact import ExactSearchPlayer
    from agent.selfplay import Runner
    teams = None
    if teams_kind == "smogon":
        from agent.teams import TeamGen
        teams = TeamGen()
    rng = np.random.default_rng(seed + 104729 * (rank + 1))
    runner = Runner(games, int(rng.integers(1 << 31)), device, teams)
    kw = {"k": 8, "width": 6, "tau": 0.02, "seed": int(rng.integers(1 << 31)), **(search_kw or {})}
    searcher, loaded = None, 0.0
    rows = {k: [] for k in FIELDS}
    t = 0
    while not stop.is_set() and (steps is None or t < steps):
        if os.path.getmtime(ckpt) > loaded:       # the learner rewrote the pair: search with the new one
            loaded = os.path.getmtime(ckpt)
            time.sleep(2)                         # let the critic file land too
            searcher = ExactSearchPlayer(ckpt, device, **kw)
            searcher.policy.eval(); searcher.critic.eval()
        team, battle, mask = runner.state()
        a, _, _ = runner.act(searcher.policy, team, battle, mask)
        live = np.flatnonzero(runner.env.results() < 0)
        searched = set()
        for s in (0, 1):     # one side per game per turn: each _search call advances that game's counts
            pick = live[(mask[live, s].sum(-1) > 1) & (rng.random(len(live)) < frac) & ~np.isin(live, list(searched))]
            if len(pick) == 0:
                continue
            searcher.record = []
            searcher._search(team[pick], mask[pick], s, runner.env, pick)
            for g, side, tm, bb, m, pi, _ in searcher.record:     # the root value is dropped on purpose
                for k, v in zip(FIELDS, (tm, bb, m, side, pi)):
                    rows[k].append(v)
            searcher.record = None
            searched.update(pick.tolist())
        # Keep the asleep/confused turn counts exact for the games not searched this turn (_search
        # counted the others), so later searches draw the hidden counters from the right durations.
        rest = np.array([g for g in live if g not in searched], np.int64)
        if len(rest):
            searcher._count_seen(team[rest], battle[rest], rest)
        runner.env.step(a.astype(np.int64))
        done = runner.env.results() >= 0
        for g in np.flatnonzero(done):
            searcher.seen.pop(int(g), None)
        runner.finish(done)
        t += 1
        while len(rows["pi"]) >= batch and not stop.is_set():
            chunk = {k: np.stack(v[:batch]) for k, v in rows.items()}
            chunk["version"] = loaded
            try:
                out.put(chunk, timeout=5)
            except queue.Full:
                continue
            rows = {k: v[batch:] for k, v in rows.items()}


class DistillBuffer:
    """The latest `size` search targets; sample(n) draws uniformly (each target is reused a few times)."""

    def __init__(self, size, seed=0):
        self.size, self.data, self.rng, self.total = size, None, np.random.default_rng(seed), 0

    def add(self, chunk):
        new = {k: chunk[k] for k in FIELDS}
        self.data = new if self.data is None else {k: np.concatenate([self.data[k], new[k]])[-self.size:] for k in FIELDS}
        self.total += len(new["pi"])

    def __len__(self):
        return 0 if self.data is None else len(self.data["pi"])

    def sample(self, n):
        i = self.rng.integers(len(self), size=min(n, len(self)))
        return {k: v[i] for k, v in self.data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.1)
    args = ap.parse_args()
    from agent.matchups import default_device
    import multiprocessing as mp
    q, stop = queue.Queue(), mp.Event()
    t0 = time.time()
    worker(0, args.ckpt, args.device or default_device(), args.games, args.frac, 1, 0, None, q, stop, steps=args.steps)
    n = q.qsize()
    rows = [q.get() for _ in range(n)]
    dt = time.time() - t0
    print(f"{n} search targets in {dt:.0f}s ({n / dt:.1f}/s incl. startup)")
    if rows:
        pi = np.concatenate([r["pi"] for r in rows])
        print("mean max prob", pi.max(-1).mean().round(3), "sums", pi.sum(-1).min().round(3), pi.sum(-1).max().round(3),
              "legal ok", bool(all(((r["pi"] > 0) <= r["mask"]).all() for r in rows)))


if __name__ == "__main__":
    main()
