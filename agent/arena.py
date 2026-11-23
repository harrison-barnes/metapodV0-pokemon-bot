"""Head-to-head matches between agents and bots, and Elo ratings for a set of them.

A player is "random", "greedy" or "smart" (agent/bots.py), "search:CKPT" (agent/search.py: one-turn
lookahead with that checkpoint's value head; needs the gen1py bindings), "q:CKPT"
(agent/qsolve.py: the equilibrium of a v4 checkpoint's payoff-matrix head, --search-k guesses
of the hidden sets; "qsafe:CKPT" adds safe exploitation of the opponent), or the path of a policy checkpoint
(played greedily, the most likely action, or with --sample by drawing from its policy). Each pairing plays GAMES battles with random ladder
teams, each player taking side 0 in half of them; a draw counts half. Ratings are a
Bradley-Terry fit to every pairing's result, scaled to Elo with the greedy bot at 1000 (or
the first player, if greedy isn't playing).

Usage: uv run python -m agent.arena PLAYER PLAYER [...] [--games 512] [--out ratings.json]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from agent.bots import greedy_actions, random_actions, smart_actions
from agent.matchups import Matchups, default_device, table_path
from agent.policy import Policy
from agent.selfplay import Runner, observe_on

BOTS = {"random": lambda mask, team, side, rng: random_actions(mask, side, rng),
        "greedy": greedy_actions, "smart": smart_actions}


_TABLES = {}


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    arch = ck.get("arch") or {"d": ck["width"], "matchups": bool(ck.get("matchups"))}
    policy = Policy(**arch).to(device)
    policy.load_state_dict(ck["policy"])
    if policy.matchups:
        path = table_path(ck["matchups"])
        key = (str(path), str(device))
        policy.mu = _TABLES[key] = _TABLES.get(key) or Matchups(device, path)   # one copy per table
    if getattr(policy, "senses", False):
        from agent.senses import Senses
        key = ("senses", str(path), str(device))
        policy.sense = _TABLES[key] = _TABLES.get(key) or Senses(policy.mu, path)
    return policy.eval()


class Player:
    def __init__(self, spec, device, sample=False):
        self.name = spec if spec in BOTS else str(Path(spec).with_suffix("")).replace("agent/runs/", "")
        self.bot = BOTS.get(spec)
        self.policy = None if self.bot else load_policy(spec, device)
        self.device = device
        self.sample = sample

    @torch.no_grad()
    def actions(self, team, battle, mask, side, rng, env=None, idx=None):
        if self.bot:
            return self.bot(mask, team, side, rng)
        m = mask[:, side].copy()
        m[~m.any(-1), 11] = True
        logits, _ = self.policy(observe_on(self.policy, team, battle, side, self.device), torch.from_numpy(m).to(self.device))
        if self.sample:
            return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        return logits.argmax(-1).cpu().numpy()


def match(p0, p1, games, seed, device, progress=False):
    """p0's score against p1 (draws half), over `games` battles split between the two sides; with
    progress, the running score each time another tenth of the games has finished."""
    runner = Runner(games, seed, device)
    for p in (p0, p1):
        if hasattr(p, "reset"):
            p.reset()
    p0_side = np.arange(games) % 2
    done = np.zeros(games, bool)
    score = np.zeros(games)
    for _ in range(1000):
        team, battle, mask = runner.state()
        acts = np.full((games, 2), 11, np.int64)
        for s in (0, 1):
            for p, mine in ((p0, (p0_side == s) & ~done), (p1, (p0_side != s) & ~done)):
                i = np.flatnonzero(mine)
                if len(i):
                    acts[i, s] = p.actions(team[i], battle[i], mask[i], s, runner.rng, env=runner.env, idx=i)
        runner.env.step(acts.astype(np.int64))
        for s in (0, 1):                     # players that model their opponent see its choices
            for p, mine in ((p0, (p0_side == s) & ~done), (p1, (p0_side != s) & ~done)):
                i = np.flatnonzero(mine)
                if len(i) and hasattr(p, "observe"):
                    p.observe(acts[i, 1 - s], env=runner.env, idx=i)
        res = runner.env.results()
        new = (res >= 0) & ~done
        score[new] = np.where(res[new] == 2, 0.5, (res[new] == p0_side[new]).astype(float))
        before = done.sum()
        done |= res >= 0
        if progress and done.sum() // max(1, games // 10) > before // max(1, games // 10):
            print(f"  progress: {getattr(p0, 'name', 'p0')} vs {getattr(p1, 'name', 'p1')}: {score[done].mean():.3f} "
                  f"after {done.sum()}/{games} games", flush=True)
        if done.all():
            break
    return float(score[done].mean()), int(done.sum())


def elo(names, results, anchor):
    """Bradley-Terry ratings (Elo scale) from {(i, j): (score of i, games)}, by gradient ascent.
    Each pairing gets one extra drawn game, so a 100% score gives a large but finite gap."""
    results = {k: ((s * n + 0.5) / (n + 1), n + 1) for k, (s, n) in results.items()}
    r = np.zeros(len(names))
    for _ in range(5000):
        g = np.zeros_like(r)
        for (i, j), (s, n) in results.items():
            p = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
            g[i] += n * (s - p)
            g[j] -= n * (s - p)
        r += 2.0 * g / max(n for _, n in results.values())
    return r - r[anchor] + 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("players", nargs="+")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out")
    ap.add_argument("--search-k", type=int, default=6, help="hidden-information guesses per search decision")
    ap.add_argument("--search-width", type=int, default=6, help="candidate actions per side in xs search")
    ap.add_argument("--search-depth", type=int, default=1, help="turns searched by xs (later turns narrower)")
    ap.add_argument("--search-width2", type=int, default=3, help="candidate actions per side after the first turn")
    ap.add_argument("--search-critic", help="critic checkpoint xs scores positions with (default: POLICY_critic.pt)")
    ap.add_argument("--search-rollouts", type=int, default=0, help="xs scores each leaf by this many policy playouts")
    ap.add_argument("--search-rollout-depth", type=int, default=2, help="turns per xs leaf playout")
    ap.add_argument("--search-opp-model", help="checkpoint xs models the opponent with (default: the searched policy)")
    ap.add_argument("--search-opp-mix", type=float, default=1.0, help="with --search-opp-model: its share of the opponent prior, the rest the searched policy's")
    ap.add_argument("--search-oracle", action="store_true", help="xs sees the opponent's true team (diagnostic)")
    ap.add_argument("--search-infer", action="store_true", help="xs weights its guesses by the opponent's past choices")
    ap.add_argument("--sample", action="store_true", help="agents draw actions from their policy instead of taking the most likely one")
    args = ap.parse_args()
    device = default_device()
    def make(spec):
        if spec.startswith("xs"):                 # xs:POLICY or xs0.1:POLICY (tau); critic = POLICY_critic.pt
            from agent.search_exact import ExactSearchPlayer
            mode, path = spec.split(":", 1)
            return ExactSearchPlayer(path, device, critic=args.search_critic, k=args.search_k, tau=float(mode[2:] or 0.05), width=args.search_width,
                                     depth=args.search_depth, width2=args.search_width2, infer=args.search_infer, oracle=args.search_oracle,
                                     rollouts=args.search_rollouts, rollout_depth=args.search_rollout_depth,
                                     opp_model=args.search_opp_model, opp_mix=args.search_opp_mix)
        if spec.startswith("search:"):
            from agent.search import SearchPlayer
            return SearchPlayer(spec[len("search:"):], device, k=args.search_k)
        if spec.startswith(("q:", "qsafe:", "qreg")):
            from agent.qsolve import QPlayer
            mode, path = spec.split(":", 1)
            if mode.startswith("qreg"):            # qreg:CRITIC or qreg0.1:CRITIC (tau)
                return QPlayer(path, device, k=args.search_k, mode="reg", tau=float(mode[4:] or 0.05))
            return QPlayer(path, device, k=args.search_k, mode="safe" if mode == "qsafe" else "nash")
        return Player(spec, device, args.sample)
    players = [make(p) for p in args.players]
    results = {}
    for (i, a), (j, b) in itertools.combinations(enumerate(players), 2):
        s, n = match(a, b, args.games, args.seed + 7 * i + j, device, progress=True)
        results[(i, j)] = (s, n)
        print(f"{a.name:>28} vs {b.name:<28} {s:.3f} over {n} games", flush=True)
    names = [p.name for p in players]
    anchor = names.index("greedy") if "greedy" in names else 0
    ratings = elo(names, results, anchor)
    print()
    for k in np.argsort(-ratings):
        print(f"{ratings[k]:7.0f}  {names[k]}")
    if args.out:
        json.dump({"ratings": dict(zip(names, ratings.round(1).tolist())),
                   "matches": [{"a": names[i], "b": names[j], "score": s, "games": n} for (i, j), (s, n) in results.items()]},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
