"""Team tournament (PA-Agent's Gen 1 recipe): which teams should we bring? Candidate teams each play
--games battles against teams real players brought (Metamon parsed replays, agent/metamon_convert.py
shards), with the same policy on both sides; the candidates are ranked by score, with standard errors.

Candidates: our builder teams (data/builder_teams_v1) plus the --top most used human teams of the
--cand-shards (rated >= --min-rating or Smogon tournament games). Opponents: human teams drawn from
the disjoint --opp-shards, so a candidate is never scored against itself. Our side is never shown the
opponent's team; the bots we play bring teams we don't know either.

    uv run python -m agent.team_tournament agent/runs/v5d/snapshots/00400.pt \
        --games 128 --top 120 --out agent/runs/team_tourn.json [--write data/tourn_teams_v1 --keep 8]
"""
import argparse
import collections
import json
import re
import time
from pathlib import Path

import gen1py  # noqa: F401
import numpy as np
import torch

from agent.arena import Player
from agent.features import DEX
from agent.selfplay import Runner

ROOT = Path(__file__).resolve().parent.parent
MOVE_NAME = {m["id"]: m["name"] for m in DEX["moves"] if m}
to_id = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
SPECIES_BY_ID = {to_id(s["name"]): s["name"] for s in DEX["species"] if s}
MOVE_BY_ID = {to_id(m["name"]): m["name"] for m in DEX["moves"] if m}
MOVE_BY_ID.update({"highjumpkick": MOVE_BY_ID["hijumpkick"], "psychic": MOVE_BY_ID["psychicm"], "visegrip": MOVE_BY_ID["vicegrip"]})


def shard_range(spec, data):
    lo, hi = (int(x) for x in spec.split("-"))
    return [Path(data) / f"shard{i:03d}.npz" for i in range(lo, hi + 1)]


def human_teams(paths, min_rating, tour_too=True):
    """[(team as six 'SPECIES:MOVE,...' strings, sorted)] from each game's first decision, own side (0)."""
    out = []
    for p in paths:
        z = np.load(p)
        g, team = z["game"], z["team"]
        first = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        ok = (z["rating"] >= min_rating) | (z["tour"] if tour_too else False)
        for i in first:
            if not ok[g[i]]:
                continue
            t = team[i, 0]
            if (t[:, 0] == 0).any():
                continue
            mons = []
            for r in t:
                mv = [MOVE_NAME[int(m)] for m in r[1:5] if m]
                if not mv:
                    break
                mons.append(DEX["species"][r[0]]["name"] + ":" + ",".join(mv))
            if len(mons) == 6:
                out.append(tuple(mons))
    return out


def showdown_team(text):
    """A Showdown export (data/builder_teams_v1/*_team) -> six set strings, in team order."""
    mons, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            cur[1].append(MOVE_BY_ID[to_id(line[2:])])
        elif not line.startswith(("EVs", "IVs", "Level", "Ability", "Nature")):
            cur = (SPECIES_BY_ID[to_id(line.split("@")[0])], [])
            mons.append(cur)
    return tuple(f"{s}:{','.join(m)}" for s, m in mons)


def showdown_export(team):
    blocks = []
    for s in team:
        sp, mv = s.split(":")
        blocks.append("\n".join([sp.replace("_", " ").title(), "EVs: 252 HP / 252 Atk / 252 Def / 252 SpA / 252 SpD / 252 Spe"]
                                + [f"- {re.sub('_M$', '', m).replace('_', ' ').title()}" for m in mv.split(",")]))
    return "\n\n".join(blocks)


def play(player, cands, opps, games, seed, device):
    """Score of each candidate (games each, half on each side) against random opponent teams."""
    n = len(cands) * games
    rng = np.random.default_rng(seed)
    runner = Runner(n, seed, device)
    cand = np.arange(n) // games
    side = np.arange(n) % 2                        # the candidate's side
    for i in range(n):
        o = opps[rng.integers(len(opps))]
        a, b = (cands[cand[i]], o) if side[i] == 0 else (o, cands[cand[i]])
        runner.env.reset(i, list(a), list(b))
    done = np.zeros(n, bool)
    score = np.zeros(n)
    t0 = time.time()
    for turn in range(1000):
        team, battle, mask = runner.state()
        acts = np.full((n, 2), 11, np.int64)
        for s in (0, 1):
            i = np.flatnonzero(~done)
            if len(i):
                acts[i, s] = player.actions(team[i], battle[i], mask[i], s, runner.rng)
        runner.env.step(acts)
        res = np.asarray(runner.env.results())
        new = (res >= 0) & ~done
        score[new] = np.where(res[new] == 2, 0.5, (res[new] == side[new]).astype(float))
        done |= res >= 0
        if turn % 25 == 0:
            print(f"  turn {turn}: {done.sum()}/{n} finished, {time.time() - t0:.0f}s", flush=True)
        if done.all():
            break
    s = score.reshape(len(cands), games)
    return s.mean(1), s.std(1, ddof=1) / np.sqrt(games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data/metamon_gen1ou")
    ap.add_argument("--cand-shards", default="0-14")
    ap.add_argument("--opp-shards", default="15-19")
    ap.add_argument("--min-rating", type=int, default=1500)
    ap.add_argument("--top", type=int, default=120, help="most used human teams added as candidates")
    ap.add_argument("--builder", default="data/builder_teams_v1")
    ap.add_argument("--cands", help="a previous --out file: re-score its best --recheck candidates instead")
    ap.add_argument("--recheck", type=int, default=24)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", action="store_true", help="policies draw from their mix instead of the most likely action")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", help="directory to write the best --keep teams to, as Showdown exports")
    ap.add_argument("--keep", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = "cpu"
    opps = human_teams(shard_range(args.opp_shards, args.data), args.min_rating)
    print(f"{len(opps)} opponent teams from shards {args.opp_shards}", flush=True)
    if args.cands:
        prev = json.load(open(args.cands))["ranking"]
        cands = [tuple(r["team"]) for r in prev[:args.recheck]]
        names = [r["name"] for r in prev[:args.recheck]]
    else:
        human = collections.Counter(tuple(sorted(t)) for t in human_teams(shard_range(args.cand_shards, args.data), args.min_rating))
        built = [showdown_team(p.read_text()) for p in sorted(Path(args.builder).rglob("*_team"))]
        cands = built + [t for t, _ in human.most_common(args.top)]
        names = [f"builder{i:02d}" for i in range(len(built))] + [f"human{i:03d}x{c}" for i, (_, c) in enumerate(human.most_common(args.top))]
    print(f"{len(cands)} candidates x {args.games} games", flush=True)
    player = Player(args.ckpt, device, args.sample)
    mean, se = play(player, cands, opps, args.games, args.seed, device)
    order = np.argsort(-mean)
    ranking = [{"name": names[k], "score": round(float(mean[k]), 4), "se": round(float(se[k]), 4), "team": list(cands[k])} for k in order]
    for r in ranking[:30]:
        print(f"  {r['score']:.3f} +- {r['se']:.3f}  {r['name']:14s} " + " / ".join(s.split(':')[0] for s in r["team"]))
    b = [r for r in ranking if r["name"].startswith("builder")]
    print(f"builder teams mean {np.mean([r['score'] for r in b]):.3f}; all candidates mean {mean.mean():.3f}")
    json.dump({"args": vars(args), "ranking": ranking}, open(args.out, "w"), indent=1)
    if args.write:
        d = Path(args.write)
        d.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(ranking[:args.keep]):
            (d / f"{i:02d}_team").write_text(showdown_export(r["team"]))


if __name__ == "__main__":
    main()
