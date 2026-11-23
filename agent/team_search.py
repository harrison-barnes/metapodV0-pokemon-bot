"""Evolutionary team search: which six ladder sets, in which order, win the most?

A candidate is six sets from the pool (data/sets_ladder.json, the sets the matchup table has solved,
so the network sees full features for every one of them), distinct species, slot 0 the lead. Each
generation every candidate plays --games battles in our engine, the same policy on both sides, half
on each side, against opponents drawn from
- real players' teams (agent/team_tournament.human_teams on --opp-shards: rated >= 1500 or tournament
  games, in their own team order), and
- with probability --hof-frac, the hall of fame: the best candidates of earlier generations, so a
  team that only beats the average human team but folds to a strong one doesn't keep winning
  (a team-level PSRO step).
A candidate's games accumulate across generations while it survives. Survivors are the top --keep by
mean score less one standard error; children are mutations of survivors (another set of the same
species, another species' set, or a different lead), --mutations of them each generation.

Neither the bots we test against nor their teams are used: the final teams meet them only in a
held-out gauntlet.

    uv run python -m agent.team_search agent/runs/v7/snapshots/00300.pt \
        --gens 30 --pop 64 --games 256 --out agent/runs/team_search_r1.json --write data/search_teams_v1
"""
import argparse
import json
import time
from pathlib import Path

import gen1py  # noqa: F401
import numpy as np
import torch

from agent.arena import load_policy
from agent.matchups import default_device
from agent.selfplay import Runner
from agent.team_tournament import human_teams, shard_range, showdown_export, showdown_team
from agent.teams import TeamGen

ROOT = Path(__file__).resolve().parent.parent
POOL = json.load(open(ROOT / "data/sets_ladder.json"))
N_LADDER = len(POOL)                              # sets past this index are new ones (--novel), unknown to the matchup table
SET_STR = [f"{s['species']}:{','.join(s['moves'])}" for s in POOL]
SPECIES = [s["species"] for s in POOL]
BY_SPECIES = {}
for i, s in enumerate(SPECIES):
    BY_SPECIES.setdefault(s, []).append(i)
SET_INDEX = {(SPECIES[i], frozenset(POOL[i]["moves"])): i for i in range(len(POOL))}


# Left out of --wide: moves that do nothing in a Gen 1 link battle; moves Gen 1 OU bans (OHKO and Evasion
# clauses); and Rage and Bide, where our engine still differs a little from Showdown (docs/ENGINE.md), so a
# search could find "strength" that's only an engine gap.
NO_EFFECT = {"SPLASH", "TELEPORT", "WHIRLWIND", "ROAR", "FISSURE", "HORN_DRILL", "GUILLOTINE", "DOUBLE_TEAM",
             "MINIMIZE", "RAGE", "BIDE"}


def _legal_moves(wide=False):
    """Per species: its legal moves (learnset, level-up and pre-evolution moves) that some ladder set
    uses, i.e. moves known to be competitive in some role; with wide, every legal move but NO_EFFECT.
    --novel sets are drawn from these."""
    dex = json.load(open(ROOT / "data/gen1.json"))
    name = {m["id"]: m["name"] for m in dex["moves"] if m}
    by_name = {s["name"]: s for s in dex["species"] if s}
    universe = {m for p in POOL for m in p["moves"]}
    out = {}
    for sp in BY_SPECIES:
        todo, seen, moves = [sp], set(), set()
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            e = by_name.get(x) if isinstance(x, str) else dex["species"][x]
            if not e:
                continue
            moves |= {name[i] for i in e["learnset"]} | {m for _, m in e["levelup"]}
            todo += list(e["prevos"])
        out[sp] = sorted(moves - NO_EFFECT) if wide else sorted(moves & universe)
    return out


LEGAL = _legal_moves()


def register(species, moves):
    """Index of this set, adding it to the pool if new."""
    key = (species, frozenset(moves))
    if key not in SET_INDEX:
        SET_INDEX[key] = len(SET_STR)
        SET_STR.append(f"{species}:{','.join(moves)}")
        SPECIES.append(species)
        BY_SPECIES.setdefault(species, []).append(SET_INDEX[key])
    return SET_INDEX[key]


def to_strings(team):
    return [SET_STR[i] for i in team]


def from_strings(sets, novel=False):
    """Six set strings -> pool indices; None if a set isn't in the pool (unless novel: then it's added)."""
    out = []
    for s in sets:
        sp, mv = s.split(":")
        mv = mv.split(",")
        i = SET_INDEX.get((sp, frozenset(mv)))
        if i is None:
            if not novel or len(set(mv)) != 4 or sp not in LEGAL:
                return None
            i = register(sp, mv)
        out.append(i)
    return tuple(out)


def mutate(team, rng, novel=0.0):
    """A different lead, another set of a member's species, another species, or (with probability
    `novel`) one move of a member swapped for another of its legal competitive moves."""
    t = list(team)
    r = rng.random()
    if r < 0.15:                                   # a different lead
        j = int(rng.integers(1, 6))
        t[0], t[j] = t[j], t[0]
        return tuple(t)
    k = int(rng.integers(6))
    sp = SPECIES[t[k]]
    if r < 0.15 + novel:                           # one move swapped
        cur = SET_STR[t[k]].split(":")[1].split(",")
        alt = [m for m in LEGAL.get(sp, []) if m not in cur]
        if alt:
            cur[int(rng.integers(4))] = str(rng.choice(alt))
            t[k] = register(sp, cur)
            return tuple(t)
    if rng.random() < 0.35 and len(BY_SPECIES[sp]) > 1:     # another set of the same species
        t[k] = int(rng.choice([i for i in BY_SPECIES[sp] if i != t[k]]))
        return tuple(t)
    have = {SPECIES[i] for j, i in enumerate(t) if j != k}  # another species (Species Clause), a ladder set
    t[k] = int(rng.choice([i for i in range(N_LADDER) if SPECIES[i] not in have]))
    return tuple(t)


@torch.no_grad()
def play(policy, pairs, seed, device, slots=4096, max_turns=500):
    """Score of the first team of each (ours, theirs) pair, ours on side i % 2 of game i. Games stream
    through `slots` engine battles: a finished slot takes the next pair at once, so a long stall holds
    up one slot rather than a whole batch. A game past max_turns counts as a draw."""
    n = len(pairs)
    w = min(slots, n)
    runner = Runner(1, seed, device)
    runner.n, runner.env = w, gen1py.VecEnv(w, seed=seed)
    side = np.arange(n) % 2
    score = np.full(n, 0.5)
    game = np.full(w, -1)                          # the pair each slot plays; -1 idle
    turns = np.zeros(w, int)
    nxt = 0

    def start(k):
        nonlocal nxt
        if nxt < n:
            a, b = pairs[nxt]
            runner.env.reset(int(k), *((a, b) if side[nxt] == 0 else (b, a)))
            game[k], turns[k] = nxt, 0
            nxt += 1
        else:
            game[k] = -1

    for k in range(w):
        start(k)
    while (game >= 0).any():
        team, battle, mask = runner.state()
        acts, _, _ = runner.act(policy, team, battle, mask, greedy=True)   # idle slots: whatever the policy picks
        runner.env.step(acts.astype(np.int64))                             # (a battle cut off at max_turns is still live)
        turns += 1
        res = np.asarray(runner.env.results())
        for k in np.flatnonzero((game >= 0) & ((res >= 0) | (turns >= max_turns))):
            g = game[k]
            if res[k] >= 0:
                score[g] = 0.5 if res[k] == 2 else float(res[k] == side[g])
            start(k)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data/metamon_gen1ou")
    ap.add_argument("--opp-shards", default="15-19")
    ap.add_argument("--seed-shards", default="0-14", help="human teams (made of pool sets) that seed the population")
    ap.add_argument("--min-rating", type=int, default=1500)
    ap.add_argument("--teams-json", help="precomputed {'opp': [...], 'seed': [...]} human teams (--dump-teams), instead of the shards")
    ap.add_argument("--dump-teams", help="write the human teams from the shards to this file and exit")
    ap.add_argument("--builder", default="data/builder_teams_v1")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--keep", type=int, default=24)
    ap.add_argument("--games", type=int, default=256, help="battles per candidate per generation")
    ap.add_argument("--hof-frac", type=float, default=0.3)
    ap.add_argument("--opp-mix", help="an agent/team_nash.py result: with probability --opp-mix-frac the opponent brings a team drawn "
                    "from its Nash mixture (a best-response search, the PSRO step)")
    ap.add_argument("--opp-mix-frac", type=float, default=0.5)
    ap.add_argument("--hof-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8192, help="battles played at once (finished ones are refilled)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume", help="a previous --out file to continue from")
    ap.add_argument("--wide", action="store_true", help="--novel sets may use any legal move, not only moves some ladder set uses")
    ap.add_argument("--novel", type=float, default=0.0, help="share of mutations that swap one move for another legal one (new sets)")
    ap.add_argument("--seed-teams", nargs="*", default=[], help="Showdown team files added to the starting population")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", help="directory for the best --write-keep teams, as Showdown exports")
    ap.add_argument("--write-keep", type=int, default=8)
    args = ap.parse_args()
    if args.wide:
        LEGAL.clear()
        LEGAL.update(_legal_moves(wide=True))
    torch.set_num_threads(args.threads)
    device = default_device()
    rng = np.random.default_rng(args.seed)
    policy = load_policy(args.ckpt, device)
    if args.teams_json:
        tj = json.load(open(args.teams_json))
        opps, seed_human = [tuple(t) for t in tj["opp"]], [tuple(t) for t in tj["seed"]]
    else:
        opps = human_teams(shard_range(args.opp_shards, args.data), args.min_rating)
        seed_human = human_teams(shard_range(args.seed_shards, args.data), args.min_rating)
    if args.dump_teams:
        json.dump({"opp": opps, "seed": seed_human}, open(args.dump_teams, "w"))
        return
    print(f"{len(opps)} opponent teams; pool of {len(POOL)} sets over {len(BY_SPECIES)} species", flush=True)

    mix_sets, mix_w = [], None
    if args.opp_mix:
        rows = [r for r in json.load(open(args.opp_mix))["population"] if r["weight"] > 0]
        mix_sets = [r["sets"] for r in rows]
        mix_w = np.array([r["weight"] for r in rows]) / sum(r["weight"] for r in rows)
        print(f"opponents: Nash mixture of {len(rows)} teams with probability {args.opp_mix_frac}", flush=True)
    stats = {}                                    # team -> [score sum, games]
    hof = []
    if args.resume:
        prev = json.load(open(args.resume))
        for r in prev["all"]:
            t = from_strings(r["sets"], novel=True)
            stats[t] = [r["score"] * r["games"], r["games"]]
        hof = [from_strings(h, novel=True) for h in prev.get("hof_sets", [])] or [tuple(t) for t in prev["hof"]]
        pop = [from_strings(r["sets"], novel=True) for r in prev["all"][:args.keep]]
        pop += [t for t in (from_strings(showdown_team(Path(f).read_text()), novel=True) for f in args.seed_teams) if t and t not in pop]
    else:
        seeds = [from_strings(showdown_team(p.read_text())) for p in sorted(Path(args.builder).rglob("*_team"))]
        human = [from_strings(t) for t in seed_human]
        human = [t for t in dict.fromkeys(human) if t is not None]
        seeds = [t for t in seeds if t is not None] + [human[i] for i in rng.permutation(len(human))[:args.pop // 2]]
        gen = TeamGen()
        while len(seeds) < args.pop:
            t = from_strings(gen(rng))
            if t is not None:
                seeds.append(t)
        pop = list(dict.fromkeys(seeds))[:args.pop]
        print(f"seeded with {sum(1 for t in pop if t in seeds[:8])} builder teams, {len(human)} pool-set human teams available", flush=True)

    def lcb(t):
        s, g = stats[t]
        m = s / g
        return m - np.sqrt(max(m * (1 - m), 0.05) / g)

    for gen_i in range(args.gens):
        t0 = time.time()
        pairs, owner = [], []
        for ci, t in enumerate(pop):
            for _ in range(args.games):
                r = rng.random()
                if mix_sets and r < args.opp_mix_frac:
                    o = list(mix_sets[rng.choice(len(mix_sets), p=mix_w)])
                elif hof and r < args.opp_mix_frac * bool(mix_sets) + args.hof_frac:
                    o = to_strings(hof[rng.integers(len(hof))])
                else:
                    o = list(opps[rng.integers(len(opps))])
                pairs.append((to_strings(t), o))
                owner.append(ci)
        owner = np.array(owner)
        score = play(policy, pairs, int(rng.integers(1 << 31)), device, slots=args.batch)
        for ci, t in enumerate(pop):
            s = stats.setdefault(t, [0.0, 0])
            s[0] += float(score[owner == ci].sum())
            s[1] += int((owner == ci).sum())
        ranked = sorted(pop, key=lcb, reverse=True)
        hof = list(dict.fromkeys(ranked[:4] + hof))[:args.hof_size]
        survivors = ranked[:args.keep]
        children = set()
        while len(children) < args.pop - len(survivors):
            c = mutate(survivors[int(rng.integers(len(survivors)))], rng, args.novel)
            if c not in stats and c not in survivors:
                children.add(c)
        pop = survivors + sorted(children)
        best = ranked[0]
        print(f"gen {gen_i}: {len(pairs)} battles in {time.time() - t0:.0f}s; best {stats[best][0] / stats[best][1]:.3f} "
              f"over {stats[best][1]} ({' / '.join(SPECIES[i] for i in best)}); pop mean {score.mean():.3f}", flush=True)
        allr = sorted(stats, key=lambda t: (stats[t][1] >= args.games * 2, lcb(t)), reverse=True)
        json.dump({"args": vars(args), "hof": [list(t) for t in hof], "hof_sets": [to_strings(t) for t in hof],
                   "all": [{"team": list(t), "score": round(stats[t][0] / stats[t][1], 4), "games": stats[t][1],
                            "lcb": round(lcb(t), 4), "sets": to_strings(t),
                            "new_sets": sum(i >= N_LADDER for i in t)} for t in allr[:200]]},
                  open(args.out, "w"), indent=1)
    if args.write:
        d = Path(args.write)
        d.mkdir(parents=True, exist_ok=True)
        top = json.load(open(args.out))["all"][:args.write_keep]
        for i, r in enumerate(top):
            (d / f"{i:02d}_team").write_text(showdown_export(r["sets"]))


if __name__ == "__main__":
    main()
