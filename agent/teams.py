"""Realistic training teams from Smogon's aggregate ladder statistics (gen1ou, 1760+ cutoff, 2026-09).

agent/selfplay.sample_team draws six sets independently by usage, which gives teams no real player
would bring (two thirds of them carry two or more sleepers; the lead is whatever came first). Here a
team is built the way the statistics describe one:

1. The lead (slot 0, the Pokemon sent out first) is drawn from Smogon's leads file.
2. The other five species are drawn one at a time, each by base weight x the product of its teammate
   lifts with the species already chosen (agent/search.teammate_lift: P(B on team | A) / P(B)),
   raised to --alpha so that five compounding lifts don't over-concentrate the teams. The base
   weights are fitted (iterative proportional fitting, `fit`) so that each species' share of the
   generated teams matches its Smogon usage.
3. Each species' set is drawn from the pool's sets for it (data/sets_ladder.json, ladder weights),
   with sets that carry a sleep move down-weighted by SLEEP_PENALTY once the team has a sleeper:
   Sleep Clause makes a second sleeper mostly redundant, and real teams rarely carry one.

Species outside the pool (or banned) are never drawn; species in the pool but missing from the
statistics keep a small floor weight so they still appear now and then.

    uv run python -m agent.teams   # fits the weights, prints the check against Smogon's numbers
"""
import json
import re
from pathlib import Path

import numpy as np

from agent.search import DEX, POOL, SPECIES_ID, USAGE, teammate_lift
from agent.selfplay import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
LEADS = ROOT / "data/smogon/gen1ou-1760-2026-09-leads.txt"
FITTED = ROOT / "data/smogon/team_weights.json"
SLEEP_MOVES = {"SLEEP_POWDER", "LOVELY_KISS", "SING", "HYPNOSIS", "SPORE"}
SLEEP_PENALTY = 0.15
FLOOR = 0.002
norm = lambda name: re.sub(r"[^a-z0-9]", "", name.lower())


class TeamGen:
    def __init__(self, alpha=0.5, weights=FITTED):
        self.alpha = alpha
        self.species = sorted({p["species"] for p, w in zip(POOL, WEIGHTS) if w > 0})
        self.sid = np.array([SPECIES_ID[s] for s in self.species])
        idx = {s: i for i, s in enumerate(self.species)}
        self.sets = [[] for _ in self.species]                 # per species: (set string, weight, sleeper)
        for p, w in zip(POOL, WEIGHTS):
            if w > 0:
                self.sets[idx[p["species"]]].append((p["species"] + ":" + ",".join(p["moves"]), w, bool(SLEEP_MOVES & set(p["moves"]))))
        by_norm = {norm(s): i for i, s in enumerate(self.species)}
        data = json.load(open(USAGE))["data"]
        self.usage = np.full(len(self.species), FLOOR)
        for name, v in data.items():
            if norm(name) in by_norm:
                self.usage[by_norm[norm(name)]] = max(v["usage"], FLOOR)
        self.lead = np.full(len(self.species), FLOOR * 0.1)
        for line in open(LEADS):
            m = re.match(r"\|\s*\d+\s*\|\s*(.+?)\s*\|\s*([\d.]+)%", line)
            if m and norm(m.group(1)) in by_norm:
                self.lead[by_norm[norm(m.group(1))]] = float(m.group(2)) / 100
        self.lead /= self.lead.sum()
        self.lift = teammate_lift()[np.ix_(self.sid, self.sid)] ** alpha
        self.base = self.usage.copy()
        if weights and Path(weights).exists():
            saved = json.load(open(weights))
            if saved.get("alpha") == alpha:
                self.base = np.array([saved["base"].get(s, u) for s, u in zip(self.species, self.usage)])

    def species_team(self, rng):
        """Six distinct species indices, the lead first."""
        team = [int(rng.choice(len(self.species), p=self.lead))]
        for _ in range(5):
            p = self.base * np.prod(self.lift[team], axis=0)
            p[team] = 0
            team.append(int(rng.choice(len(p), p=p / p.sum())))
        return team

    def __call__(self, rng):
        """A team as the engine takes it: six "SPECIES:MOVE,MOVE,MOVE,MOVE" strings, the lead first."""
        out, sleeper = [], False
        for s in self.species_team(rng):
            w = np.array([wt * (SLEEP_PENALTY if sleeper and sl else 1.0) for _, wt, sl in self.sets[s]])
            k = int(rng.choice(len(w), p=w / w.sum()))
            out.append(self.sets[s][k][0])
            sleeper |= self.sets[s][k][2]
        return out

    def fit(self, rounds=12, teams=20000, seed=0):
        """Base weights such that each species' share of generated teams matches its Smogon usage."""
        rng = np.random.default_rng(seed)
        target = np.minimum(self.usage, 1.0)
        for r in range(rounds):
            share = self.shares(rng, teams)
            self.base *= np.clip((target + 1e-4) / (share + 1e-4), 0.5, 2.0)
        return self.shares(rng, teams), target

    def shares(self, rng, teams):
        c = np.zeros(len(self.species))
        for _ in range(teams):
            c[self.species_team(rng)] += 1
        return c / teams

    def save(self, path=FITTED):
        json.dump({"alpha": self.alpha, "base": dict(zip(self.species, self.base.round(6).tolist()))}, open(path, "w"), indent=0)


def main():
    import argparse
    import collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--teams", type=int, default=20000)
    args = ap.parse_args()
    g = TeamGen(args.alpha, weights=None)
    share, target = g.fit(teams=args.teams)
    g.save()
    rng = np.random.default_rng(1)
    T = [g(rng) for _ in range(args.teams)]
    order = np.argsort(-target)[:15]
    print("species share of teams, generated vs Smogon usage:")
    for i in order:
        print(f"  {g.species[i]:<11} {share[i]:.3f}  {target[i]:.3f}")
    print(f"  mean |error| over all species: {np.abs(share - target).mean():.4f}")
    # pair co-occurrence vs Smogon's teammate statistics, for the top pairs
    data = json.load(open(USAGE))["data"]
    w = {norm(k): sum(v["Abilities"].values()) for k, v in data.items()}
    n_teams = sum(w.values()) / 6
    names = [{m.split(":")[0] for m in t} for t in T]
    print("pairs, generated vs Smogon (share of all teams):")
    for a, b in [("TAUROS", "SNORLAX"), ("TAUROS", "CHANSEY"), ("CHANSEY", "SNORLAX"), ("EXEGGUTOR", "TAUROS"),
                 ("JYNX", "EXEGGUTOR"), ("GENGAR", "ALAKAZAM"), ("STARMIE", "ALAKAZAM"), ("ZAPDOS", "RHYDON")]:
        real = next((v["Teammates"].get(bn, 0) for k, v in data.items() if norm(k) == norm(a)
                     for bn in v["Teammates"] if norm(bn) == norm(b)), 0) / n_teams
        gen = np.mean([a in t and b in t for t in names])
        print(f"  {a}+{b:<10} {gen:.3f}  {real:.3f}")
    sleepers = collections.Counter(sum(bool(SLEEP_MOVES & set(m.split(":")[1].split(","))) for m in t) for t in T)
    print("sleepers per team:", {k: round(v / len(T), 3) for k, v in sorted(sleepers.items())})
    leads = collections.Counter(t[0].split(":")[0] for t in T)
    print("leads:", [(k, round(v / len(T), 3)) for k, v in leads.most_common(6)])
    print("examples:")
    for t in T[:6]:
        print("  ", " / ".join(m.split(":")[0] for m in t))


if __name__ == "__main__":
    main()


class TeamMix:
    """Specialisation teams (agent/actor_learner.py --teams mix:DIR:FRAC or mix1:DIR:FRAC): with probability FRAC one of
    our own teams (the Showdown exports in DIR, e.g. data/search_teams_v1), otherwise a real player's
    team from the Metamon replays (data/human_teams_v1.json, agent/team_search.py --dump-teams).
    Each side draws independently, so our teams also meet each other."""

    def __init__(self, spec, human=ROOT / "data/human_teams_v1.json"):
        from agent.team_tournament import showdown_team
        kind, d, frac = spec.split(":")
        self.frac = float(frac)
        # "mix1": only side 0 ever brings our teams, side 1 always a real player's (no mirror matches). Battles
        # are reset side 0 then side 1 (agent/selfplay.Runner), so calls alternate sides.
        self.one_side, self.calls = kind == "mix1", 0
        self.ours = [list(showdown_team(p.read_text())) for p in sorted(Path(d).glob("*_team"))]
        h = json.load(open(human))
        self.human = [list(t) for t in h["opp"] + h["seed"]]

    def __call__(self, rng):
        side, self.calls = self.calls % 2, self.calls + 1
        if (side == 0 or not self.one_side) and rng.random() < self.frac:
            return list(self.ours[rng.integers(len(self.ours))])
        return list(self.human[rng.integers(len(self.human))])
