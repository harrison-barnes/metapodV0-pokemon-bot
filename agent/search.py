"""Decision-time search for the 6v6 agent: a one-turn simultaneous-move lookahead.

For each battle where the searching side has a real choice:
1. Guess the hidden information K times (a determinization). Each opposing Pokemon not yet
   seen becomes a set drawn from the ladder pool by usage (a species not already on that team).
   Each seen one keeps its species and revealed moves, and gets a set drawn from the posterior of
   agent/matchups.py (usage weight, restricted to sets containing every revealed move); its
   unrevealed slots take that set's other moves.
2. For every pair (own action, opposing action), play the turn out in a copy of each guess, each
   copy with its own random stream, and score the resulting position with the network's value
   head from the searching side's view (+1 / -1 / 0 when the battle ends).
3. Average over the K guesses into a payoff matrix, solve it as a zero-sum matrix game (linear
   programming), and play the equilibrium mix.

Exact HP and sleep counters are copied from the real battle into the guesses, where a player
would only see an HP fraction; that small leak is the only hidden information the search uses.

Needs the engine's Python bindings (engine/py: VecEnv.copy_from, set_members, clear), installed as the
`gen1py` package (`uv sync` builds them).

    from agent.search import SearchPlayer
    p = SearchPlayer("agent/runs/selfplay_v3/latest.pt", "mps", k=6)   # usable in agent.arena.match
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import gen1py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

from agent.obs import ACTIVE, MOVES, MOVES_REVEALED, PRESENT, REVEALED, SPECIES  # noqa: E402
from agent.selfplay import observe_on  # noqa: E402

DEX = json.load(open(ROOT / "data/gen1.json"))
POOL = json.load(open(ROOT / "data/sets_ladder.json"))
SPECIES_ID = {s["name"]: i for i, s in enumerate(DEX["species"])}
MOVE_ID = {m["name"]: m["id"] for m in DEX["moves"] if m}
MOVE_NAME = {m["id"]: m["name"] for m in DEX["moves"] if m}
POOL_SPECIES = np.array([SPECIES_ID[s["species"]] for s in POOL])
POOL_W = np.array([s["weight"] for s in POOL], np.float64)
USAGE = ROOT / "data/smogon/gen1ou-1760-2026-09.json"   # the same ladder statistics the pool came from



# Moves a made-up set shouldn't assume (two-turn, self-KO, recharge or conditional moves)
NO_FILL = {"EXPLOSION", "SELFDESTRUCT", "HYPER_BEAM", "SOLARBEAM", "SKY_ATTACK", "RAZOR_WIND", "SKULL_BASH",
           "DREAM_EATER", "DIG", "FLY"}


def fallback_set(sp, known):
    """A set for a species the pool doesn't cover: its known move ids, then its strongest learnset moves."""
    d = DEX["species"][sp]
    fill = sorted((m for m in d["learnset"] if m not in known and MOVE_NAME[m] not in NO_FILL),
                  key=lambda m: -DEX["moves"][m]["power"])
    return {"species": d["name"], "moves": [MOVE_NAME[m] for m in (list(known) + fill)[:4]]}


def safe_choice(rng, w, size=None):
    """rng.choice over len(w) by weights w, robust to float32 rounding, negatives, NaN and all-zero weights
    (uniform then); numpy refuses probabilities that don't sum to 1 within float64 tolerance."""
    w = np.nan_to_num(np.asarray(w, np.float64), nan=0.0, posinf=0.0)
    w = np.clip(w, 0, None)
    total = w.sum()
    p = w / total if total > 0 else np.full(len(w), 1.0 / len(w))
    return rng.choice(len(w), size=size, p=p)

def teammate_lift(path=USAGE, clip=10.0):
    """(N_SPECIES, N_SPECIES): P(B on the team | A on the team) / P(B), from Smogon's weighted teammate
    counts; 1 where there is no data. Clipped to [1/clip, clip]."""
    norm = lambda name: re.sub(r"[^a-z0-9]", "", name.lower())
    sid = {norm(s["name"]): i for i, s in enumerate(DEX["species"])}
    data = json.load(open(path))["data"]
    w = {k: sum(v["Abilities"].values()) for k, v in data.items()}
    teams = sum(w.values()) / 6
    lift = np.ones((len(DEX["species"]), len(DEX["species"])))
    for a, v in data.items():
        for b, c in v["Teammates"].items():
            if norm(a) in sid and norm(b) in sid and w[a] > 0 and w.get(b, 0) > 0:
                lift[sid[norm(a)], sid[norm(b)]] = np.clip((c / w[a]) / (w[b] / teams), 1 / clip, clip)
    return lift


def solve_game(Q):
    """Row player's equilibrium mix for the zero-sum payoff matrix Q (rows maximise)."""
    r, c = Q.shape
    if r == 1:
        return np.ones(1)
    if c == 1:
        out = np.zeros(r)
        out[Q[:, 0].argmax()] = 1
        return out
    # maximise v s.t. Q^T x >= v, sum x = 1, x >= 0  ->  variables (x, v), minimise -v.
    res = linprog(np.r_[np.zeros(r), -1.0], A_ub=np.c_[-Q.T, np.ones(c)], b_ub=np.zeros(c),
                  A_eq=np.r_[np.ones(r), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * r + [(None, None)], method="highs")
    if not res.success:
        out = np.zeros(r)
        out[Q.min(1).argmax()] = 1
        return out
    x = np.clip(res.x[:r], 0, None)
    return x / x.sum()


class SearchPlayer:
    """An arena player (see agent.arena.match) that searches one turn ahead with a policy's value head."""

    def __init__(self, ckpt, device, k=6, block=96, seed=0, cheat=False, mode="nash", teammates=False):
        from agent.arena import load_policy
        self.policy = load_policy(ckpt, device)
        self.mu = self.policy.mu
        self.device, self.k, self.block = device, k, block
        self.name = "search:" + str(Path(ckpt).with_suffix("")).replace("agent/runs/", "")
        self.rng = np.random.default_rng(seed)
        self.base = gen1py.VecEnv(block * k, seed=seed + 1)
        self.sims = None
        self.calls = 0
        self.cheat = cheat     # diagnostics only: search with the true hidden information
        self.mode = mode       # "nash": play the equilibrium mix; "maxmin" / "mean": pure best row
        # Unseen opponents drawn by usage alone, or by usage times the teammate lift from every
        # opponent seen or already guessed (coherent teams).
        self.lift = teammate_lift() if teammates else None
        # v4 checkpoints trained with a privileged critic score fully revealed positions, which
        # is what a guess of the hidden sets gives (agent/train_v4.py).
        self.privileged = bool(torch.load(ckpt, map_location="cpu", weights_only=True).get("privileged"))

    # -- hidden information ------------------------------------------------------------------
    def guess(self, opp):
        """One guess for an opposing team (6, TEAM_FIELDS): [(slot, "SPECIES:M,M,M,M")] edits."""
        edits = []
        seen = [int(opp[j, SPECIES]) for j in range(6) if opp[j, PRESENT] and opp[j, REVEALED]]
        used = set(seen)
        cand, w = self.mu.posterior(torch.as_tensor(opp[None].astype(np.int64), device=self.device))
        cand, w = cand[0].cpu().numpy(), w[0].cpu().numpy()
        for j in range(6):
            if not opp[j, PRESENT]:
                continue
            if opp[j, REVEALED]:
                slots = [int(m) for m in opp[j, MOVES]]
                known = [(int(opp[j, MOVES_REVEALED]) >> k) & 1 and slots[k] > 0 for k in range(4)]
                pick = POOL[cand[j][safe_choice(self.rng, w[j])]] if w[j].sum() > 0 else None
                if pick is None or SPECIES_ID[pick["species"]] != int(opp[j, SPECIES]):
                    # a species with no set in the pool (its candidate row is padding): keep the species, with the
                    # moves it has shown and the rest from its learnset
                    pick = fallback_set(int(opp[j, SPECIES]), [slots[k] for k in range(4) if known[k]])
                rest = [MOVE_ID[m] for m in pick["moves"] if MOVE_ID[m] not in {slots[k] for k in range(4) if known[k]}]
                moves = [slots[k] if known[k] else (rest.pop(0) if rest else 0) for k in range(4)]
                edits.append((j, pick["species"] + ":" + ",".join(MOVE_NAME[m] for m in moves if m)))
            else:
                ok = ~np.isin(POOL_SPECIES, list(used))
                p = POOL_W * ok
                if self.lift is not None:
                    p = p * np.prod(self.lift[list(used)][:, POOL_SPECIES], axis=0) if used else p
                s = int(safe_choice(self.rng, p))
                used.add(int(POOL_SPECIES[s]))
                edits.append((j, POOL[s]["species"] + ":" + ",".join(POOL[s]["moves"])))
        return edits

    # -- search ------------------------------------------------------------------------------
    @torch.no_grad()
    def actions(self, team, battle, mask, side, rng, env=None, idx=None):
        """Actions for `side` in battles idx of `env` (team/battle/mask are those battles' rows)."""
        out = np.where(mask[:, side].any(-1), mask[:, side].argmax(-1), 11)
        choose = np.flatnonzero(mask[:, side].sum(-1) > 1)
        for b0 in range(0, len(choose), self.block):
            blk = choose[b0:b0 + self.block]
            out[blk] = self._search(team[blk], mask[blk], side, env, idx[blk])
        self.calls += 1
        return out

    @torch.no_grad()
    def _search(self, team, mask, side, env, games):
        n, K, opp_side = len(games), self.k, 1 - side
        # 1. K guesses per battle in self.base (row g*K + k).
        self.base.clear()
        self.base.copy_from(env, list(np.repeat(games, K)), list(range(n * K)), int(self.rng.integers(1 << 62)))
        edits = []
        for g in range(n):
            for k in range(K):
                edits += [(g * K + k, opp_side, j, s) for j, s in self.guess(team[g, opp_side])]
        if not self.cheat:
            self.base.set_members(edits)
        # 2. Every action pair in a copy of every guess.
        mine = [np.flatnonzero(mask[g, side]) for g in range(n)]
        theirs = [np.flatnonzero(mask[g, opp_side]) if mask[g, opp_side].any() else np.array([11]) for g in range(n)]
        src, acts, owner = [], [], []
        for g in range(n):
            for k in range(K):
                for a in mine[g]:
                    for b in theirs[g]:
                        src.append(g * K + k)
                        acts.append((a, b) if side == 0 else (b, a))
                        owner.append((g, k, a, b))
        m = len(src)
        if self.sims is None or len(self.sims) < m:
            self.sims = gen1py.VecEnv(max(m, 2 * (self.sims.__len__() if self.sims else 0)), seed=7)
        self.sims.clear()
        self.sims.copy_from(self.base, src, list(range(m)), int(self.rng.integers(1 << 62)))
        full = np.full((len(self.sims), 2), 11, np.int64)
        full[:m] = np.array(acts, np.int64)
        self.sims.step(full)
        # 3. Score each result from the searching side's view.
        res = self.sims.results()[:m]
        t, bb = self.sims.team_state().astype(np.int16)[:m], self.sims.battle_bytes()[:m]
        val = np.where(res == side, 1.0, np.where(res == opp_side, -1.0, 0.0)).astype(np.float32)
        live = np.flatnonzero(res < 0)
        for c in range(0, len(live), 8192):
            sel = live[c:c + 8192]
            ts = t[sel]
            if self.privileged:
                from agent.train_v4 import reveal
                ts = reveal(ts)
            o = observe_on(self.policy, ts, bb[sel], side, self.device)
            m_side = torch.ones(len(sel), 12, dtype=torch.bool, device=self.device)
            _, v = self.policy(o, m_side)
            val[sel] = v.float().cpu().numpy()
        # 4. Payoff matrices, equilibrium mixes, sampled actions.
        choice = np.zeros(n, np.int64)
        Q = [np.zeros((len(mine[g]), len(theirs[g]))) for g in range(n)]
        row = [{a: i for i, a in enumerate(mine[g])} for g in range(n)]
        col = [{b: i for i, b in enumerate(theirs[g])} for g in range(n)]
        for (g, k, a, b), v in zip(owner, val):
            Q[g][row[g][a], col[g][b]] += v / K
        for g in range(n):
            x = None
            if self.mode == "mean":
                choice[g] = mine[g][Q[g].mean(1).argmax()]
            elif self.mode == "maxmin":
                choice[g] = mine[g][Q[g].min(1).argmax()]
            else:
                x = solve_game(Q[g])
                choice[g] = mine[g][safe_choice(self.rng, x)]
            if g == 0:                                 # for decision traces (agent/showdown_player.py --trace)
                self.last = {"mine": mine[0].tolist(), "theirs": theirs[0].tolist(), "Q": Q[0], "mix": x}
        return choice
