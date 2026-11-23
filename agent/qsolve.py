"""Decision by equilibrium: play the mixed-strategy solution of the network's payoff matrix.

A v4 network with the Q head (agent/policy.py) predicts Q[a, b], this side's value for playing
action a while the opponent plays b. QPlayer turns that into a decision:
1. Guess the opponent's hidden sets K times (agent/search.py's posterior: unseen Pokemon from the
   ladder pool by usage, seen ones keep their revealed moves and get a consistent set), and build
   the fully revealed observation for each guess. This needs no engine copies: only the team
   arrays change.
2. Average Q over the guesses (the payoff matrix under the current belief), keep the legal rows
   (own actions) and columns (the opponent's legal actions are public: which Pokemon it can
   switch to, whether it must recharge), and solve the zero-sum matrix game by LP.
3. Play the equilibrium mix ("nash"), or ("safe") exploit this opponent safely. After each turn
   the arena shows the player the opponent's actual choice (`observe`). The player keeps, per
   battle, the value it has gained from the opponent's departures from equilibrium (its choice's
   value against our mix, minus the equilibrium expectation), and how often the opponent attacks,
   switches or passes. It then plays a best response to that observed tendency mixed into the
   equilibrium, with weight capped by the value banked so far: a wrong model can only give back
   what the opponent's mistakes already gave us (after Ganzfried & Sandholm, "Safe opponent
   exploitation", 2012).

    p = QPlayer("agent/runs/v4/latest.pt", "mps", k=8)     # usable in agent.arena.match
"""
from pathlib import Path

import numpy as np
import torch

from agent.obs import MOVES, MOVES_REVEALED, PRESENT, REVEALED, SPECIES, HP, MAX_HP
from agent.search import MOVE_ID, POOL, POOL_SPECIES, POOL_W, solve_game
from agent.selfplay import observe_on


def guess_teams(mu, opp, k, rng, device):
    """k guesses of one opposing team (6, 18) as fully revealed arrays: (k, 6, 18)."""
    out = np.repeat(opp[None].astype(np.int16), k, 0)
    cand, w = mu.posterior(torch.as_tensor(opp[None].astype(np.int64), device=device))
    cand, w = cand[0].cpu().numpy(), w[0].cpu().numpy()
    seen = [int(opp[j, SPECIES]) for j in range(6) if opp[j, PRESENT] and opp[j, REVEALED]]
    for g in range(k):
        used = set(seen)
        t = out[g]
        for j in range(6):
            if not opp[j, PRESENT]:
                continue
            if opp[j, REVEALED]:
                pick = POOL[cand[j][rng.choice(len(w[j]), p=w[j])]]
                slots = [int(m) for m in opp[j, MOVES]]
                known = [((int(opp[j, MOVES_REVEALED]) >> i) & 1) and slots[i] > 0 for i in range(4)]
                have = {slots[i] for i in range(4) if known[i]}
                rest = [MOVE_ID[m] for m in pick["moves"] if MOVE_ID[m] not in have]
                t[j, MOVES] = [slots[i] if known[i] else (rest.pop(0) if rest else 0) for i in range(4)]
            else:
                ok = ~np.isin(POOL_SPECIES, list(used))
                p = POOL_W * ok
                s = int(rng.choice(len(POOL), p=p / p.sum()))
                used.add(int(POOL_SPECIES[s]))
                t[j, SPECIES] = POOL_SPECIES[s]
                t[j, MOVES] = [MOVE_ID[m] for m in POOL[s]["moves"]] + [0] * (4 - len(POOL[s]["moves"]))
                from agent.features import MAX_HP as SPECIES_MAX_HP
                t[j, MAX_HP] = SPECIES_MAX_HP[t[j, SPECIES]]
                t[j, HP] = t[j, MAX_HP]                     # never sent out, so at full HP
            t[j, REVEALED] = 1
            t[j, MOVES_REVEALED] = 15
    return out


class QPlayer:
    def __init__(self, ckpt, device, k=8, seed=0, mode="nash", prior=None, tau=0.05):
        from agent.arena import load_policy
        self.policy = load_policy(ckpt, device)
        assert self.policy.qhead, f"{ckpt} has no Q head"
        self.mu, self.device, self.k, self.mode = self.policy.mu, device, k, mode
        self.rng = np.random.default_rng(seed)
        # "reg": the policy network's mixes are the prior both sides' play is regularized toward.
        self.prior = load_policy(prior or str(ckpt).replace("_critic", ""), device) if mode == "reg" else None
        self.tau = tau
        tag = {"nash": "", "safe": "s", "reg": f"r{tau:g}"}[mode]
        self.name = f"q{k}{tag}:" + str(Path(ckpt).with_suffix("")).replace("agent/runs/", "")
        self.games = {}       # (env id, battle index) -> {"banked", "counts", "pending"}

    @torch.no_grad()
    def matrices(self, team, battle, side):
        """Belief-averaged Q (n, 12, 12) for `side` of each battle."""
        n, K = len(team), self.k
        big = np.repeat(team.astype(np.int16), K, 0)                              # (n*K, 2, 6, 18)
        for g in range(n):
            big[g * K:(g + 1) * K, 1 - side] = guess_teams(self.mu, team[g, 1 - side], K, self.rng, self.device)
        bat = np.repeat(battle, K, 0)
        o = observe_on(self.policy, big, bat, side, self.device)
        mask = torch.ones(len(big), 12, dtype=torch.bool, device=self.device)
        _, _, ex = self.policy(o, mask, extras=True)
        Q = ex["q"].float().cpu().numpy().reshape(n, K, 12, 12).mean(1)
        if self.prior is None:
            return Q
        # Prior mixes: ours from our real view; the opponent's from its view under each guess.
        mine, _ = self.prior(observe_on(self.prior, team, battle, side, self.device), torch.ones(n, 12, dtype=torch.bool, device=self.device))
        theirs, _ = self.prior(observe_on(self.prior, big, bat, 1 - side, self.device), mask)
        theirs = torch.softmax(theirs.float(), -1).cpu().numpy().reshape(n, K, 12).mean(1)
        return Q, mine.float().cpu().numpy(), theirs

    def actions(self, team, battle, mask, side, rng, env=None, idx=None):
        out = np.where(mask[:, side].any(-1), mask[:, side].argmax(-1), 11)
        choose = np.flatnonzero(mask[:, side].sum(-1) > 1)
        if len(choose) == 0:
            return out
        Q = self.matrices(team[choose], battle[choose], side)
        if self.prior is not None:
            Q, lg_me, p_opp = Q
        for i, g in enumerate(choose):
            rows = np.flatnonzero(mask[g, side])
            cols = np.flatnonzero(mask[g, 1 - side]) if mask[g, 1 - side].any() else np.array([11])
            Qs = Q[i][np.ix_(rows, cols)]
            if self.prior is not None:
                x = regularized_game(Qs, _softmax(lg_me[i, rows]), _norm(p_opp[i, cols]), self.tau)
                out[g] = rows[self.rng.choice(len(x), p=x)]
                continue
            x = solve_game(Qs)
            if self.mode == "safe" and idx is not None:
                st = self.games.setdefault((id(env), int(idx[g])), {"banked": 0.0, "counts": np.zeros(3), "pending": None})
                y_eq = solve_game(-Qs.T)
                cls = action_class(cols)
                n_obs = st["counts"].sum()
                if n_obs > 0:
                    per = np.array([st["counts"][c] / max((cls == c).sum(), 1) for c in cls])
                    y_obs = per / per.sum() if per.sum() > 0 else y_eq
                    y_model = (n_obs * y_obs + 5 * y_eq) / (n_obs + 5)
                    x = _tilt(Qs, x, y_model, st["banked"])
                st["pending"] = (Qs, x, y_eq, cols)
            out[g] = rows[self.rng.choice(len(x), p=x)]
        return out

    def observe(self, opp_actions, env=None, idx=None):
        """The opponent's actual choices in battles idx (see the module docstring)."""
        if self.mode != "safe":
            return
        for g, b in zip(idx, opp_actions):
            st = self.games.get((id(env), int(g)))
            if st is None or st["pending"] is None:
                continue
            Qs, x, y_eq, cols = st["pending"]
            st["pending"] = None
            hit = np.flatnonzero(cols == b)
            if len(hit) == 0:
                continue
            # Unbiased when the opponent plays equilibrium; no floor at 0, which would make noise
            # accumulate as "banked" value.
            st["banked"] += float(x @ Qs[:, hit[0]] - x @ Qs @ y_eq)
            st["counts"][action_class(np.array([b]))[0]] += 1


def _softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


def _norm(p):
    return p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))


def regularized_game(Q, px, py, tau, iters=200, both=False):
    """Equilibrium of the zero-sum game Q with each side's play regularized toward its prior by a
    KL penalty of weight tau (a logit equilibrium with prior: x ~ px exp(Q y / tau), and likewise
    y). Small tau trusts Q; large tau returns the priors. Damped fixed-point iteration. Returns
    the row mix, or (row mix, column mix) with both=True."""
    x, y = px.copy(), py.copy()
    lx, ly = np.log(np.maximum(px, 1e-9)), np.log(np.maximum(py, 1e-9))
    for _ in range(iters):
        x = 0.5 * x + 0.5 * _softmax(lx + Q @ y / tau)
        y = 0.5 * y + 0.5 * _softmax(ly - Q.T @ x / tau)
    if both:
        return x / x.sum(), y / y.sum()
    return x / x.sum()


def action_class(actions):
    """0 a move (or Struggle), 1 a switch, 2 pass."""
    return np.where(actions <= 4, 0, np.where(actions <= 10, 1, 2))


def _tilt(Qs, x_eq, y_model, banked, cap=0.5):
    """Equilibrium mix tilted toward the best response to y_model, by at most the banked value."""
    lam = min(cap, banked)
    if lam <= 0:
        return x_eq
    br = np.zeros_like(x_eq)
    br[(Qs @ y_model).argmax()] = 1
    return (1 - lam) * x_eq + lam * br
