"""M1 of docs/V5_DESIGN.md: one-turn determinized simultaneous-move search with exact chance and a
policy-anchored root solve.

For each battle where the searching side has a real choice:
1. Guess the opponent's hidden team K times (SearchPlayer.guess: usage prior, revealed moves kept).
2. Priors: our policy's mix from our real view; the opponent's from its view under each guess,
   averaged. Candidates are each side's top `width` actions by prior (all of them if fewer).
3. Every candidate pair is played in every guess under *every* chance outcome
   (VecEnv.branch: damage rolls in `buckets` groups, identical results merged, at most `max_out`
   outcomes kept per pair), and each result is scored by a privileged critic (it sees the guessed
   team in full, which is what a guess is) or +1 / -1 / 0 when the battle is over.
4. The probability-weighted, guess-averaged payoff matrix is solved as a zero-sum game with each
   side's play KL-regularized toward its prior (qsolve.regularized_game, weight tau); we sample
   our action from that mix. Small tau trusts the search, large tau returns the policy.

Hidden counters (hidden=True, the default): a player sees that a Pokemon is asleep or confused, not
for how long, nor how much HP a Substitute has left. Each guess draws them from Gen 1's rules given
what has been seen: sleep from a move lasts 1-7 turns (uniform), Rest sleep 2, confusion 2-5, and each
turn already spent asleep or confused (counted here, per battle) rules out the shorter durations; a
Substitute gets a quarter of its owner's max HP. A side known to be continuing Wrap/Clamp/Bind/Fire
Spin (the Showdown adapter sets trap[game][side] = continuation hits so far) gets its turns left drawn
from the move's 2-5 hits (3/8, 3/8, 1/8, 1/8) given the hits already made. This is what makes the search the same player in our
engine (which would otherwise copy the true counters) and on Showdown (whose adapter can't know them,
agent/showdown_player.py). hidden=False copies the battle's counters as they are (the old behaviour,
which on Showdown means the adapter's placeholders). Exact HP is still copied (a player sees a fraction).

    from agent.search_exact import ExactSearchPlayer
    p = ExactSearchPlayer("agent/runs/v4/snapshots/00500.pt", "mps")   # usable in agent.arena.match
"""
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from agent.qsolve import _norm, _softmax, regularized_game
from agent.search import SearchPlayer, gen1py, safe_choice
from agent.obs import ACTIVE, MAX_HP, MOVES, PP, PRESENT, SLEPT_BY_FOE, STATUS
from agent.selfplay import observe_on

SIDE = 50                                          # battle bytes per side (data/dataset_smogon_v1/schema.json)
STATUS_BYTE, BS1, BS2, CONF_BYTE, ATTACKS_LEFT, SUB_BYTE = 20, 27, 28, 30, 34, 35
SLP_MASK, CONFUSED, SUB, TRAPPING = 0b111, 1 << 7, 1 << 4, 1 << 5
TRAP_HITS = np.array([3, 3, 1, 1]) / 8          # continuation hits after the first: 1, 2, 3 or 4 (engine.rs, eff::TRAPPING)


class ExactSearchPlayer(SearchPlayer):
    trap_uncertain = True   # _draw_hidden draws whether a trap that MAY continue does, per guess (the adapter checks this)

    def __init__(self, ckpt, device, critic=None, k=8, width=6, tau=0.05, buckets=3, max_out=16, block=64, seed=0,
                 teammates=False, depth=1, width2=3, max_out2=4, infer=False, oversample=4, history=12, beta=1.0, oracle=False,
                 rollouts=0, rollout_depth=2, opp_model=None, hidden=True, opp_mix=1.0,
                 calib=None, rounds=1, budget=None, max_rounds=None, opp_width=None, opp_eps=0.0, bayes=False, opp_tau=None, pp_cost=0.0):
        from agent.arena import load_policy
        super().__init__(ckpt, device, k=k, block=block, seed=seed, teammates=teammates)
        critic = critic or str(ckpt).replace(".pt", "_critic.pt")
        self.critic = load_policy(critic, device)
        self.critic_privileged = bool(torch.load(critic, map_location="cpu", weights_only=True).get("privileged"))
        self.width, self.tau, self.buckets, self.max_out = width, tau, buckets, max_out
        # depth > 1: turns after the first are searched too, narrower (width2 actions a side, the
        # max_out2 likeliest chance outcomes per pair), before the critic scores the position.
        self.depth, self.width2, self.max_out2 = depth, width2, max_out2
        self.levels = {}
        # infer: weight the guesses by how well each explains the opponent's past choices.
        # oversample * k candidate guesses are drawn from the prior; each gets the log-likelihood,
        # under our policy from the opponent's view of that guessed team, of its last `history`
        # real choices, tempered by `beta` (the opponent isn't our policy); k are resampled by it.
        self.infer, self.oversample, self.history, self.beta = infer, oversample, history, beta
        self.past, self.pending = {}, {}
        self.hist_env = None
        # oracle (a diagnostic, not a player): the guesses are the opponent's true team, to measure
        # what guessing the hidden information costs.
        self.oracle = oracle
        # rollouts > 0: each leaf is scored by `rollouts` playouts of `rollout_depth` turns, both sides
        # sampling from the policy (the opponent from its view of the guessed team), each ending in the
        # critic's value (or the result); their mean replaces the critic's value at the leaf itself.
        # Errors of one critic call average out, and short-term tactics are played rather than guessed.
        self.hidden, self.seen = hidden, {}      # seen[game][(side, slot)] = [turns asleep, turns confused]
        self.trap = {}                           # trap[game][side] = continuation hits so far (set by the Showdown adapter)
        self.rollouts, self.rollout_depth = rollouts, rollout_depth
        # opp_model: a checkpoint that models the opponent (its move prior in the root game, and the
        # likelihood of its past choices for infer) in place of our own policy, e.g. a network trained to
        # imitate human players (agent/train_bc.py) against opponents that learned from humans.
        # opp_mix (DiL-piKL's human-anchored prior): the opponent's move probabilities are opp_mix x the
        # opp_model's + (1 - opp_mix) x our own policy's (from its view of the guessed team).
        self.opp_policy = load_policy(opp_model, device) if opp_model else self.policy
        self.opp_mix = opp_mix if opp_model else 1.0
        self.roll_env = None
        # calib: the critic's leaf values put on the scale of game results (agent/next_search_calib.py): a JSON file
        # or dict {"a", "b"}; v -> 2 sigmoid(a + b logit((v + 1) / 2)) - 1. Terminal values (+1 / -1 / 0) are exact
        # and untouched. tau is then in calibrated units (about b times the raw ones near an even position).
        if isinstance(calib, (str, Path)):
            calib = json.load(open(calib))
        self.calib = (float(calib["a"]), float(calib["b"]), float(calib.get("clip", 0.999))) if calib else None
        # rounds / budget / max_rounds: the root game is averaged over `rounds` rounds of k guesses (rounds * k worlds);
        # with budget (seconds per _search call, read at each call so a caller may change it per decision), further
        # rounds are added while the slowest round so far still fits, up to max_rounds.
        self.rounds, self.budget = rounds, budget
        self.max_rounds = max_rounds if max_rounds is not None else (rounds if budget is None else 64)
        # opp_width: the opponent's candidate replies (default: width). opp_eps: its prior over them is mixed with
        # uniform at this weight, so a reply our policy finds unlikely (Counter) is still a reply the solve respects.
        self.opp_width, self.opp_eps = opp_width, opp_eps
        # bayes: solve the root as a Bayesian game. The opponent knows its own set, so each *type* (the guesses that
        # give its active Pokemon the same moves and the same legal actions) gets its own reply mix over the actions
        # it really has, regularized toward its own prior; we maximise the average over types. Without it, one reply
        # mix is solved against the guess-averaged matrix, where a move only one guess in eight has is a column the
        # opponent can always play. With all moves revealed there is one type and the two solves agree.
        self.bayes = bayes
        # opp_tau: the opponent's own regularization weight in the root solve (default: tau). Larger = the opponent
        # is taken to play nearer its prior (its model's prediction) than a best reply, and our mix exploits that
        # prediction; equal to tau is the symmetric logit equilibrium. Deeper turns keep tau.
        self.opp_tau = opp_tau
        # pp_cost: the critic barely prices PP (agent/next_search_pp.py: a whole team at 0 PP costs it 0.15), so a
        # move's root payoff is charged pp_cost / (PP it has left): nothing much with a full move, pp_cost itself
        # for the last one. It breaks near-ties away from burning a scarce move into a target that just recovers.
        self.pp_cost = pp_cost
        self._guess_game = "game" in inspect.signature(self.guess).parameters   # belief's per-battle context
        self.name = f"xs{k}w{width}t{tau:g}{'m' if teammates else ''}{f'd{depth}' if depth > 1 else ''}{'i' if infer else ''}{'o' if oracle else ''}{f'r{rollouts}x{rollout_depth}' if rollouts else ''}{('h' if opp_mix >= 1 else f'h{opp_mix:g}') if opp_model else ''}{'' if hidden else 'p'}{'c' if self.calib else ''}{f'R{rounds}' if rounds > 1 else ''}{f'B{budget:g}' if budget else ''}{f'ow{opp_width}' if opp_width else ''}{f'e{opp_eps:g}' if opp_eps else ''}{'b' if bayes else ''}{f'ot{opp_tau:g}' if opp_tau else ''}{f'pp{pp_cost:g}' if pp_cost else ''}:" + str(Path(ckpt).with_suffix("")).replace("agent/runs/", "")
        self.leaves = 0
        # Set to a list to keep every decision as a training example (agent/train_xit.py):
        # (env index, side, team, battle bytes, legal mask, search mix over the 12 actions, root value).
        self.record = None
        # Set to a list to keep every root game (agent/search_audit.py): (env index, my candidate
        # actions, the opponent's, Q (mine x theirs, side's view), my mix, my prior, their prior).
        self.trace = None

    # -- the opponent's past choices (agent.arena.match calls reset, actions, then observe) ----------
    def reset(self):
        self.past, self.pending, self.seen, self.trap = {}, {}, {}, {}

    def _count_seen(self, team, bbs, games):
        """Turns each active Pokemon has spent asleep / confused so far (before this one), per battle;
        updates the counts for this turn."""
        out = []
        for g, game in enumerate(games):
            seen = self.seen.setdefault(int(game), {})
            before = {k: list(v) for k, v in seen.items()}
            for s in (0, 1):
                act = team[g, s, :, ACTIVE]
                if not act.any():
                    continue
                j = int(act.argmax())
                st = seen.setdefault((s, j), [0, 0])
                asleep = team[g, s, j, STATUS] & SLP_MASK
                confused = bbs[g][s * SIDE + BS1] & CONFUSED
                for (ss, jj), v in seen.items():               # confusion ends on switching out
                    if ss == s and jj != j:
                        v[1] = 0
                st[0] = st[0] + 1 if asleep else 0
                st[1] = st[1] + 1 if confused else 0
            out.append(before)
        return out

    def _draw_hidden(self, n, K, side, seen, traps=None):
        """Rewrites self.base's guess rows (g*K+k) with hidden counters drawn as the module docstring says."""
        t, b = self.base.team_state()[:n * K].astype(np.int64), self.base.battle_bytes()[:n * K].copy()
        turns = self.base.turns()[:n * K]
        for r in range(n * K):
            known = seen[r // K]
            for s in (0, 1):
                o = s * SIDE
                act = t[r, s, :, ACTIVE]
                j_act = int(act.argmax()) if act.any() else -1
                for j in range(6):
                    if not t[r, s, j, PRESENT] or not t[r, s, j, STATUS] & SLP_MASK:
                        continue
                    e = known.get((s, j), [0, 0])[0]
                    if t[r, s, j, SLEPT_BY_FOE]:
                        left = int(self.rng.integers(1, max(1, 7 - e) + 1))
                    else:                                  # Rest
                        left = max(1, 2 - e)
                    t[r, s, j, STATUS] = (t[r, s, j, STATUS] & ~SLP_MASK) | left
                    if j == j_act:
                        b[r, o + STATUS_BYTE] = (int(b[r, o + STATUS_BYTE]) & 0xF8) | left
                if j_act >= 0 and b[r, o + BS1] & CONFUSED:
                    e = known.get((s, j_act), [0, 0])[1]
                    b[r, o + CONF_BYTE] = int(self.rng.integers(max(1, 2 - e), max(1, 5 - e) + 1))
                if j_act >= 0 and b[r, o + BS2] & SUB and (s != side or b[r, o + SUB_BYTE] == 0):
                    b[r, o + SUB_BYTE] = max(1, int(t[r, s, j_act, MAX_HP]) // 4)
                h = (traps[r // K] if traps else {}).get(s)
                if h is not None and b[r, o + BS1] & TRAPPING:
                    # The trap MAY continue (Showdown no longer says): N continuation hits in all, N >= the h seen.
                    # N == h: it is over in this guess (flag cleared); else N - h are left. (h = 0: always continues.)
                    w = TRAP_HITS * (np.arange(1, 5) >= max(h, 1))
                    N = int(safe_choice(self.rng, w) + 1) if w.sum() > 0 else h + 1
                    if N <= h:
                        b[r, o + BS1] = int(b[r, o + BS1]) & ~TRAPPING
                        b[r, o + ATTACKS_LEFT] = 0
                    else:
                        b[r, o + ATTACKS_LEFT] = N - h
            self.base.load(r, t[r], bytes(b[r]), int(turns[r]))

    def actions(self, team, battle, mask, side, rng, env=None, idx=None):
        if self.infer:
            opp = 1 - side
            for r in np.flatnonzero(mask[:, opp].sum(-1) > 1):       # the opponent has a real choice here
                self.pending[int(idx[r])] = (team[r].astype(np.int64), bytes(battle[r]))
        return super().actions(team, battle, mask, side, rng, env=env, idx=idx)

    def observe(self, opp_actions, env=None, idx=None):
        for a, i in zip(opp_actions, idx):
            p = self.pending.pop(int(i), None)
            if p is not None:
                self.past.setdefault(int(i), []).append((*p, int(a)))

    def _guess(self, opp_team, game):
        return self.guess(opp_team, game=int(game)) if self._guess_game else self.guess(opp_team)

    @torch.no_grad()
    def _infer(self, team, games, opp):
        """K guesses per battle, resampled from oversample * K prior guesses by the tempered likelihood
        of the opponent's recent choices (each past state rebuilt with the guess, as the opponent saw it)."""
        n, K, M = len(games), self.k, self.k * self.oversample
        cands = [[self._guess(team[g, opp], games[g]) for _ in range(M)] for g in range(n)]
        rows = [(g, c, h) for g in range(n) for h in self.past.get(int(games[g]), [])[-self.history:] for c in range(M)]
        if not rows:
            return [c[:K] for c in cands]
        if self.hist_env is None or len(self.hist_env) < len(rows):
            self.hist_env = gen1py.VecEnv(len(rows), seed=13)
        self.hist_env.clear()
        edits = []
        for r, (g, c, (t, b, _)) in enumerate(rows):
            self.hist_env.load(r, t, b, 0)
            edits += [(r, opp, j, s) for j, s in cands[g][c]]
        self.hist_env.set_members(edits)
        R = len(rows)
        ht, hb, hm = self.hist_env.team_state().astype(np.int16)[:R], self.hist_env.battle_bytes()[:R], self.hist_env.legal_mask()[:R, opp]
        act = np.array([h[2] for _, _, h in rows])
        logp = np.zeros(R, np.float32)
        for c0 in range(0, R, 8192):
            sl = slice(c0, c0 + 8192)
            m = hm[sl].copy()
            m[~m.any(-1), 11] = True
            lp = np.log(self._opp_probs(ht[sl], hb[sl], m, opp) + 1e-12)
            logp[sl] = np.maximum(lp[np.arange(len(lp)), act[sl]], np.log(1e-3))   # floor: an off-model move isn't proof
        ll = np.zeros((n, M))
        np.add.at(ll, ([g for g, _, _ in rows], [c for _, c, _ in rows]), logp)
        out = []
        for g in range(n):
            w = np.exp(self.beta * (ll[g] - ll[g].max()))
            out.append([cands[g][c] for c in safe_choice(self.rng, w, K)])
        return out

    @torch.no_grad()
    def _opp_probs(self, team, bb, mask, opp):
        """The opponent's move probabilities (rows of team/bb, its legal mask), mixed per opp_mix."""
        m = torch.from_numpy(mask).to(self.device)
        lo, _ = self.opp_policy(observe_on(self.opp_policy, team, bb, opp, self.device), m)
        p = torch.softmax(lo.float(), -1).cpu().numpy()
        if self.opp_mix < 1.0:
            lm, _ = self.policy(observe_on(self.policy, team, bb, opp, self.device), m)
            p = self.opp_mix * p + (1 - self.opp_mix) * torch.softmax(lm.float(), -1).cpu().numpy()
        return p

    @torch.no_grad()
    def _search(self, team, mask, side, env, games):
        t0 = time.perf_counter()
        n, K, opp = len(games), self.k, 1 - side
        dev = self.device
        bbs = env.battle_bytes()[games]
        # Counted once per decision (not per round of guesses): turns asleep / confused, trap hits.
        seen = self._count_seen(team, bbs, games) if self.hidden else None
        traps = [self.trap.get(int(g), {}) for g in games]
        mine = px = theirs = None
        num = den = psum = None
        rounds, slow = 0, 0.0
        while True:
            tr = time.perf_counter()
            # 1. K guesses per battle in self.base (row g*K + k).
            self.base.clear()
            self.base.copy_from(env, list(np.repeat(games, K)), list(range(n * K)), int(self.rng.integers(1 << 62)))
            guesses = [[[]] * K for _ in range(n)] if self.oracle else self._infer(team, games, opp) if self.infer else [[self._guess(team[g, opp], games[g]) for _ in range(K)] for g in range(n)]
            edits = [(g * K + k, opp, j, s) for g in range(n) for k in range(K) for j, s in guesses[g][k]]
            self.base.set_members(edits)
            if self.hidden:
                self._draw_hidden(n, K, side, seen, traps)
            gt, gb = self.base.team_state().astype(np.int16)[:n * K], self.base.battle_bytes()[:n * K]
            gmask = self.base.legal_mask()[:n * K]
            # 2. Priors and candidates (ours once; the opponent's menu is fixed by the first round of guesses).
            m_op = gmask[:, opp].copy()
            m_op[~m_op.any(-1), 11] = True
            p_op = self._opp_probs(gt, gb, m_op, opp).reshape(n, K, -1)
            if mine is None:
                m_me = torch.from_numpy(mask[:, side].copy()).to(dev)
                lg, _ = self.policy(observe_on(self.policy, team, bbs, side, dev), m_me)
                p_me = torch.softmax(lg.float(), -1).cpu().numpy()
                mine, theirs, px = [], [], []
                for g in range(n):
                    legal = np.flatnonzero(mask[g, side])
                    rows = legal[np.argsort(-p_me[g, legal])[:self.width]]
                    olegal = np.flatnonzero(m_op[g * K:(g + 1) * K].any(0))   # legal in some guess
                    q = p_op[g].mean(0)
                    cols = olegal[np.argsort(-q[olegal])[:self.opp_width or self.width]]
                    mine.append(rows); theirs.append(cols)
                    px.append(_norm(p_me[g, rows]))
                num = [np.zeros((len(mine[g]), len(theirs[g]))) for g in range(n)]
                den = [np.zeros((len(mine[g]), len(theirs[g]))) for g in range(n)]
                psum = [np.zeros(len(theirs[g])) for g in range(n)]
                types = [{} for _ in range(n)]
            for g in range(n):
                psum[g] += p_op[g].mean(0)[theirs[g]]
            # 3. Exact one-turn expansion of every candidate pair in every guess.
            # An action that is illegal in a guess whose side can only pass there (guesses differ in who is locked
            # in a trap) is played as the pass; otherwise the guess makes the pair impossible and it is skipped.
            jobs, owner = [], []
            m_my = gmask[:, side]
            for g in range(n):
                for k in range(K):
                    ok, okm = m_op[g * K + k], m_my[g * K + k]
                    for i, a in enumerate(mine[g]):
                        a = a if okm[a] else 11 if okm[11] else None
                        if a is None:
                            continue
                        for j, b in enumerate(theirs[g]):
                            b = b if ok[b] else 11 if gmask[g * K + k, opp, 11] else None
                            if b is None:
                                continue   # this guess makes that opposing action impossible
                            jobs.append((g * K + k, int(a), int(b)) if side == 0 else (g * K + k, int(b), int(a)))
                            owner.append((g, k, i, j))
            cap = len(jobs) * self.max_out
            if self.sims is None or len(self.sims) < cap:
                self.sims = gen1py.VecEnv(cap, seed=7)
            idx, prob = self.sims.branch(self.base, jobs, buckets=self.buckets, max_outcomes=self.max_out)
            idx, prob = np.asarray(idx), np.asarray(prob)
            val = self._value(self.sims, len(idx), side, self.depth - 1)
            # 4. Weighted payoff per (guess, pair), summed over the guesses where the pair exists.
            pair_v = np.bincount(idx, weights=prob * val, minlength=len(jobs))
            for (g, k, i, j), v in zip(owner, pair_v):
                num[g][i, j] += v
                den[g][i, j] += 1
            if self.bayes:                               # per-guess matrices, grouped by the opponent's type
                Qw = [np.full((K, len(mine[g]), len(theirs[g])), np.nan) for g in range(n)]
                for (g, k, i, j), v in zip(owner, pair_v):
                    Qw[g][k, i, j] = v
                for g in range(n):
                    act = gt[g * K:(g + 1) * K, opp, :, ACTIVE]
                    for k in range(K):
                        r = g * K + k
                        mv = tuple(int(x) for x in gt[r, opp, int(act[k].argmax()), MOVES]) if act[k].any() else ()
                        key = (mv, tuple(bool(x) for x in ~np.isnan(Qw[g][k]).all(0)))
                        t = types[g].setdefault(key, [0, np.zeros_like(num[g]), np.zeros_like(num[g]), np.zeros(len(theirs[g]))])
                        t[0] += 1
                        t[1] += np.nan_to_num(Qw[g][k]); t[2] += ~np.isnan(Qw[g][k]); t[3] += p_op[g, k][theirs[g]]
            # More rounds of K guesses: at least `rounds`; then, with a time budget (seconds per call), as many as
            # fit, judged by the slowest round so far (a round is never interrupted), at most `max_rounds`.
            rounds += 1
            now = time.perf_counter()
            slow = max(slow, now - tr)
            if rounds >= self.max_rounds or (rounds >= self.rounds and (self.budget is None or now - t0 + 1.25 * slow > self.budget)):
                break
        py = [_norm(psum[g]) for g in range(n)]
        if self.opp_eps > 0:         # a floor under every candidate reply: the opponent isn't our policy
            py = [(1 - self.opp_eps) * y + self.opp_eps / len(y) for y in py]
        choice = np.zeros(n, np.int64)
        for g in range(n):
            seen_pair = den[g] > 0
            Q = np.where(seen_pair, num[g] / np.maximum(den[g], 1), 0.0)
            keep = seen_pair.any(0)                      # opposing actions possible in some guess
            if self.pp_cost > 0 and team[g, side, :, ACTIVE].any():
                left = team[g, side, int(team[g, side, :, ACTIVE].argmax()), PP]
                fee = np.array([self.pp_cost / max(int(left[a]), 1) if a < 4 else 0.0 for a in mine[g]])
                Q = Q - fee[:, None]
                for t in types[g].values():
                    t[1] -= fee[:, None] * t[2]
            if self.opp_tau is None:
                x, y = regularized_game(Q[:, keep], px[g], _norm(py[g][keep]), self.tau, both=True)
            v_root = None
            if self.opp_tau is not None or (self.bayes and len(types[g]) > 1):
                ty = list(types[g].values()) if self.bayes else [[1, Q, seen_pair.astype(float), py[g]]]
                x, y, v_root = _bayes_game(ty, keep, px[g], self.tau, 0.0 if not self.bayes else self.opp_eps, opp_tau=self.opp_tau)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):   # a NaN prior or value (seen once on the ladder)
                self.nan_mixes = getattr(self, "nan_mixes", 0) + 1
                Qk = np.nan_to_num(Q[:, keep])
                x = np.eye(len(mine[g]))[Qk.mean(1).argmax()]
                y = np.full(Qk.shape[1], 1.0 / Qk.shape[1])
            x = np.asarray(x, np.float64)
            x = x / x.sum()
            choice[g] = mine[g][safe_choice(self.rng, x)]
            if g == 0:                                  # for decision traces (agent/showdown_player.py --trace)
                self.last = {"mine": mine[0].tolist(), "Q": Q[:, keep], "mix": x, "value": float(x @ Q[:, keep] @ y) if v_root is None else v_root,
                             "theirs": np.asarray(theirs[0])[keep].tolist(), "y": np.asarray(y, np.float64),
                             "px": np.asarray(px[0], np.float64), "py": np.asarray(_norm(py[0][keep]), np.float64),
                             "worlds": rounds * K, "secs": time.perf_counter() - t0, "types": len(types[0]) if self.bayes else 1}
            if self.trace is not None:
                self.trace.append((int(games[g]), mine[g].copy(), np.asarray(theirs[g])[keep], Q[:, keep].copy(), x.copy(),
                                   px[g].copy(), _norm(py[g][keep])))
            if self.record is not None:
                pi = np.zeros(12, np.float32)
                pi[mine[g]] = x
                self.record.append((int(games[g]), side, team[g].copy(), bbs[g].copy(), mask[g, side].copy(), pi,
                                    float(x @ Q[:, keep] @ y) if v_root is None else v_root))
        return choice

    def _value(self, env, m, side, depth):
        """Values of env's battles 0..m-1 from `side`'s view: +1 / -1 / 0 when over, else the critic
        (depth 0) or another turn: each side's top `width2` actions by the policy, every pair expanded
        under its `max_out2` likeliest chance outcomes, valued recursively, the regularized game solved."""
        dev, opp = self.device, 1 - side
        res = env.results()[:m]
        val = np.where(res == side, 1.0, np.where(res == opp, -1.0, 0.0)).astype(np.float32)
        live = np.flatnonzero(res < 0)
        self.leaves += m
        if not len(live):
            return val
        if depth == 0 and self.rollouts:
            val[live] = self._rollout_value(env, live, side)
            return val
        t, bb = env.team_state().astype(np.int16)[:m], env.battle_bytes()[:m]
        if depth == 0:
            for c in range(0, len(live), 8192):
                sel = live[c:c + 8192]
                ts = t[sel]
                if self.critic_privileged:
                    from agent.train_v4 import reveal
                    ts = reveal(ts)
                o = observe_on(self.critic, ts, bb[sel], side, dev)
                _, v = self.critic(o, torch.ones(len(sel), 12, dtype=torch.bool, device=dev))
                val[sel] = self._cal(v.float().cpu().numpy())
            return val
        # Candidates per live battle: each side's top width2 legal actions by its policy (pass if none).
        mask = env.legal_mask()[:m][live]
        w = self.width2
        cand, prior = [], []
        for s in (side, opp):
            ms = mask[:, s].copy()
            ms[~ms.any(-1), 11] = True
            p = np.zeros((len(live), 12), np.float32)
            for c in range(0, len(live), 8192):
                sel = slice(c, c + 8192)
                lg, _ = self.policy(observe_on(self.policy, t[live[sel]], bb[live[sel]], s, dev), torch.from_numpy(ms[sel]).to(dev))
                p[sel] = torch.softmax(lg.float(), -1).cpu().numpy()
            p = np.where(ms, p, -1.0)
            top = np.argsort(-p, -1)[:, :w]                                  # (L, w)
            ok = np.take_along_axis(p, top, -1) >= 0
            pr = np.where(ok, np.take_along_axis(p, top, -1), 0.0)
            cand.append((top, ok)); prior.append(pr / pr.sum(-1, keepdims=True))
        (ta, oka), (tb, okb) = cand
        L = len(live)
        li, i, j = np.nonzero(oka[:, :, None] & okb[:, None, :])
        a, b = ta[li, i], tb[li, j]
        jobs = [(int(live[l]), int(x), int(y)) if side == 0 else (int(live[l]), int(y), int(x)) for l, x, y in zip(li, a, b)]
        child = self.levels.get(depth)
        cap = len(jobs) * self.max_out2
        if child is None or len(child) < cap:
            child = self.levels[depth] = gen1py.VecEnv(cap, seed=11 + depth)
        cidx, cprob = child.branch(env, jobs, buckets=self.buckets, max_outcomes=self.max_out2)
        cidx, cprob = np.asarray(cidx), np.asarray(cprob)
        cval = self._value(child, len(cidx), side, depth - 1)
        pair = np.bincount(cidx, weights=cprob * cval, minlength=len(jobs))
        Q = np.zeros((L, w, w))
        Q[li, i, j] = pair
        val[live] = _batched_game(Q, prior[0], prior[1], oka, okb, self.tau)
        return val


    def _cal(self, v):
        """Critic values on the scale of results (see calib); the identity without a calibration."""
        if self.calib is None:
            return v
        a, b, clip = self.calib
        p = np.clip((v.astype(np.float64) + 1) / 2, 1 - clip, clip)
        return (2 / (1 + np.exp(-(a + b * np.log(p / (1 - p))))) - 1).astype(np.float32)

    def _critic_values(self, t, bb, side):
        """The critic's values of positions (team arrays t, battle bytes bb) from `side`'s view."""
        out = np.zeros(len(t), np.float32)
        for c in range(0, len(t), 8192):
            sel = slice(c, c + 8192)
            ts = t[sel]
            if self.critic_privileged:
                from agent.train_v4 import reveal
                ts = reveal(ts)
            o = observe_on(self.critic, ts, bb[sel], side, self.device)
            _, v = self.critic(o, torch.ones(len(ts), 12, dtype=torch.bool, device=self.device))
            out[sel] = self._cal(v.float().cpu().numpy())
        return out

    def _rollout_value(self, env, live, side):
        """Mean over self.rollouts playouts of self.rollout_depth turns from each of env's battles
        `live`: both sides sample from the policy; a finished playout scores its result, the rest the critic."""
        R, opp, dev = self.rollouts, 1 - side, self.device
        n = len(live) * R
        if self.roll_env is None or len(self.roll_env) < n:
            self.roll_env = gen1py.VecEnv(n, seed=17)
        e = self.roll_env
        e.copy_from(env, [int(x) for x in np.repeat(live, R)], list(range(n)), int(self.rng.integers(1 << 62)))
        for _ in range(self.rollout_depth):
            t, bb, mask = e.team_state().astype(np.int16), e.battle_bytes(), e.legal_mask()
            # rows past n hold leftover battles from earlier calls: any legal action keeps step() valid
            act = np.where(mask.any(-1), mask.argmax(-1), 11).astype(np.int64)
            running = np.flatnonzero(e.results()[:n] < 0)
            if not len(running):
                break
            for s in (0, 1):
                ms = mask[running, s].copy()
                ms[~ms.any(-1), 11] = True
                for c in range(0, len(running), 8192):
                    sel = running[c:c + 8192]
                    lg, _ = self.policy(observe_on(self.policy, t[sel], bb[sel], s, dev), torch.from_numpy(ms[c:c + 8192]).to(dev))
                    act[sel, s] = torch.distributions.Categorical(logits=lg.float()).sample().cpu().numpy()
            e.step(act)
        res = e.results()[:n]
        val = np.where(res == side, 1.0, np.where(res == opp, -1.0, 0.0)).astype(np.float32)
        go = np.flatnonzero(res < 0)
        if len(go):
            t, bb = e.team_state().astype(np.int16)[:n], e.battle_bytes()[:n]
            val[go] = self._critic_values(t[go], bb[go], side)
        self.leaves += n
        return val.reshape(len(live), R).mean(1)


def _batched_game(Q, px, py, okx, oky, tau, iters=200):
    """Values x Q y of a batch of padded games (qsolve.regularized_game, vectorized); padded actions
    (ok False) have zero prior and stay unplayed."""
    lx = np.where(okx, np.log(np.maximum(px, 1e-9)), -1e9)
    ly = np.where(oky, np.log(np.maximum(py, 1e-9)), -1e9)
    x, y = px.copy(), py.copy()
    sm = lambda z: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(z - z.max(-1, keepdims=True)))
    for _ in range(iters):
        x = 0.5 * x + 0.5 * sm(lx + np.einsum("nab,nb->na", Q, y) / tau)
        y = 0.5 * y + 0.5 * sm(ly - np.einsum("nab,na->nb", Q, x) / tau)
    return np.einsum("na,nab,nb->n", x, Q, y)


def _bayes_game(types, keep, px, tau, eps=0.0, iters=300, opp_tau=None):
    """The root as a Bayesian game (see ExactSearchPlayer.bayes). types: [count, sum of Q, count per cell, sum of
    the reply prior] per opponent type, over all candidate columns; keep: the columns in the reported game.
    Each type t replies y_t ~ py_t exp(-Q_t' x / tau) over the columns it has; x ~ px exp(sum_t w_t Q_t y_t / tau).
    Returns x, the marginal reply mix over the kept columns, and the value sum_t w_t x Q_t y_t."""
    w = np.array([t[0] for t in types], np.float64)
    w /= w.sum()
    Qs, lys, has = [], [], []
    for _, qs, cnt, ps in types:
        h = (cnt > 0).any(0) & keep
        q = np.where(cnt > 0, qs / np.maximum(cnt, 1), 0.0)[:, keep]
        p = _norm(np.where(h, ps, 0.0)[keep])
        if eps > 0:
            p = (1 - eps) * p + eps * h[keep] / max(h[keep].sum(), 1)
        Qs.append(q); has.append(h[keep]); lys.append(np.where(h[keep], np.log(np.maximum(p, 1e-9)), -1e9))
    lx = np.log(np.maximum(px, 1e-9))
    x = px.astype(np.float64).copy()
    ys = [_softmax(ly) for ly in lys]
    for _ in range(iters):
        x = 0.5 * x + 0.5 * _softmax(lx + sum(wt * (q @ y) for wt, q, y in zip(w, Qs, ys)) / tau)
        ys = [0.5 * y + 0.5 * _softmax(ly - q.T @ x / (opp_tau or tau)) for q, y, ly in zip(Qs, ys, lys)]
    x = x / x.sum()
    return x, sum(wt * y for wt, y in zip(w, ys)), float(sum(wt * (x @ q @ y) for wt, q, y in zip(w, Qs, ys)))
