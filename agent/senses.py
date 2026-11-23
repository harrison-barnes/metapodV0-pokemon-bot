"""Senses: multi-Pokemon quantities computed from the exact 1v1 table, as extra network inputs.

agent/matchups.py gives each 1v1 value in isolation, which a network can approximate on its own.
These chain or invert those values into things that are cheap to compute exactly but take many
layers (and many games) to infer (docs/V4_DESIGN.md, part 1). For one side's view of every env:

Per own team member i (own_senses, (n, 6, 4)):
- sweep: probability that i alone beats every seen, still-standing opposing Pokemon in turn, with
  the opponent choosing the order adversarially (each step it sends the Pokemon i is least
  likely to beat). HP carries over: after a win, i's HP is its expected HP left given the win
  (table hp_left / value). Status carries over unchanged.
- predictability: the opponent's posterior probability of i's true set, given the moves i has
  shown (0 while i hasn't been seen: the opponent doesn't know the species either).
- concealment: how much the opponent's belief misjudges i, as i's true 1v1 value minus the value
  the opponent expects under its belief, averaged over the seen opposing Pokemon. Positive means
  the opponent underestimates i: hidden information worth keeping (the "cav u" gap of
  Aumann-Maschler, measured on the 1v1 values).
- valid: 1 where i is alive and the quantities mean something.

Per opposing team member j (opp_senses, (n, 6, 4)):
- threat: probability that j alone sweeps our standing team (adversarial order from our side:
  we send whoever beats j most likely), averaged over j's set posterior.
- valid, then two zeros (same width as own_senses).

Global (global_senses, (n, 4)): sack gain (best standing bench member's value against the opposing
active minus the active's own value: what a free switch after a faint is worth), the best
switch-in value, the fraction of the opponent's team still unseen, and our own fraction unrevealed.

Unseen opposing Pokemon are left out of the chains (their species is unknown).
"""
import numpy as np
import torch

from agent.matchups import Matchups, has_hp_left, hp_interp, load_hp_left, status_options
from agent.obs import ACTIVE, ALIVE, HP, MAX_HP, MOVES_REVEALED, MOVES, PRESENT, REVEALED, SLEPT_BY_FOE, SPECIES, STATUS

N_OWN, N_OPP, N_GLOBAL = 4, 4, 4


class Senses:
    def __init__(self, mu: Matchups, path):
        assert has_hp_left(path), f"{path} has no hp_left (rebuild with tools/build_matchup_table.py)"
        self.mu = mu
        self.hp_left = torch.from_numpy(load_hp_left(path)).to(mu.device).reshape(-1)

    def lookup(self, A, B, fa, fb, sa, sb, ca, cb):
        """Value and A's expected HP fraction left (both broadcast to A's shape) for set A vs set B
        at HP fractions fa, fb (interpolated), status indices sa, sb, sleep-blocked flags ca, cb."""
        st = self.mu.st
        hi_a, lo_a, wa = hp_interp(fa)
        hi_b, lo_b, wb = hp_interp(fb)
        base = A * st[0] + B * st[1] + sa * st[4] + sb * st[5] + ca * st[6] + cb * st[7]
        val, hl = 0.0, 0.0
        for ha, xa in ((hi_a, 1 - wa), (lo_a, wa)):
            for hb, xb in ((hi_b, 1 - wb), (lo_b, wb)):
                idx = base + ha * st[2] + hb * st[3]
                f = xa * xb
                v = self.mu.value[idx].float()
                val = val + torch.where(torch.isfinite(v), v, torch.full_like(v, 0.5)) * f
                hl = hl + self.hp_left[idx].float() / 255 * f
        return val, hl

    def chain(self, A, fa, sa, ca, B, fb, sb, cb, ok_b, steps=6):
        """Probability that attacker A (shape S) beats every defender in B (shape S + (6,)), the
        defenders' side choosing the order adversarially. ok_b marks real defenders."""
        p = torch.ones_like(fa)
        f = fa.clone()
        left = ok_b.clone()
        for _ in range(steps):
            if not left.any():
                break
            v, hl = self.lookup(A[..., None], B, f[..., None], fb, sa[..., None], sb, ca[..., None], cb)
            v = torch.where(left, v, torch.full_like(v, 2.0))
            j = v.argmin(-1, keepdim=True)
            vj = v.gather(-1, j)[..., 0]
            hj = hl.gather(-1, j)[..., 0]
            any_left = left.any(-1)
            p = torch.where(any_left, p * vj.clamp(0, 1), p)
            f = torch.where(any_left, (hj / vj.clamp(min=1e-6)).clamp(0.02, 1.0), f)
            left = left & ~torch.nn.functional.one_hot(j[..., 0], 6).bool()
        return p

    @torch.no_grad()
    def features(self, team, side):
        mu = self.mu
        team = torch.as_tensor(np.asarray(team, np.int64) if not torch.is_tensor(team) else team, device=mu.device).long()
        me, opp = team[:, side], team[:, 1 - side]
        n = len(team)
        frac = lambda t: (t[..., HP].float() / t[..., MAX_HP].clamp(min=1).float()).clamp(0.02, 1.0)
        mine = mu.set_ids(me)
        A = mine.clamp(min=0)
        cand, w = mu.posterior(opp)                                            # (n, 6, K)
        sa, sb = status_options(me[..., STATUS])[0], status_options(opp[..., STATUS])[0]
        c_me = (opp[..., SLEPT_BY_FOE] > 0).any(-1).long()
        c_opp = (me[..., SLEPT_BY_FOE] > 0).any(-1).long()
        my_ok = (me[..., ALIVE] > 0) & (me[..., PRESENT] > 0) & (mine >= 0)
        seen = (opp[..., PRESENT] > 0) & (opp[..., REVEALED] > 0) & (opp[..., ALIVE] > 0)
        K = cand.shape[-1]
        # Own sweeps: attacker i (n, 6, K) over each opposing candidate set... the chain needs one
        # defender set per slot, so use each opposing Pokemon's most likely set.
        top = w.argmax(-1, keepdim=True)
        B_top = cand.gather(-1, top)[..., 0]                                   # (n, 6)
        sweep = self.chain(A, frac(me), sa, c_me[:, None].expand(-1, 6),
                           B_top[:, None].expand(-1, 6, -1), frac(opp)[:, None].expand(-1, 6, -1),
                           sb[:, None].expand(-1, 6, -1), c_opp[:, None, None].expand(-1, 6, 6),
                           seen[:, None].expand(-1, 6, -1))
        sweep = torch.where(my_ok & seen.any(-1, keepdim=True), sweep, torch.zeros_like(sweep))
        # Threats: attacker j with set candidate k (n, 6, K) against our standing team.
        Aj = cand
        threat_k = self.chain(Aj, frac(opp)[..., None].expand(-1, -1, K), sb[..., None].expand(-1, -1, K),
                              c_opp[:, None, None].expand(-1, 6, K),
                              A[:, None, None].expand(-1, 6, K, -1), frac(me)[:, None, None].expand(-1, 6, K, -1),
                              sa[:, None, None].expand(-1, 6, K, -1), c_me[:, None, None, None].expand(-1, 6, K, 6),
                              my_ok[:, None, None].expand(-1, 6, K, -1))
        threat = (threat_k * w).sum(-1) * seen
        # Predictability and concealment: the opponent's posterior over our sets.
        my_cand, my_w = mu.posterior(me)                                       # restricted to revealed moves
        is_true = (my_cand == A[..., None]).float()
        revealed = (me[..., REVEALED] > 0) & (me[..., PRESENT] > 0)
        predict = (my_w * is_true).sum(-1) * revealed
        f_me, f_opp = frac(me), frac(opp)
        v_true, _ = self.lookup(A[:, :, None], B_top[:, None], f_me[:, :, None], f_opp[:, None], sa[:, :, None], sb[:, None],
                                c_me[:, None, None], c_opp[:, None, None])     # (n, 6, 6)
        v_bel, _ = self.lookup(my_cand[:, :, None], B_top[:, None, :, None], f_me[:, :, None, None], f_opp[:, None, :, None],
                               sa[:, :, None, None], sb[:, None, :, None], c_me[:, None, None, None], c_opp[:, None, None, None])
        v_bel = (v_bel * my_w[:, :, None]).sum(-1)                             # (n, 6, 6)
        pair = (my_ok[:, :, None] & seen[:, None]).float()
        conceal = ((v_true - v_bel) * pair).sum(-1) / pair.sum(-1).clamp(min=1)
        conceal = conceal * revealed                                           # unrevealed: species unknown to them too
        own = torch.stack([sweep, predict, conceal, my_ok.float()], -1)
        oppf = torch.stack([threat, seen.float(), torch.zeros_like(threat), torch.zeros_like(threat)], -1)
        # Globals: sack gain and best switch-in against the opposing active.
        r = torch.arange(n, device=mu.device)
        ia, ib = me[..., ACTIVE].argmax(-1), opp[..., ACTIVE].argmax(-1)
        v_vs_act = v_true[r, :, ib]                                            # (n, 6)
        bench = my_ok & (me[..., ACTIVE] == 0)
        best = torch.where(bench, v_vs_act, torch.full_like(v_vs_act, -1.0)).max(-1).values
        has_bench = bench.any(-1)
        best = torch.where(has_bench, best, torch.zeros_like(best))
        sack = torch.where(has_bench, best - v_vs_act[r, ia], torch.zeros_like(best))
        unseen_opp = ((opp[..., PRESENT] > 0) & (opp[..., REVEALED] == 0)).float().sum(-1) / 6
        unrev_me = ((me[..., PRESENT] > 0) & (me[..., REVEALED] == 0)).float().sum(-1) / 6
        glob = torch.stack([sack, best, unseen_opp, unrev_me], -1)
        return {"own_senses": own, "opp_senses": oppf, "global_senses": glob}
