"""Exact 1v1 matchup features for the 6v6 agent, from the solver's table (tools/build_matchup_table.py).

For one side's view of every env:
- mu_value (n, 6, 6): own Pokemon i's equilibrium 1v1 value against opposing Pokemon j if both
  came in fresh now, at their current HP and status and with the current Sleep Clause state.
- mu_stall (n, 6, 6): posterior probability that the 1v1 is a long stall, where the value is
  unreliable (the solver ignores PP; tools/build_matchup_table.py).
- mu_known (n, 6, 6): 1 where that value means something (both alive, j seen, the pair solved).
- mu_mix (n, 4): own active's equilibrium move mix (per move slot) against the opposing active.
- mu_mix_known (n,): 1 where mu_mix means something.

Hidden information is respected:
- An opposing Pokemon counts only once it has been seen.
- Its set is a posterior over the ladder sets of its species: usage weight, restricted to sets
  that contain every move it has revealed. Values and mixes are posterior averages.
- Sleep counters are hidden, so a sleeping Pokemon averages the "2 turns left" and "5 turns
  left" grid points.
HP is interpolated linearly between grid points (opposing HP is visible as a fraction in game).

The table lives on the network's device and the lookups run there: a rollout step needs about
two million of them.
"""
import json
from pathlib import Path

import numpy as np
import torch

from agent.obs import ACTIVE, ALIVE, HP, MAX_HP, MOVES, MOVES_REVEALED, PRESENT, REVEALED, SLEPT_BY_FOE, SPECIES, STATUS

ROOT = Path(__file__).resolve().parent.parent


def table_path(path):
    """A checkpoint's matchup table path, usable on this machine: as stored if it exists, else the
    file of that name under data/ (older checkpoints store an absolute path from another machine)."""
    p = Path(path)
    return p if p.exists() else ROOT / "data" / p.name


def _hp_left_file(path):
    """The matchup table ships as two compressed files, to keep each under GitHub's 50 MB guideline: PATH (value, mix,
    stall, solved, hp_grid) and its sibling `<stem>_hp_left.npz` (hp_left, read by agent/senses.py). A single file
    that holds hp_left itself works too."""
    p = Path(path)
    return p if "hp_left" in np.load(p).files else p.with_name(p.stem + "_hp_left.npz")


def has_hp_left(path):
    f = _hp_left_file(path)
    return f.is_file() and "hp_left" in np.load(f).files


def load_hp_left(path):
    return np.load(_hp_left_file(path))["hp_left"]


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
GRID = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.1])


def hp_interp(frac):
    """Grid indices (hi, lo) and the weight on lo for linear interpolation in HP fraction."""
    grid = GRID.to(frac.device)
    f = frac.clamp(float(grid[-1]), 1.0)
    lo = (grid[None] > f.reshape(-1, 1)).sum(-1).reshape(f.shape).clamp(1, len(grid) - 1)  # first grid point <= f
    hi = lo - 1
    w = (grid[hi] - f) / (grid[hi] - grid[lo])
    return hi, lo, w


def status_options(status):
    """Two grid status indices per Pokemon (equal unless asleep, when both sleep points count)."""
    base = torch.zeros_like(status)
    for bit, idx in ((32, 5), (8, 3), (16, 2), (64, 1)):     # later assignments win: PAR > BRN > PSN > FRZ
        base = torch.where(status & bit > 0, idx, base)
    asleep = (status & 7) > 0
    return torch.where(asleep, 4, base), torch.where(asleep, 5, base)


class Matchups:
    def __init__(self, device, path=ROOT / "data/matchup_table.npz", pool=ROOT / "data/sets_ladder.json"):
        z = np.load(path)
        self.device = device
        self.value = torch.from_numpy(z["value"]).to(device).reshape(-1)
        self.mix = torch.from_numpy(z["mix"]).to(device).reshape(-1, 4)
        self.stall = torch.from_numpy(z["stall"]).to(device).reshape(-1)
        self.solved = torch.from_numpy(z["solved"]).to(device)
        shape = z["value"].shape
        self.st = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
        dex = json.load(open(ROOT / "data/gen1.json"))
        sp = {s["name"]: i for i, s in enumerate(dex["species"])}
        mv = {m["name"]: m["id"] for m in dex["moves"] if m}
        sets = json.load(open(pool))
        by_species = {}
        for k, s in enumerate(sets):
            by_species.setdefault(sp[s["species"]], []).append(k)
        self.K = max(len(v) for v in by_species.values())
        n_species = len(dex["species"])
        cand = np.zeros((n_species, self.K), np.int64)          # candidate set ids, padded with 0
        cand_w = np.zeros((n_species, self.K), np.float32)      # usage weights, 0 for padding
        cand_moves = np.full((n_species, self.K, 4), -1, np.int64)
        for s, ks in by_species.items():
            for c, k in enumerate(ks):
                cand[s, c], cand_w[s, c] = k, sets[k]["weight"]
                ids = [mv[m] for m in sets[k]["moves"]]
                cand_moves[s, c, :len(ids)] = ids
        to = lambda a: torch.from_numpy(a).to(device)
        self.cand, self.cand_w, self.cand_moves = to(cand), to(cand_w), to(cand_moves)
        # Sorted moves (padding as 0, like the env's empty slots) to identify a team member's set.
        self.cand_sorted = to(np.sort(np.where(cand_moves < 0, 0, cand_moves), -1))

    def set_ids(self, t):
        """Exact set index of each team member; -1 for an empty slot or a set not in the pool."""
        sp = t[..., SPECIES]
        match = (self.cand_sorted[sp] == t[..., MOVES].sort(-1).values[..., None, :]).all(-1) & (self.cand_w[sp] > 0)
        return torch.where(match.any(-1) & (t[..., PRESENT] > 0), self.cand[sp].gather(-1, match.long().argmax(-1, keepdim=True))[..., 0], -1)

    def posterior(self, opp):
        """Candidate sets (n, 6, K) and normalised weights for each opposing Pokemon."""
        sp = opp[..., SPECIES]
        cand, w = self.cand[sp], self.cand_w[sp]
        revealed = ((opp[..., MOVES_REVEALED][..., None] >> torch.arange(4, device=opp.device)) & 1).bool() & (opp[..., MOVES] > 0)
        moves = torch.where(revealed, opp[..., MOVES], -2)                                       # (n, 6, 4)
        has = (moves[:, :, None, :, None] == self.cand_moves[sp][:, :, :, None, :]).any(-1)      # (n, 6, K, 4)
        consistent = (has | (moves[:, :, None, :] < 0)).all(-1)
        w2 = w * consistent
        w = torch.where(w2.sum(-1, keepdim=True) > 0, w2, w)
        return cand, w / w.sum(-1, keepdim=True).clamp(min=1e-12)

    @torch.no_grad()
    def features(self, team, side):
        """team: (n, 2, 6, TEAM_FIELDS) integer array or tensor. Returns float tensors on the device."""
        team = torch.as_tensor(np.asarray(team, np.int64) if not torch.is_tensor(team) else team, device=self.device).long()
        me, opp = team[:, side], team[:, 1 - side]
        n = len(team)
        st = self.st
        mine = self.set_ids(me)                                                                  # (n, 6)
        cand, w = self.posterior(opp)                                                            # (n, 6, K)
        frac = lambda t: t[..., HP].float() / t[..., MAX_HP].clamp(min=1).float()
        hi_a, lo_a, wa = hp_interp(frac(me))
        hi_b, lo_b, wb = hp_interp(frac(opp))
        sa, sb = status_options(me[..., STATUS]), status_options(opp[..., STATUS])
        c_me = (opp[..., SLEPT_BY_FOE] > 0).any(-1).long()                                       # my sleep moves blocked
        c_opp = (me[..., SLEPT_BY_FOE] > 0).any(-1).long()
        A = mine.clamp(min=0)
        base = (A * st[0])[:, :, None, None] + (cand * st[1])[:, None] + (c_me * st[6] + c_opp * st[7])[:, None, None, None]
        val = torch.zeros(n, 6, 6, self.K, device=self.device)
        stall = torch.zeros(n, 6, 6, self.K, device=self.device)
        for ha, fa in ((hi_a, 1 - wa), (lo_a, wa)):
            for hb, fb in ((hi_b, 1 - wb), (lo_b, wb)):
                f = (0.25 * fa[:, :, None] * fb[:, None, :])[..., None]
                for s1 in sa:
                    for s2 in sb:
                        off = (ha * st[2] + s1 * st[4])[:, :, None] + (hb * st[3] + s2 * st[5])[:, None, :]
                        val += self.value[base + off[..., None]].float() * f
                        stall += self.stall[base + off[..., None]].float() * f
        solved = self.solved[A[:, :, None, None], cand[:, None]]
        val = torch.where(torch.isfinite(val) & solved, val, 0.5)
        value = (val * w[:, None]).sum(-1)
        stall = (stall * w[:, None]).sum(-1)
        pair_ok = (solved * w[:, None]).sum(-1) > 0.99
        seen = (opp[..., PRESENT] > 0) & (opp[..., REVEALED] > 0)
        known = (mine >= 0)[:, :, None] & (me[..., ALIVE] > 0)[:, :, None] & (seen & (opp[..., ALIVE] > 0))[:, None] & pair_ok
        # Own active's move mix against the opposing active (nearest HP grid point, posterior-averaged).
        r = torch.arange(n, device=self.device)
        ia, ib = me[..., ACTIVE].argmax(-1), opp[..., ACTIVE].argmax(-1)
        ha = torch.where(wa[r, ia] > 0.5, lo_a[r, ia], hi_a[r, ia])
        hb = torch.where(wb[r, ib] > 0.5, lo_b[r, ib], hi_b[r, ib])
        idx = (A[r, ia] * st[0])[:, None] + cand[r, ib] * st[1] + (ha * st[2] + hb * st[3] + sa[0][r, ia] * st[4]
                                                                  + sb[0][r, ib] * st[5] + c_me * st[6] + c_opp * st[7])[:, None]
        mix = (self.mix[idx].float() / 255 * w[r, ib][..., None]).sum(1)                         # (n, 4)
        mk = known[r, ia, ib]
        return {"mu_value": value * known, "mu_stall": stall * known, "mu_known": known.float(), "mu_mix": mix * mk[:, None], "mu_mix_known": mk.float()}
