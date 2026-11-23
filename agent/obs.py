"""Observations for the 6v6 agent: what one side can see, as a Showdown player would.

Twelve team tokens (own six, then the opponent's six) plus the two active Pokemon's volatile state.
- Own team: everything (species, moves, PP, HP, status).
- Opposing team: a Pokemon is visible once it has been sent out. Until then its token is
  "unknown". Its moves are visible once used. The rest are filled in with the probability of
  each move for that species, from the ladder set pool (the usage-weighted prior, PRIOR). To keep
  host-to-device copies small, observations carry move ids; Policy builds the move vectors.
- Hidden for both sides, as in game: sleep and confusion counters, Disable and trapping turns
  left. Also hidden for the opponent: exact HP (only the fraction shows), Substitute HP and
  Bide damage. PP left on the opponent's revealed moves is shown only to networks built with
  opp_pp (Policy(opp_pp=True)); a player can count it, but older networks never saw it.

Team state columns come from gen1py.VecEnv.team_state (TEAM_FIELDS = 18).
"""
import json
from pathlib import Path

import numpy as np

from agent.features import N_MOVES, N_SPECIES, side_scalars
from analysis.dataset import decode_sides

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.load(open(ROOT / "data/dataset_smogon_v1/schema.json"))
SPECIES, MOVES, HP, MAX_HP, STATUS, PP, REVEALED, MOVES_REVEALED, ACTIVE, ALIVE, SLEPT_BY_FOE, PRESENT = (
    0, slice(1, 5), 5, 6, 7, slice(8, 12), 12, 13, 14, 15, 16, 17)
UNKNOWN_SPECIES = N_SPECIES + 1  # token species index: 0 empty slot, 1..151 species, 152 unknown
MAX_PP = np.zeros(N_MOVES, np.float32)
for m in json.load(open(ROOT / "data/gen1.json"))["moves"]:
    if m:
        MAX_PP[m["id"]] = m["pp"] // 5 * 8
# side_scalars columns hidden in game (see agent/features.py for the layout).
HIDDEN_BOTH = [27, 52, 55, 56]          # sleep counter, confusion counter, Disable turns, trapping turns
HIDDEN_OPP = HIDDEN_BOTH + [1, 57, 58]   # + exact HP, Substitute HP, Bide damage
TOKEN_SCALARS = 17


def move_prior(pool_path=ROOT / "data/sets_ladder.json"):
    """P(species has move) from the usage-weighted set pool: (N_SPECIES, N_MOVES)."""
    dex = json.load(open(ROOT / "data/gen1.json"))
    sp = {s["name"]: i for i, s in enumerate(dex["species"])}
    mv = {m["name"]: m["id"] for m in dex["moves"] if m}
    prior = np.zeros((N_SPECIES, N_MOVES), np.float32)
    total = np.zeros(N_SPECIES, np.float32)
    for s in json.load(open(pool_path)):
        i = sp[s["species"]]
        total[i] += s["weight"]
        for m in s["moves"]:
            prior[i, mv[m]] += s["weight"]
    return prior / np.maximum(total, 1e-12)[:, None]


PRIOR = move_prior()


def status_onehot(status):
    return np.stack([status == 0, (status & 8) > 0, (status & 16) > 0, (status & 32) > 0, (status & 64) > 0, (status & 7) > 0], -1)


def observe(team, battle, side, opp_pp=False):
    """Observation for `side` of every env. team: (n, 2, 6, 18) int; battle: (n, 102) uint8.
    opp_pp: also show the PP left on the opponent's revealed moves (a player can count each use;
    the Showdown adapter does, agent/showdown_player.py). Off for networks trained without it."""
    n = len(team)
    me, opp = team[:, side], team[:, 1 - side]
    species = np.zeros((n, 12), np.int64)
    move_ids = np.zeros((n, 12, 4), np.int64)
    move_known = np.zeros((n, 12, 4), np.float32)
    prior_species = np.zeros((n, 12), np.int64)   # row of PRIOR for an opposing token, else 0
    use_prior = np.zeros((n, 12), np.float32)
    scal = np.zeros((n, 12, TOKEN_SCALARS), np.float32)
    rows = np.arange(n)[:, None]
    for half, t in ((0, me), (1, opp)):
        own = half == 0
        present = t[..., PRESENT] > 0
        seen = present & ((t[..., REVEALED] > 0) | own)
        sl = slice(6 * half, 6 * half + 6)
        species[:, sl] = np.where(seen, t[..., SPECIES] + 1, np.where(present, UNKNOWN_SPECIES, 0))
        ids = t[..., MOVES]                                        # (n, 6, 4)
        if own:
            known = ids > 0
        else:
            known = ((t[..., MOVES_REVEALED][..., None] >> np.arange(4)) & 1).astype(bool) & (ids > 0) & seen[..., None]
        # The network builds each token's move vector on the device (Policy.move_vectors): known
        # moves one-hot, plus for a seen opponent its usage prior over the moves still unknown.
        move_ids[:, sl] = np.where(known, ids, 0)
        move_known[:, sl] = known
        if not own:
            prior_species[:, sl] = np.where(seen, t[..., SPECIES], 0)
            use_prior[:, sl] = seen
        hp = t[..., HP] / np.maximum(t[..., MAX_HP], 1)
        pp = t[..., PP] / np.maximum(MAX_PP[ids], 1)
        s = scal[:, sl]
        s[..., 0] = present
        s[..., 1] = own
        s[..., 2] = np.where(seen, hp, 1.0)
        s[..., 3:9] = status_onehot(t[..., STATUS]) * seen[..., None]
        s[..., 9] = t[..., ACTIVE]
        s[..., 10] = np.where(present, t[..., ALIVE], 0)
        s[..., 11] = seen
        s[..., 12:16] = pp * (ids > 0) * (own or (opp_pp and known))
        s[..., 16] = t[..., SLEPT_BY_FOE] * seen
    sides = decode_sides(battle, SCHEMA)
    vol = np.stack([side_scalars(sides, side), side_scalars(sides, 1 - side)], 1)  # own, opp
    vol[:, 0, HIDDEN_BOTH] = 0
    vol[:, 1, HIDDEN_OPP] = 0
    active = me[..., ACTIVE].argmax(-1)
    opp_active = opp[..., ACTIVE].argmax(-1)                       # the opposing active is always visible
    act_moves = me[rows[:, 0], active][:, MOVES]                   # (n, 4)
    act_pp = me[rows[:, 0], active][:, PP] / np.maximum(MAX_PP[act_moves], 1)
    return {
        "species": species, "move_ids": move_ids, "move_known": move_known, "prior_species": prior_species,
        "use_prior": use_prior, "scalars": scal, "volatile": vol.astype(np.float32),
        "active": active.astype(np.int64), "opp_active": opp_active.astype(np.int64), "act_moves": act_moves.astype(np.int64), "act_pp": act_pp.astype(np.float32),
    }
