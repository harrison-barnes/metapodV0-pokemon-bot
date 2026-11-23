"""Turns dataset rows (raw 1v1 battle positions) into network inputs and targets.

Every field the engine keeps about a side is encoded, so the input is lossless: identity (species,
types, the four moves, the move data left in RAM, the stale selection) as embedding indices, and
everything numeric (HP, stats, stages, status, the three battle-status bytes as bits, counters)
as scaled scalars. Targets come straight from the solver's labels.

Positions are symmetric: swapping the sides swaps every input and maps the targets (value -> 1 -
value, win <-> lose, Q -> 1 - Q transposed, mixes swapped). `swap_sides` implements this for
training-time augmentation.
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEX = json.load(open(ROOT / "data/gen1.json"))
N_SPECIES = len(DEX["species"])
N_MOVES = len(DEX["moves"])  # index 0 = no move
N_TYPES = len(DEX["types"])
NO_CHOICE, UNUSED = 254, 255


def max_hp_table(level=100, dv=15, stat_exp=65535):
    """CalcStat for HP (home/move_mon.asm) for every species."""
    b = min(255, int(np.ceil(np.sqrt(stat_exp))))
    return np.array([((s["base"]["hp"] + dv) * 2 + b // 4) * level // 100 + level + 10 for s in DEX["species"]], np.float32)


MAX_HP = max_hp_table()
# Move power, accuracy and type, for the move data each side has in RAM.
MOVE_INFO = np.zeros((N_MOVES, 3), np.float32)
for m in DEX["moves"]:
    if m:
        MOVE_INFO[m["id"]] = [m["power"] / 255, m["accuracy"] / 255, m["type"]]

SCALARS = 66  # per side; see side_scalars


def side_scalars(sides, s):
    """Numeric features of side s: (rows, SCALARS) float32."""
    f = lambda name: sides[name][:, s].astype(np.float32)
    species = sides["species"][:, s].astype(np.int64)
    max_hp = MAX_HP[species]
    hp = f("hp")
    stats, unmod = f("stats"), np.maximum(f("unmod"), 1)
    status = sides["status"][:, s]
    slp = status & 7
    bits = lambda name: np.unpackbits(sides[name][:, s:s + 1].astype(np.uint8), axis=1, bitorder="little").astype(np.float32)
    disabled = sides["disabled"][:, s]
    parts = [
        (hp / max_hp)[:, None], (hp / 700)[:, None], (max_hp / 700)[:, None],
        stats / 500, stats / unmod, unmod / 500,                       # 12
        (f("mods") - 7) / 6,                                          # 6
        np.stack([status == 0, (status & 8) > 0, (status & 16) > 0, (status & 32) > 0, (status & 64) > 0, slp > 0], 1).astype(np.float32),
        (slp / 7)[:, None],                                           # 7
        bits("bs1"), bits("bs2"), bits("bs3"),                        # 24
        (f("confused") / 7)[:, None], (f("toxic") / 16)[:, None],
        ((disabled >> 4) / 4)[:, None], ((disabled & 15) / 8)[:, None],
        (f("attacks_left") / 5)[:, None], (f("sub_hp") / 256)[:, None], (f("bide_dmg") / 1000)[:, None],
        np.unpackbits(sides["clauses"][:, s:s + 1].astype(np.uint8), axis=1, bitorder="little")[:, :3].astype(np.float32),
        MOVE_INFO[sides["ram"][:, s, 0], :2], (sides["ram"][:, s, 1:2] / 255).astype(np.float32),
        (sides["ram"][:, s, 3:4] / 255).astype(np.float32),
    ]
    out = np.concatenate(parts, axis=1).astype(np.float32)
    assert out.shape[1] == SCALARS, out.shape
    return out


def encode(rows, sides, damage_offset=100):
    """Inputs for a batch of rows: index tensors and scalars, sides stacked on dim 1."""
    sc = np.stack([side_scalars(sides, 0), side_scalars(sides, 1)], axis=1)
    # wDamage, shared by both sides (offset from schema.json "battle.damage").
    battle = np.ascontiguousarray(rows["battle"])
    damage = battle[:, damage_offset:damage_offset + 2].view("<u2")[:, 0]
    shared = (damage.astype(np.float32) / 1000)[:, None]
    return {
        "species": torch.from_numpy(sides["species"].astype(np.int64)),             # (B, 2)
        "types": torch.from_numpy(sides["types"].astype(np.int64)),                 # (B, 2, 2)
        "moves": torch.from_numpy(sides["moves"].astype(np.int64)),                 # (B, 2, 4)
        "ram_move": torch.from_numpy(sides["ram"][:, :, 0].astype(np.int64)),       # (B, 2)
        "selected": torch.from_numpy(np.where(sides["selected"] < N_MOVES, sides["selected"], 0).astype(np.int64)),
        "scalars": torch.from_numpy(sc),                                            # (B, 2, SCALARS)
        "shared": torch.from_numpy(shared),                                         # (B, 1)
        # Which move slot each matrix row/column is: 0-3 a slot, 4 Struggle, 5 no choice, 6 unused.
        "choice": torch.from_numpy(choice_codes(rows["choices"].reshape(-1, 2, 4))),  # (B, 2, 4)
    }


def choice_codes(ch):
    out = np.where(ch == NO_CHOICE, 5, np.where(ch == UNUSED, 6, ch)).astype(np.int64)
    return np.minimum(out, 6)


def targets(rows):
    win, dko, endless = (rows[k].astype(np.float32) for k in ("p_win", "p_double_ko", "p_endless"))
    lose = np.clip(1 - win - dko - endless, 0, 1)
    outcome = np.stack([win, lose, dko, endless], 1)
    return {
        "value": torch.from_numpy(rows["value"].astype(np.float32)),
        "outcome": torch.from_numpy(outcome / outcome.sum(1, keepdims=True)),
        "hp_left": torch.from_numpy(rows["hp_left"].astype(np.float32)),
        "turns": torch.from_numpy(np.log1p(np.minimum(rows["turns"].astype(np.float32), 1e4))),
        "q": torch.from_numpy(rows["q"].reshape(-1, 4, 4).astype(np.float32)),
        "mix": torch.from_numpy(rows["mix"].reshape(-1, 2, 4).astype(np.float32)),
    }


def swap_sides(x, y):
    """The same positions with the sides exchanged (inputs and targets)."""
    flip = lambda t: t.flip(1)
    x = {k: (flip(v) if k not in ("shared",) else v) for k, v in x.items()}
    o = y["outcome"]
    y = {
        "value": 1 - y["value"],
        "outcome": torch.stack([o[:, 1], o[:, 0], o[:, 2], o[:, 3]], 1),
        "hp_left": y["hp_left"].flip(1),
        "turns": y["turns"],
        "q": 1 - y["q"].transpose(1, 2),
        "mix": y["mix"].flip(1),
    }
    return x, y
