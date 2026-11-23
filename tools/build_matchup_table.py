"""Packs the dataset's fresh-entry rows into a lookup table of exact 1v1 matchup values, for the
6v6 agent's observations (agent/matchups.py).

For every ordered pair of sets (a on side 0, b on side 1) and every fresh-entry root on the grid
- HP fraction 1, 0.75, 0.5, 0.25, 0.1 for each side,
- status none, PAR, BRN, PSN, asleep 2 turns, asleep 5 turns for each side,
- each side's sleep moves blocked by Sleep Clause or not,
it stores a's equilibrium value (win + half of double KOs and endless games) and a's equilibrium
move mix over its four move slots, plus a stall flag. Rows with Freeze Clause active are skipped (freezes are rare
enough to ignore here). Each row fills both orientations: (b, a) gets 1 - value and side 1's mix.
The stall flag marks roots where equilibrium play runs long (over 64 expected turns, or more than 1%
endless games). The solver doesn't track PP, so there the value may be wrong: in the real game a
healing move can run out (docs/ENGINE.md).
The engine merges roots that play out identically, so a Sleep Clause flag that makes no difference
(the side has no sleep move) only appears unset; those cells are copied from the unset one. Cells
for a status the type is immune to (never looked up) are copied from no status, and an unsolved
pair's cells stay NaN.

It also stores hp_left: a's expected HP fraction when the 1v1 ends (0 if it fainted), x255, which
agent/senses.py chains into multi-Pokemon estimates. Directories are read in order and a later one
overwrites an earlier one's cells for the same pairs, so the healing-PP re-solve (data/dataset_pp/*)
goes last to replace the stall pairs' cells.

Usage: uv run python -m tools.build_matchup_table OUT.npz DATASET_DIR [DATASET_DIR ...]
"""
import sys

import numpy as np

from agent.features import MAX_HP
from analysis.dataset import load

HP_GRID = np.array([1.0, 0.75, 0.5, 0.25, 0.1])
SLEEP_BLOCKED, FREEZE_BLOCKED = 1, 2


def status_index(status):
    """Status byte -> grid index: 0 none, 1 PAR, 2 BRN, 3 PSN, 4 asleep <= 3 turns, 5 asleep longer.
    Freeze (no grid point) maps to 5: a frozen Pokemon rarely acts again."""
    slp = status & 7
    return np.select([status & 64 > 0, status & 16 > 0, status & 8 > 0, status & 32 > 0, (slp > 0) & (slp <= 3), slp > 3],
                     [1, 2, 3, 5, 4, 5], 0)


def hp_index(frac):
    return np.abs(np.asarray(frac)[..., None] - HP_GRID).argmin(-1)


def main():
    out_path, dirs = sys.argv[1], sys.argv[2:]
    n_sets = None
    for d in dirs:
        rows, sides, sets = load(d)
        if n_sets is None:
            n_sets = len(sets)
            shape = (n_sets, n_sets, 5, 5, 6, 6, 2, 2)
            value = np.full(shape, np.nan, np.float16)
            mix = np.zeros(shape + (4,), np.uint8)
            stall = np.zeros(shape, np.uint8)
            hp_left = np.zeros(shape, np.uint8)
            solved = np.zeros((n_sets, n_sets), bool)
        for k in range(0, len(rows), 2_000_000):
            r = rows[k:k + 2_000_000]
            r = r[r["kind"] == 0]
            s = {f: sides[f][k:k + 2_000_000][rows["kind"][k:k + 2_000_000] == 0] for f in ("hp", "species", "status", "clauses")}
            keep = (s["clauses"] & FREEZE_BLOCKED).sum(1) == 0
            r, s = r[keep], {f: v[keep] for f, v in s.items()}
            a, b = r["a"].astype(np.int64), r["b"].astype(np.int64)
            h = hp_index(s["hp"] / MAX_HP[s["species"]])
            st = status_index(s["status"])
            c = (s["clauses"] & SLEEP_BLOCKED).astype(np.int64)
            v = r["value"]
            value[a, b, h[:, 0], h[:, 1], st[:, 0], st[:, 1], c[:, 0], c[:, 1]] = v
            value[b, a, h[:, 1], h[:, 0], st[:, 1], st[:, 0], c[:, 1], c[:, 0]] = 1 - v
            solved[a, b] = solved[b, a] = True
            long = ((r["turns"] > 64) | (r["p_endless"] > 0.01)).astype(np.uint8)
            stall[a, b, h[:, 0], h[:, 1], st[:, 0], st[:, 1], c[:, 0], c[:, 1]] = long
            stall[b, a, h[:, 1], h[:, 0], st[:, 1], st[:, 0], c[:, 1], c[:, 0]] = long
            hl = np.round(np.clip(r["hp_left"], 0, 1) * 255).astype(np.uint8)
            hp_left[a, b, h[:, 0], h[:, 1], st[:, 0], st[:, 1], c[:, 0], c[:, 1]] = hl[:, 0]
            hp_left[b, a, h[:, 1], h[:, 0], st[:, 1], st[:, 0], c[:, 1], c[:, 0]] = hl[:, 1]
            ch = r["choices"].reshape(-1, 2, 4)
            mx = r["mix"].reshape(-1, 2, 4)
            for side, (x, y, hx, hy, sx, sy, cx, cy) in enumerate([(a, b, 0, 1, 0, 1, 0, 1), (b, a, 1, 0, 1, 0, 1, 0)]):
                slot_mix = np.zeros((len(r), 4), np.float32)
                for col in range(4):
                    slot = ch[:, side, col]
                    ok = slot < 4
                    slot_mix[ok, slot[ok]] += mx[ok, side, col]
                mix[x, y, h[:, hx], h[:, hy], st[:, sx], st[:, sy], c[:, cx], c[:, cy]] = np.round(slot_mix * 255).astype(np.uint8)
        print(f"{d}: {solved.sum() // 2:,} set pairs so far", flush=True)
    for fill in ((6, 0), (7, 0), (4, 0), (5, 0)):           # (axis, index to copy from)
        axis, src = fill
        missing = ~np.isfinite(value)
        value = np.where(missing, np.take(value, [src], axis=axis), value)
        mix = np.where(missing[..., None], np.take(mix, [src], axis=axis), mix)
        stall = np.where(missing, np.take(stall, [src], axis=axis), stall)
        hp_left = np.where(missing, np.take(hp_left, [src], axis=axis), hp_left)
    filled = np.isfinite(value[solved]).mean()
    print(f"{solved.sum():,} of {n_sets * n_sets:,} ordered pairs solved; {filled:.1%} of their cells filled")
    print(f"stall-flagged cells: {stall[solved].mean():.2%}")
    np.savez(out_path, value=value, mix=mix, stall=stall, hp_left=hp_left, solved=solved, hp_grid=HP_GRID)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
