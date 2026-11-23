"""Loader and integrity checks for the 1v1 dataset written by `meta dataset` (docs/DATASET.md).

Usage: uv run analysis/dataset.py DIR      # prints a summary and runs the checks

    from analysis.dataset import load
    rows, sides, sets = load(DIR)   # rows: numpy structured array; sides: decoded battle fields
"""
import json
import sys
from pathlib import Path

import numpy as np


def row_dtype(schema):
    names, formats, offsets = [], [], []
    for name, off, ty, n in schema["row"]:
        names.append(name)
        formats.append((ty, (n,)) if n > 1 else ty)
        offsets.append(off)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": schema["row_bytes"]})


def load(path):
    path = Path(path)
    schema = json.load(open(path / "schema.json"))
    done = [json.loads(l) for l in open(path / "done.jsonl")]
    end = max((d["offset"] + d["rows"] * schema["row_bytes"] for d in done), default=0)
    rows = np.memmap(path / "rows.bin", dtype=row_dtype(schema), mode="r", shape=(end // schema["row_bytes"],))
    sets = json.load(open(path / "sets.json"))
    return rows, decode_sides(rows["battle"], schema), sets


def decode_sides(battle, schema):
    """Battle bytes -> dict of per-side fields, each shaped (rows, 2, ...)."""
    b = schema["battle"]
    size = b["side_bytes"]
    out = {}
    for name, off, ty, n in b["side"]:
        cols = []
        for s in range(2):
            start = s * size + off
            raw = np.ascontiguousarray(battle[:, start:start + np.dtype(ty).itemsize * n])
            cols.append(raw.view(ty).reshape(len(battle), n))
        out[name] = np.stack(cols, axis=1).squeeze(-1) if n == 1 else np.stack(cols, axis=1)
    return out


def check(rows, sides):
    n = len(rows)
    problems = []
    v = rows["value"].astype(np.float64)
    implied = rows["p_win"] + 0.5 * rows["p_double_ko"] + 0.5 * rows["p_endless"]
    if not ((v >= -1e-6) & (v <= 1 + 1e-6)).all():
        problems.append("values outside [0, 1]")
    gap = np.abs(implied - v)
    if gap.max() > 1e-3:
        problems.append(f"outcome split disagrees with value (max gap {gap.max():.2e})")
    mix = rows["mix"].reshape(n, 2, 4).astype(np.float64)
    if np.abs(mix.sum(axis=2) - 1).max() > 1e-4:
        problems.append("mixes don't sum to 1")
    # The value of the turn's matrix game under both mixes is the position's value.
    q = rows["q"].reshape(n, 4, 4).astype(np.float64)
    nash = np.einsum("ni,nij,nj->n", mix[:, 0], q, mix[:, 1])
    err = np.abs(nash - v)
    if err.max() > 1e-4:
        problems.append(f"mix0 . Q . mix1 differs from value (max {err.max():.2e})")
    hp = sides["hp"]
    if (hp == 0).any():
        problems.append("a labelled position has a fainted Pokémon")
    return problems


def main():
    rows, sides, sets = load(sys.argv[1])
    kinds = np.bincount(rows["kind"], minlength=2)
    pairs = len({(int(a), int(b)) for a, b in zip(rows["a"], rows["b"])})
    print(f"{len(rows):,} rows from {pairs} set pairs: {kinds[0]:,} entry positions, {kinds[1]:,} mid-battle")
    prec = {}
    for b, q in zip(rows["buckets"], rows["hp_levels"]):
        prec[(int(b), int(q))] = prec.get((int(b), int(q)), 0) + 1
    print("precision (buckets, HP levels):", dict(sorted(prec.items(), key=lambda kv: -kv[1])))
    print(f"value: mean {rows['value'].mean():.3f}; double KO in {np.mean(rows['p_double_ko'] > 1e-3):.1%} of positions; "
          f"median length {np.median(rows['turns']):.1f} turns")
    problems = check(rows, sides)
    print("checks:", "all passed" if not problems else "; ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
