"""Moveset pool for the 1v1 dataset, from Smogon ladder usage statistics (see docs/DATASET.md).

The usage file gives each move's frequency per Pokémon but not whole movesets. Moves are treated
as independently included: a move with frequency f contributes odds f / (1 - f), and the most
likely four-move sets are the ones with the largest product of odds. Every set is checked against
the learnset, and Smogon's bans (Evasion and OHKO Clauses, Mew, Mewtwo) are applied.

Species with usage below the cutoff get the heuristic set from `meta pools --rules smogon`, at a
small weight, so the evaluator still sees every legal Pokémon.

Usage: uv run tools/ladder_sets.py data/smogon/gen1ou-1760-2026-09.json HEURISTIC.json \
           --out data/sets_ladder.json
"""
import argparse
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED_MOVES = {"DOUBLE_TEAM", "MINIMIZE", "FISSURE", "HORN_DRILL", "GUILLOTINE"}
BANNED_SPECIES = {"MEW", "MEWTWO"}
# Showdown ids that differ from the cartridge's move names once punctuation is dropped.
RENAMES = {"psychic": "PSYCHIC_M", "highjumpkick": "HI_JUMP_KICK", "visegrip": "VICEGRIP"}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def sets_wanted(usage):
    if usage >= 0.20:
        return 5
    if usage >= 0.05:
        return 4
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usage")
    ap.add_argument("heuristic", help="sets from `meta pools --rules smogon`, used for unlisted species")
    ap.add_argument("--min-usage", type=float, default=0.005)
    ap.add_argument("--min-move", type=float, default=0.02, help="ignore moves rarer than this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dex = json.load(open(ROOT / "data/gen1.json"))
    move_by_id = {norm(m["name"]): m for m in dex["moves"] if m}
    move_by_id.update({k: move_by_id[norm(v)] for k, v in RENAMES.items()})
    species = {norm(s["name"]): s for s in dex["species"]}
    stats = json.load(open(args.usage))
    pool, skipped = [], []

    for sid, entry in sorted(stats["data"].items(), key=lambda kv: -kv[1]["usage"]):
        usage = entry["usage"]
        sp = species[norm(sid)]
        if usage < args.min_usage or sp["name"] in BANNED_SPECIES:
            continue
        counts = {k: v for k, v in entry["Moves"].items() if k}
        weight = sum(entry["Moves"].values()) / 4  # every Pokémon has 4 slots ('' = empty)
        learn = set(sp["learnset"])
        freq = {}
        for mid, c in counts.items():
            m = move_by_id.get(mid)
            if m is None:
                skipped.append((sp["name"], mid, "unknown move"))
            elif m["id"] not in learn or m["name"] in BANNED_MOVES:
                skipped.append((sp["name"], m["name"], "illegal here"))
            elif c / weight >= args.min_move:
                freq[m["name"]] = min(c / weight, 0.999)
        size = min(4, len(freq))
        odds = {m: math.log(f / (1 - f)) for m, f in freq.items()}
        ranked = sorted(itertools.combinations(sorted(freq), size), key=lambda s: -sum(odds[m] for m in s))
        top = ranked[: sets_wanted(usage)]
        # Share of this species' usage each kept set represents (normalized over the kept sets).
        scores = [math.exp(sum(odds[m] for m in s)) for s in top]
        for s, sc in zip(top, scores):
            pool.append({"species": sp["name"], "moves": sorted(s, key=lambda m: -freq[m]),
                         "weight": usage * sc / sum(scores), "source": "ladder"})

    listed = {p["species"] for p in pool}
    for h in json.load(open(args.heuristic)):
        if h["species"] not in listed and h["species"] not in BANNED_SPECIES:
            pool.append({**h, "weight": args.min_usage / 10, "source": "heuristic"})

    json.dump(pool, open(args.out, "w"), indent=1)
    n_ladder = sum(p["source"] == "ladder" for p in pool)
    print(f"{len(pool)} sets: {n_ladder} from the ladder ({len(listed)} species), "
          f"{len(pool) - n_ladder} heuristic; wrote {args.out}")
    for s, m, why in skipped:
        print(f"  skipped {s} {m}: {why}")


if __name__ == "__main__":
    main()
