"""Differential test: our engine vs Pokemon Showdown's Gen 1 simulator under uniformly random play.

Random matchups (fully evolved Pokemon, random legal movesets) are played many times in both
simulators with each side choosing uniformly among its legal moves. If the mechanics agree, win
rates match within sampling error.

Usage: uv run tools/diffcheck.py [--matchups 160] [--battles 3000] [--procs 4] [--seed 1]
"""
import argparse
import json
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNSUPPORTED = {"METRONOME", "MIRROR_MOVE", "MIMIC", "STRUGGLE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchups", type=int, default=160)
    ap.add_argument("--battles", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "agent/runs/diffcheck.json"))
    args = ap.parse_args()
    data = json.load(open(ROOT / "data/gen1.json"))
    moves = {m["id"]: m["name"] for m in data["moves"] if m}
    roster = [s for s in data["species"] if s["fully_evolved"]]
    rng = random.Random(args.seed)

    def random_set():
        s = rng.choice(roster)
        pool = [moves[m] for m in s["learnset"] if moves[m] not in UNSUPPORTED]
        return {"species": s["name"], "moves": rng.sample(pool, min(4, len(pool)))}

    jobs = [{"id": k, "a": random_set(), "b": random_set(), "n": args.battles, "seed": args.seed * 1000 + k}
            for k in range(args.matchups)]

    def showdown(chunk):
        inp = "\n".join(json.dumps(j) for j in chunk) + "\n"
        out = subprocess.run(["node", "random_battles.js"], input=inp, capture_output=True, text=True,
                             cwd=ROOT / "tools/showdown", check=True).stdout
        return [json.loads(l) for l in out.splitlines() if l.strip()]

    def ours(job):
        label = lambda m: m["species"] + ":" + ",".join(m["moves"])
        out = subprocess.run([str(ROOT / "engine/target/release/gen1"), "sim", label(job["a"]), label(job["b"]),
                              "--battles", "100000", "--seed", str(job["seed"]), "--no-sleep-clause"],
                             capture_output=True, text=True, check=True).stdout
        last = out.strip().splitlines()[-1].split()
        w0, w1, d = int(last[6]), int(last[8]), int(last[10])
        return (w0 + 0.5 * d) / (w0 + w1 + d)

    chunks = [jobs[i::args.procs] for i in range(args.procs)]
    with ThreadPoolExecutor(args.procs) as ex:
        sd = {r["id"]: r for chunk in ex.map(showdown, chunks) for r in chunk}
    with ThreadPoolExecutor(4) as ex:
        mine = dict(zip([j["id"] for j in jobs], ex.map(ours, jobs)))

    rows = []
    for j in jobs:
        r = sd[j["id"]]
        n = sum(r["wins"]) + r["draws"]
        p_sd = (r["wins"][0] + 0.5 * r["draws"]) / n
        p_us = mine[j["id"]]
        se = math.sqrt(max(p_sd * (1 - p_sd), 1e-4) / n + max(p_us * (1 - p_us), 1e-4) / 100000)
        rows.append({**j, "showdown": p_sd, "ours": p_us, "z": (p_us - p_sd) / se, "turns": r["turns"]})
    rows.sort(key=lambda r: -abs(r["z"]))
    json.dump(rows, open(args.out, "w"), indent=1)
    zs = [r["z"] for r in rows]
    print(f"{len(rows)} matchups x {args.battles} Showdown battles")
    print(f"mean |diff| {sum(abs(r['ours'] - r['showdown']) for r in rows) / len(rows):.4f}, "
          f"|z|>3: {sum(abs(z) > 3 for z in zs)}, |z|>2: {sum(abs(z) > 2 for z in zs)} (expect ~{0.046 * len(rows):.0f} by chance)")
    for r in rows[:15]:
        print(f"z={r['z']:+6.2f} ours {r['ours']:.3f} showdown {r['showdown']:.3f}  "
              f"{r['a']['species']} {r['a']['moves']} vs {r['b']['species']} {r['b']['moves']}")


if __name__ == "__main__":
    main()
