"""Differential test of the 6v6 engine against Pokemon Showdown's gen1ou format.

Random teams of six different species are drawn from the ladder set pool (data/sets_ladder.json,
weighted by usage). Both simulators play each pair of teams many times with the same random
policy (`sim6`; tools/showdown/random_battles6.js), and the win rates should agree within
sampling error.

Usage: uv run tools/diffcheck6.py [--teams 60] [--battles 2000] [--procs 4] [--seed 1] [--showdown-dir DIR]
(--showdown-dir: e.g. the Showdown checkout of a local server, to test against the build you play on.)
"""
import argparse
import json
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OURS = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=60)
    ap.add_argument("--battles", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--switch-prob", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "agent/runs/diffcheck6.json"))
    ap.add_argument("--showdown-dir", default=str(ROOT / "tools/showdown"),
                    help="directory with random_battles6.js and the pokemon-showdown package to test against")
    args = ap.parse_args()
    pool = json.load(open(ROOT / "data/sets_ladder.json"))
    rng = random.Random(args.seed)

    def team():
        out, used = [], set()
        while len(out) < 6:
            s = rng.choices(pool, weights=[0 if {"DIG", "FLY"} & set(p["moves"]) else p["weight"] for p in pool])[0]
            if s["species"] not in used:
                used.add(s["species"])
                out.append({"species": s["species"], "moves": s["moves"]})
        return out

    jobs = [{"id": k, "a": team(), "b": team(), "n": args.battles, "seed": args.seed * 1000 + k, "p": args.switch_prob}
            for k in range(args.teams)]

    def showdown(chunk):
        inp = "\n".join(json.dumps(j) for j in chunk) + "\n"
        out = subprocess.run(["node", "random_battles6.js"], input=inp, capture_output=True, text=True,
                             cwd=args.showdown_dir, check=True).stdout
        return [json.loads(l) for l in out.splitlines() if l.strip()]

    spec = lambda t: ";".join(m["species"] + ":" + ",".join(m["moves"]) for m in t)

    def ours(job):
        out = subprocess.run([str(ROOT / "engine/target/release/sim6"), spec(job["a"]), spec(job["b"]),
                              "--battles", str(OURS), "--seed", str(job["seed"]), "--switch-prob", str(job["p"])],
                             capture_output=True, text=True, check=True).stdout.split()
        w0, w1, d, turns = int(out[3]), int(out[5]), int(out[7]), float(out[10])
        return (w0 + 0.5 * d) / (w0 + w1 + d), turns

    chunks = [jobs[i::args.procs] for i in range(args.procs)]
    with ThreadPoolExecutor(args.procs) as ex:
        sd = {r["id"]: r for chunk in ex.map(showdown, chunks) for r in chunk}
    with ThreadPoolExecutor(args.procs) as ex:
        mine = dict(zip([j["id"] for j in jobs], ex.map(ours, jobs)))

    rows = []
    for j in jobs:
        r = sd[j["id"]]
        n = sum(r["wins"]) + r["draws"]
        p_sd = (r["wins"][0] + 0.5 * r["draws"]) / n
        p_us, t_us = mine[j["id"]]
        se = math.sqrt(max(p_sd * (1 - p_sd), 1e-4) / n + max(p_us * (1 - p_us), 1e-4) / OURS)
        rows.append({**j, "showdown": p_sd, "ours": p_us, "z": (p_us - p_sd) / se,
                     "turns_showdown": r["turns"], "turns_ours": t_us})
    rows.sort(key=lambda r: -abs(r["z"]))
    json.dump(rows, open(args.out, "w"), indent=1)
    zs = [r["z"] for r in rows]
    print(f"{len(rows)} team pairs x {args.battles} Showdown battles ({OURS} ours)")
    print(f"mean |diff| {sum(abs(r['ours'] - r['showdown']) for r in rows) / len(rows):.4f}, "
          f"|z|>3: {sum(abs(z) > 3 for z in zs)}, |z|>2: {sum(abs(z) > 2 for z in zs)} "
          f"(expect ~{0.046 * len(rows):.1f} by chance)")
    mt = lambda k: sum(r[k] for r in rows) / len(rows)
    print(f"mean battle length: ours {mt('turns_ours'):.1f}, Showdown {mt('turns_showdown'):.1f} turns")
    for r in rows[:8]:
        print(f"z={r['z']:+6.2f} ours {r['ours']:.3f} showdown {r['showdown']:.3f}  "
              f"{[m['species'] for m in r['a']]} vs {[m['species'] for m in r['b']]}")


if __name__ == "__main__":
    main()
