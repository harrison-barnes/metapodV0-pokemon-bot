"""Ground-truth test of the Showdown adapter's Gen 1 stat penalties (OurPlayer._track_stats): replay the protocol
lines of random Showdown battles through the tracker and compare the Attack and Speed it implies for each active
Pokemon with Showdown's own modifiedStats (tools/showdown/gen1_stat_drops.js).

    (cd tools/showdown && npm install && node gen1_stat_drops.js 300 > ../../drops.jsonl)
    uv run --group showdown python -m agent.test_stat_drops drops.jsonl
"""
import collections
import json
import sys

from agent.showdown_player import STAGE, OurPlayer, Tracker


def implied(stored, boost, n, div):
    v = min(999, max(1, stored * STAGE[boost + 6][0] // STAGE[boost + 6][1]))
    for _ in range(n):
        v = max(1, v // div)
    return v


def main(path, old_rule=False):
    tr, game = None, None
    n = bad = 0
    kinds = collections.Counter()
    examples = []
    for row in map(json.loads, open(path)):
        if row["game"] != game:
            tr, game = Tracker(), row["game"]
        for line in row["lines"]:
            m = line.split("|")
            if len(m) >= 3:
                OurPlayer._track_stats(tr, m[1], m)
        for a in row["actives"]:
            if a["transformed"]:
                continue
            par_q, brn_h = tr.drops.get(a["key"], [0, 0])
            if old_rule:      # the adapter before the fix: one penalty while the status lasts
                par_q, brn_h = int(a["status"] == "par"), int(a["status"] == "brn")
            got = [implied(a["stored"][0], a["boosts"][0], brn_h, 2), implied(a["stored"][1], a["boosts"][1], par_q, 4)]
            for i, name in enumerate(("atk", "spe")):
                n += 1
                if got[i] != a["mod"][i]:
                    bad += 1
                    kinds[(name, "too high" if got[i] > a["mod"][i] else "too low")] += 1
                    if len(examples) < 5:
                        examples.append((row["game"], row["turn"], a["key"], name, got[i], a["mod"][i], a["status"], a["boosts"]))
    print(f"{'old rule' if old_rule else 'tracker'}: {bad} of {n} stat checks differ from Showdown ({bad / n:.2%}) {dict(kinds)}")
    for e in examples:
        print("  game %d turn %d %s %s: ours %d, Showdown %d (status %r, boosts %s)" % e)
    return bad


if __name__ == "__main__":
    main(sys.argv[1], old_rule=True)
    sys.exit(1 if main(sys.argv[1]) else 0)
