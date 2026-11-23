"""Scores and paired differences from the paired harness's rows (agent/next_eval_box.py -> games.jsonl).

    uv run python -m agent.next_eval_analyze agent/runs/next_ab1/games.jsonl [more files] [--ref live] [--json out.json]

For every arm: games, score (a tie counts half), a 95% Wilson interval, the same per opponent, plus health counts
(stalled pairs, adapter errors, search fallbacks, seconds per decision, turns, record in games of 80+ turns).

For every arm against the reference arm (--ref, default the first arm seen), three estimates of the difference in score:
- matched: battles are matched on (opponent, replica, index) - the same opponent process type, the same team of ours
  (the shared schedule), the same minutes on the same box. The mean of the within-match differences, with a 95%
  interval from a bootstrap that resamples whole (opponent, replica) units (the unit is the cluster: its battles share
  a pair of processes and a schedule), and a two-sided sign-flip p-value at the unit level; with fewer than 10 units
  the interval is the wider of that and a bootstrap over single pairs;
- stratified: all battles (matched or not), difference of mean scores within each (opponent, our team) cell, cells
  weighted by their harmonic sample size; normal interval from the binomial variances;
- per opponent: the matched difference inside each opponent, same bootstrap.
The matched interval is the one to quote; the stratified one is a check that uses unmatched games too. Battle
outcomes are mostly luck given the teams, so matching removes little variance (the printed "pair correlation" says how
much); the design's value is that both arms share the box, the opponents, the teams and the hour.
Arms are each compared with the reference, so with k arms expect about k/20 intervals to miss zero by chance.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def score(r):
    return 0.5 if r["won"] is None else float(r["won"])


def wilson(s, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = s / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def matched(G, arm, ref, opps=None, B=10000, seed=0):
    """Matched difference arm - ref over cells (opp, rep, idx) present in both, clustered by (opp, rep)."""
    units = defaultdict(list)
    for (a, o, rep, i), y in G.items():
        if a == arm and (opps is None or o in opps) and (ref, o, rep, i) in G:
            units[(o, rep)].append((y, G[(ref, o, rep, i)]))
    if not units:
        return None
    S = np.array([sum(a - b for a, b in v) for v in units.values()])     # per-unit sum of differences
    N = np.array([len(v) for v in units.values()])
    est = S.sum() / N.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(S), size=(B, len(S)))
    boot = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    D = np.array([a - b for v in units.values() for a, b in v])            # with few units the unit bootstrap is too
    if len(S) < 10:                                                         # narrow: take the wider of it and a
        pb = D[rng.integers(len(D), size=(min(B, 4000), len(D)))].mean(1)   # bootstrap over single pairs
        plo, phi = np.quantile(pb, [0.025, 0.975])
        lo, hi = min(lo, plo), max(hi, phi)
    flips = rng.choice([-1.0, 1.0], size=(B, len(S)))
    p = float((np.abs((flips * S).sum(1)) >= abs(S.sum()) - 1e-12).mean())
    ya = np.array([a for v in units.values() for a, _ in v])
    yb = np.array([b for v in units.values() for _, b in v])
    corr = float(np.corrcoef(ya, yb)[0, 1]) if ya.std() > 0 and yb.std() > 0 else float("nan")
    return {"diff": float(est), "lo": float(lo), "hi": float(hi), "p": p, "pairs": int(N.sum()), "units": len(S),
            "arm": float(ya.mean()), "ref": float(yb.mean()), "pair_corr": corr}


def stratified(rows, arm, ref):
    cells = defaultdict(lambda: {arm: [], ref: []})
    for r in rows:
        if r["arm"] in (arm, ref):
            cells[(r["opp"], r.get("team"))][r["arm"]].append(score(r))
    num = den = var = 0.0
    for c in cells.values():
        a, b = np.array(c[arm]), np.array(c[ref])
        if len(a) < 2 or len(b) < 2:
            continue
        w = len(a) * len(b) / (len(a) + len(b))
        num += w * (a.mean() - b.mean())
        den += w
        var += w * w * (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if den == 0:
        return None
    d, se = num / den, math.sqrt(var) / den
    return {"diff": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "se": se}


LUCK = [f"{s}_{k}" for s in ("us", "them") for k in ("crit", "miss", "fpar", "frz", "par")]


def luck_adjust(rows):
    """Luck-adjusted scores: row index -> score - beta . x, where x = observed - expected counts of the battle's chance
    events (ten kinds: each side's crits, misses, full paralyses, freezes, paralyses from secondary effects) and beta is
    the least-squares fit of the score on x within (arm, opponent) cells. x has mean zero under any play, so the
    adjustment removes luck without moving an arm's expected score. Returns (adjusted, mean adjustment per arm, R^2)."""
    idx = [i for i, r in enumerate(rows) if isinstance(r.get("luck"), dict)]
    if len(idx) < 200:
        return None
    X = np.array([[rows[i]["luck"].get(k, [0, 0, 0])[1] - rows[i]["luck"].get(k, [0, 0, 0])[2] for k in LUCK] for i in idx])
    y = np.array([score(rows[i]) for i in idx])
    cell = [(rows[i]["arm"], rows[i]["opp"]) for i in idx]
    Xc, yc = X.copy(), y.copy()
    for c in set(cell):
        m = np.array([x == c for x in cell])
        Xc[m] -= X[m].mean(0)
        yc[m] -= y[m].mean()
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    adj = {i: float(y[j] - X[j] @ beta) for j, i in enumerate(idx)}
    arms = sorted({rows[i]["arm"] for i in idx})
    shift = {a: float(np.mean([X[j] @ beta for j, i in enumerate(idx) if rows[i]["arm"] == a])) for a in arms}
    return adj, shift, float(1 - ((yc - Xc @ beta) ** 2).sum() / (yc ** 2).sum()), dict(zip(LUCK, beta.round(4).tolist()))


BLOCKS = {"metamon": ["TaurosV0", "Kakuna", "TaurosEnsemble", "KakunaEnsemble"], "standins": ["bc", "bcoff", "zx"],
          "foulplay": ["fp200hum", "fp1000hum", "fp200mir", "fp1000mir"]}
# agent/runs/g_final2, arm SF (spec2@200, search_teams_v1, tau .02, --infer): the .642 the live bot was chosen on
G_FINAL2 = {"TaurosV0": (129, 200), "Kakuna": (133, 200), "KakunaEnsemble": (64, 100), "TaurosEnsemble": (59.5, 100)}


def analyze(rows, ref=None):
    stalls = defaultdict(int)
    for r in rows:
        if r.get("stalled") and r.get("role", "arm") == "arm":
            stalls[r["arm"]] += 1
    opp_err = defaultdict(dict)                      # our own stand-in opponents' adapter errors (cumulative per process)
    for r in rows:
        if r.get("role") == "opp" and "won" in r:
            opp_err[r["opp"] if False else r.get("team", "?").split("/")[0]][r.get("me")] = max(
                opp_err[r.get("team", "?").split("/")[0]].get(r.get("me"), 0), (r.get("errors") or 0) + (r.get("fallbacks") or 0))
    rows = [r for r in rows if r.get("role", "arm") == "arm" and "won" in r]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    opps = list(dict.fromkeys(r["opp"] for r in rows))
    ref = ref or (arms[0] if arms else None)
    G = {(r["arm"], r["opp"], r["rep"], r["idx"]): score(r) for r in rows}
    out = {"ref": ref, "arms": {}, "vs_ref": {}, "opps": opps,
           "opp_errors": {t: sum(v.values()) for t, v in opp_err.items() if sum(v.values())}}
    la = luck_adjust(rows)
    if la:
        GA = {(rows[i]["arm"], rows[i]["opp"], rows[i]["rep"], rows[i]["idx"]): v for i, v in la[0].items()}
        out["luck"] = {"shift": la[1], "r2": la[2], "beta": la[3]}
    for a in arms:
        ra = [r for r in rows if r["arm"] == a]
        s, n = sum(map(score, ra)), len(ra)
        by = {}
        for o in opps:
            ro = [r for r in ra if r["opp"] == o]
            if ro:
                so = sum(map(score, ro))
                by[o] = {"n": len(ro), "score": so / len(ro), "ci": wilson(so, len(ro))}
        long = [r for r in ra if r["turns"] >= 80]
        last = {}
        for r in ra:                                  # errors / fallbacks are cumulative per process: take each unit's last
            k = (r["opp"], r["rep"], r.get("me"))
            last[k] = (max(last.get(k, (0, 0))[0], r.get("errors", 0)), max(last.get(k, (0, 0))[1], r.get("fallbacks", 0)))
        out["arms"][a] = {"n": n, "score": s / n, "ci": wilson(s, n), "by_opp": by, "ties": sum(r["won"] is None for r in ra),
                          "macro": float(np.mean([v["score"] for v in by.values()])),
                          "stalls": stalls[a], "errors": sum(v[0] for v in last.values()), "fallbacks": sum(v[1] for v in last.values()),
                          "mean_s": float(np.mean([r["mean_s"] for r in ra if r.get("mean_s") is not None] or [float("nan")])),
                          "max_s": max([r["max_s"] for r in ra if r.get("max_s") is not None] or [float("nan")]),
                          "turns": float(np.mean([r["turns"] for r in ra])),
                          "by_len": {name: {"n": len(v), "score": (sum(map(score, v)) / len(v)) if v else None}
                                     for name, v in (("<40", [r for r in ra if r["turns"] < 40]),
                                                     ("40-79", [r for r in ra if 40 <= r["turns"] < 80]),
                                                     ("80+", long))},
                          "long": {"n": len(long), "score": (sum(map(score, long)) / len(long)) if long else None},
                          "teams": {t: sum(1 for r in ra if r.get("team") == t) for t in sorted({str(r.get("team")) for r in ra})}}
        if a != ref:
            out["vs_ref"][a] = {"matched": matched(G, a, ref), "stratified": stratified(rows, a, ref),
                                "matched_luck": matched(GA, a, ref) if la else None,
                                "by_opp_luck": {o: matched(GA, a, ref, {o}) for o in opps} if la else {},
                                "by_opp": {o: matched(G, a, ref, {o}) for o in opps}}
    return out


def pct(x):
    return "  n/a" if x is None or x != x else f"{100 * x:5.1f}"


def report(res):
    L = []
    L.append(f"reference arm: {res['ref']}")
    if res.get("opp_errors"):
        L.append(f"WARNING: stand-in opponent processes had decision errors (they then play a fallback move, so that opponent "
                 f"was not what its name says), by the opponent's team pool: {res['opp_errors']}")
    L.append(f"{'arm':14s} {'games':>6s} {'score':>6s}  95% interval   macro  " + "  ".join(f"{o[:10]:>10s}" for o in res["opps"]))
    for a, v in res["arms"].items():
        L.append(f"{a:14s} {v['n']:6d} {pct(v['score'])}  [{pct(v['ci'][0])},{pct(v['ci'][1])}]  {pct(v['macro'])}  "
                 + "  ".join(f"{pct(v['by_opp'][o]['score']) if o in v['by_opp'] else '     '}/{v['by_opp'][o]['n'] if o in v['by_opp'] else 0:<4d}" for o in res["opps"]))
    L.append("")
    L.append("health: " + "; ".join(f"{a}: stalls {v['stalls']}, adapter errors {v['errors']}, search fallbacks {v['fallbacks']}, "
                                    f"{v['mean_s']:.2f}s/decision (max {v['max_s']}), {v['turns']:.0f} turns, 80+ turns "
                                    f"{pct(v['long']['score'])}% of {v['long']['n']}" for a, v in res["arms"].items()))
    L.append("score by game length (turns; length is an outcome of play, so compare shares as well as scores): " + "; ".join(
        f"{a}: " + ", ".join(f"{k} {pct(b['score']).strip()}% of {b['n']}" for k, b in v["by_len"].items()) for a, v in res["arms"].items()))
    if res.get("luck"):
        L.append(f"luck adjustment (chance events' observed - expected, fitted within arm x opponent): explains {100 * res['luck']['r2']:.0f}% of result "
                 "variance; mean luck per arm in points (should be near 0): " + ", ".join(f"{a} {100 * x:+.1f}" for a, x in res["luck"]["shift"].items()))
    for a, v in res["vs_ref"].items():
        m, s = v["matched"], v["stratified"]
        L.append("")
        if m:
            L.append(f"{a} - {res['ref']}: matched {100 * m['diff']:+.1f} points [{100 * m['lo']:+.1f}, {100 * m['hi']:+.1f}], p={m['p']:.3f} "
                     f"({m['pairs']} pairs in {m['units']} units; {pct(m['arm'])} vs {pct(m['ref'])}; pair correlation {m['pair_corr']:.2f})")
        ml = v.get("matched_luck")
        if ml:
            L.append(f"{'':{len(a) + len(res['ref']) + 5}s}luck-adjusted {100 * ml['diff']:+.1f} points [{100 * ml['lo']:+.1f}, {100 * ml['hi']:+.1f}] ({ml['pairs']} pairs)")
        if s:
            L.append(f"{'':{len(a) + len(res['ref']) + 5}s}stratified {100 * s['diff']:+.1f} points [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}]")
        for o, mo in v["by_opp"].items():
            if mo:
                L.append(f"    vs {o:12s} {100 * mo['diff']:+6.1f} [{100 * mo['lo']:+.1f}, {100 * mo['hi']:+.1f}]  ({mo['pairs']} pairs, {mo['units']} units)")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--json")
    ap.add_argument("--all", action="store_true", help="one table over every opponent instead of one per block")
    args = ap.parse_args()
    rows = load(args.files)
    res = analyze(rows, args.ref)
    if args.all:
        print(report(res))
    else:                                            # the two blocks are judged separately
        seen = {r.get("opp") for r in rows}
        other = sorted(o for o in seen if o and all(o not in b for b in BLOCKS.values()))   # e.g. "Kakuna@fix2"
        for name, opps in list(BLOCKS.items()) + ([("other", other)] if other else []):
            sub = [r for r in rows if r.get("opp") in opps]
            if any("won" in r for r in sub):
                print(f"=== block: {name} ===")
                rb = analyze(sub, args.ref)
                print(report(rb))
                if name == "metamon" and rb["ref"] in rb["arms"]:
                    by = rb["arms"][rb["ref"]]["by_opp"]
                    print("reference arm vs g_final2 (spec2@200, same protocol): " + "; ".join(
                        f"{o} {100 * by[o]['score']:.1f}% of {by[o]['n']} (then {100 * w / n:.1f}% of {n})" for o, (w, n) in G_FINAL2.items() if o in by))
                    if all(o in by for o in G_FINAL2):
                        mix = sum(by[o]["score"] * n for o, (w, n) in G_FINAL2.items()) / 600
                        print(f"  at g_final2's opponent mix (2:2:1:1): {100 * mix:.1f}% (then 64.2%)")
                print()
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
