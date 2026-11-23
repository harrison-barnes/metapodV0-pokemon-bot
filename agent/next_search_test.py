"""Checks of agent/search_exact.py's play-time options.

1. With default options the search is the same player as the pre-change file (pass --orig PATH to a copy of the
   old agent/search_exact.py): same choices, same payoff matrices, the same mix to 1e-5 (the reply prior is now
   accumulated in float64), on a batch of positions with history.
2. rounds=2 at k=4 uses 8 worlds; budget stops within its limit (after the first round) and reports the time.
3. calib maps critic leaves monotonically and leaves terminal values alone; opp_eps floors the reply prior.

    uv run python -m agent.next_search_test [--orig /path/to/old/search_exact.py]
"""
import argparse
import importlib.util
import time

import gen1py  # noqa: F401
import numpy as np
import torch

from agent.next_search_bench import positions
from agent.search_exact import ExactSearchPlayer

CK = "checkpoints/spec2_00200.pt"


def run(cls, env, ok, team, mask, **kw):
    p = cls(CK, "cpu", tau=0.02, **kw)
    p.trace = []
    out = [int(p._search(team[[g]], mask[[g]], 0, env, [int(g)])[0]) for g in ok]
    return p, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig")
    a = ap.parse_args()
    torch.set_num_threads(1)
    env, ok = positions(CK, "cpu", 16, 5)
    ok = ok[:6]
    team, mask = env.team_state().astype(np.int16), env.legal_mask()
    p1, c1 = run(ExactSearchPlayer, env, ok, team, mask, k=4)
    if a.orig:
        spec = importlib.util.spec_from_file_location("search_exact_orig", a.orig)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        for kw in ({"k": 4}, {"k": 3, "depth": 2, "width": 3, "width2": 2}):   # (rollouts sample from torch's global RNG: not repeatable)
            pn, cn = run(ExactSearchPlayer, env, ok, team, mask, **kw)
            po, co = run(mod.ExactSearchPlayer, env, ok, team, mask, **kw)
            assert cn == co, (kw, cn, co)
            assert all(np.array_equal(x[3], y[3]) and np.allclose(x[4], y[4], atol=1e-5) for x, y in zip(pn.trace, po.trace)), kw
            print(f"1. same as the old file with {kw}: choices {cn}")
    p2, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, rounds=2)
    assert p2.last["worlds"] == 8
    pb, _ = run(ExactSearchPlayer, env, ok[:1], team, mask, k=4, budget=0.01)
    assert pb.last["worlds"] == 4, pb.last["worlds"]
    t0 = time.perf_counter()
    pb, _ = run(ExactSearchPlayer, env, ok[:2], team, mask, k=4, budget=1.5)
    per = (time.perf_counter() - t0) / 2
    print(f"2. rounds=2 -> 8 worlds; budget 1.5 s -> {pb.last['worlds']} worlds in {pb.last['secs']:.2f} s (call {per:.2f} s)")
    assert pb.last["secs"] < 1.5 * 1.3 and pb.last["worlds"] > 4
    pc, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, calib={"a": -0.17, "b": 1.58})
    v = np.linspace(-0.99, 0.99, 41).astype(np.float32)
    c = pc._cal(v)
    assert (np.diff(c) > 0).all() and abs(c[20] - (2 / (1 + np.exp(0.17)) - 1)) < 1e-5 and (np.abs(c) <= 1).all()
    q0, qc = np.concatenate([x[3].ravel() for x in p1.trace]), np.concatenate([x[3].ravel() for x in pc.trace])
    print(f"3. calib: root |Q| mean {np.abs(q0).mean():.3f} -> {np.abs(qc).mean():.3f}")
    pe, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, opp_eps=0.1, opp_width=12)
    assert all(x[6].min() >= 0.1 / len(x[6]) - 1e-9 for x in pe.trace)
    assert max(len(x[6]) for x in pe.trace) >= max(len(x[6]) for x in p1.trace)
    print("   opp_eps floors the reply prior; opp_width widens the reply menu:", [len(x[6]) for x in p1.trace], "->", [len(x[6]) for x in pe.trace])
    # 4. Bayesian root game: on fully revealed opponents (oracle: one type) it is the plain solve; with hidden sets
    #    it reports >= 1 types and a proper mix.
    po, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, oracle=True, hidden=False)
    pq, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, oracle=True, hidden=False, bayes=True)
    assert all(np.allclose(x[4], y[4], atol=1e-6) for x, y in zip(po.trace, pq.trace))
    pbz, _ = run(ExactSearchPlayer, env, ok, team, mask, k=8, bayes=True)
    ty = []
    for g in ok:
        pbz._search(team[[g]], mask[[g]], 0, env, [int(g)])
        ty.append(pbz.last["types"])
        assert abs(pbz.last["mix"].sum() - 1) < 1e-9 and abs(pbz.last["y"].sum() - 1) < 1e-6 and abs(pbz.last["value"]) <= 1
    print("4. bayes: one type = the plain solve; opponent types per position at k=8:", ty)
    from agent.search_exact import _bayes_game
    # a phantom column: 1 guess in 8 has a reply that punishes row 0 hard; the plain solve abandons row 0, the Bayesian one keeps it
    Qa, Qb = np.array([[0.3, 0.3], [0.1, 0.1]]), np.array([[0.3, -0.9], [0.1, 0.1]])
    ones = np.ones((2, 2))
    types = [[7, 7 * Qa * [1, 0], 7 * ones * [1, 0], np.array([7.0, 0])], [1, Qb, ones, np.array([0.5, 0.5])]]
    xb, yb, vb = _bayes_game(types, np.array([True, True]), np.array([0.5, 0.5]), 0.02)
    from agent.qsolve import regularized_game
    xp = regularized_game(np.array([[0.3, -0.9], [0.1, 0.1]]), np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.02)
    assert xb[0] > 0.9 and xp[0] < 0.5, (xb, xp)
    print(f"   phantom reply held by 1 guess in 8: plain solve plays row 0 with {xp[0]:.2f}, Bayesian {xb[0]:.2f} (value {vb:.3f})")
    # 5. opp_tau: equal to tau it is the plain solve; large, the reply mix stays at its prior.
    pt, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, opp_tau=0.02)
    assert all(np.allclose(x[4], y[4], atol=1e-3) for x, y in zip(p1.trace, pt.trace))
    ph, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, opp_tau=100.0)
    ph._search(team[[ok[0]]], mask[[ok[0]]], 0, env, [int(ok[0])])
    assert np.allclose(ph.last["y"], ph.last["py"], atol=1e-2)
    print("5. opp_tau = tau is the plain solve; opp_tau large keeps the reply at its prior")
    # 6. pp_cost charges moves (rows < 4) by 1 / PP left and leaves switches alone.
    pp, _ = run(ExactSearchPlayer, env, ok, team, mask, k=4, pp_cost=0.04)
    for x, y in zip(p1.trace, pp.trace):
        d = x[3].mean(1) - y[3].mean(1)
        assert (d[x[1] >= 4] == 0).all() and (d[x[1] < 4] > 0).all() and d.max() <= 0.04 + 1e-9, d
    print("6. pp_cost lowers move rows only, by at most pp_cost")
    print("ok")


if __name__ == "__main__":
    main()
