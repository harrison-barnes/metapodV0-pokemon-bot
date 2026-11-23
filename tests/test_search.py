"""The exact one-turn search (agent/search_exact.py) and its set guesser (agent/search.py)."""


def test_unknown_species(run):
    # opponents bringing species with no set in the pool (the fallback_set fix)
    assert "without an error" in run("-m", "agent.test_search_unknown_species")


def test_search_options(run):
    # rounds / budget / calibration / Bayesian root / opp_tau / pp_cost options, all off by default
    assert "ok" in run("-m", "agent.next_search_test").splitlines()


def test_play_one_game(run, tmp_path):
    out = run("scripts/play_local.py", "--games", "1", "--opponent", "random", "--max-turns", "25", "--out", str(tmp_path / "g.log"))
    assert "turn   1:" in out and "score:" in out
