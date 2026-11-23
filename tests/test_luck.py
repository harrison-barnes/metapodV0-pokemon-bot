"""Luck counts (tools/luck.py) from a Showdown battle log: trials, observed and expected events per side and kind."""
import gzip
import json

from tools.luck import counts as luck_counts


def test_luck_counts_from_a_log():
    import tools.luck as A
    rows = [json.loads(l) for l in gzip.open("tests/fixtures/stat_drops_150.jsonl.gz", "rt")]
    lines = [l for r in rows if r["game"] == 0 for l in r["lines"]]
    out = luck_counts(lines, "Player 1")
    assert out and all(k.split("_")[0] in ("us", "them") and k.split("_")[1] in A.KINDS for k in out)
    for n, obs, exp in out.values():
        assert 0 <= obs <= n and 0 <= exp <= n
