"""The Showdown adapter (agent/showdown_player.py), offline: no server is contacted.

The fixtures were generated with the Showdown simulator (npm package pokemon-showdown, tools/showdown):
    cd tools/showdown && npm install
    node gen1_stat_drops.js 150 | gzip > ../../tests/fixtures/stat_drops_150.jsonl.gz
    node gen1_truth.js --games 40 --seed 3 | gzip > ../../tests/fixtures/truth_exotic_40.jsonl.gz
"""
import gzip
import shutil


def test_cases(run):
    assert "17 of 17 cases pass" in run("-m", "agent.test_adapter_cases")


def test_stat_drops(run, tmp_path):
    raw = tmp_path / "drops.jsonl"
    with gzip.open("tests/fixtures/stat_drops_150.jsonl.gz", "rb") as f, open(raw, "wb") as g:
        shutil.copyfileobj(f, g)
    assert "tracker: 0 of" in run("-m", "agent.test_stat_drops", str(raw))


def test_state_rebuild_against_showdown(run):
    # "exotic" OU-legal teams that lean on rare mechanics; the known residual gaps (Mimic / Transform / Rage corner
    # cases, docs in agent/showdown_player.py) stay below 0.5% of decisions
    out = run("-m", "agent.test_adapter_truth", "tests/fixtures/truth_exotic_40.jsonl.gz", "--max-rate", "0.005")
    assert "rebuild errors: 0" in out
