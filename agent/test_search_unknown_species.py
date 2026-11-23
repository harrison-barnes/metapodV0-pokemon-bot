"""Regression test: the play-time search must keep working when the opponent brings a species that
data/sets_ladder.json has no set for (on the ladder: Pikachu, Graveler, Metapod). Before the fix the set guess for
such a Pokemon was a pool Tauros, the engine refused the species change, and every later search of the battle failed.

    uv run python -m agent.test_search_unknown_species
"""
import gen1py  # noqa: F401
import numpy as np

from agent.obs import PRESENT, REVEALED, SPECIES
from agent.search import DEX, POOL, SPECIES_ID, fallback_set
from agent.search_exact import ExactSearchPlayer
from agent.selfplay import Runner

ODD = ["PIKACHU:THUNDERBOLT,SURF,THUNDER_WAVE,SEISMIC_TOSS", "GRAVELER:EARTHQUAKE,ROCK_SLIDE,EXPLOSION,BODY_SLAM",
       "METAPOD:HARDEN,TACKLE,STRING_SHOT"]


def main():
    pool_species = {s["species"] for s in POOL}
    assert not pool_species & {o.split(":")[0] for o in ODD}, "the test species must be outside the pool"
    fs = fallback_set(SPECIES_ID["PIKACHU"], [])
    assert fs["species"] == "PIKACHU" and 1 <= len(fs["moves"]) <= 4
    norm = lambda rng: [s["species"] + ":" + ",".join(s["moves"]) for s in
                        [POOL[i] for i in rng.choice(len(POOL), 6, replace=False)]]
    games = 4
    runner = Runner(games, 5, "cpu", norm)
    for i in range(games):       # side 1 leads with the odd species, so they are revealed from turn 1
        mine = norm(runner.rng)
        runner.env.reset(i, mine, ODD + norm(runner.rng)[:3])
    p = ExactSearchPlayer("checkpoints/spec2_00200.pt", "cpu", tau=0.02, infer=True, k=8)
    rng = np.random.default_rng(0)
    checked = 0
    for turn in range(60):
        team, battle, mask = runner.state()
        live = np.flatnonzero(runner.env.results() < 0)
        if not len(live):
            break
        acts = np.full((games, 2), 11, np.int64)
        for g in live:           # every guess keeps each revealed opponent's species
            opp = team[g, 1]
            for j, s in p.guess(opp):
                if opp[j, PRESENT] and opp[j, REVEALED]:
                    assert s.split(":")[0] == DEX["species"][int(opp[j, SPECIES])]["name"], (s, int(opp[j, SPECIES]))
                    checked += 1
        acts[live, 0] = p.actions(team[live], battle[live], mask[live], 0, runner.rng, env=runner.env, idx=live)
        m = mask[:, 1].copy()
        m[~m.any(-1), 11] = True
        acts[:, 1] = (rng.random(m.shape) * m).argmax(-1)
        runner.env.step(acts)
        p.observe(acts[live, 1], env=runner.env, idx=live)
    print(f"ok: {turn} turns searched without an error; {checked} revealed-species guesses checked")


if __name__ == "__main__":
    main()
