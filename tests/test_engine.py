"""The Rust engine's Python bindings: random 6v6 battles run to the end with legal actions only."""
import gen1py
import numpy as np

from agent.selfplay import sample_team


def test_random_battles_finish():
    rng = np.random.default_rng(0)
    n = 16
    env = gen1py.VecEnv(n, seed=3)
    for i in range(n):
        env.reset(i, sample_team(rng), sample_team(rng))
    for _ in range(1000):
        res = env.results()
        if (res >= 0).all():
            break
        mask = env.legal_mask()
        assert mask.shape == (n, 2, 12)
        m = mask.copy()
        m[~m.any(-1)] = False
        m[..., 11] |= ~mask.any(-1)
        acts = (rng.random(m.shape) * m).argmax(-1).astype(np.int64)
        env.step(acts)
    assert (env.results() >= 0).all()
    team = env.team_state()
    assert team.shape == (n, 2, 6, 18)
