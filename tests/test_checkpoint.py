"""The shipped MetapodV0 checkpoint: architecture and exact parameter counts (README), and one forward pass."""
import numpy as np
import torch

from agent.arena import load_policy
from agent.selfplay import Runner, observe_on

CK, CRITIC = "checkpoints/spec2_00200.pt", "checkpoints/spec2_00200_critic.pt"


def n_params(path):
    return sum(v.numel() for v in torch.load(path, map_location="cpu", weights_only=True)["policy"].values())


def test_architecture_and_counts():
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    assert ck["arch"]["d"] == 256 and ck["arch"]["layers"] == 4 and ck["arch"]["heads"] == 8 and ck["iter"] == 200
    assert n_params(CK) == 4_236_549
    cr = torch.load(CRITIC, map_location="cpu", weights_only=True)
    assert cr["privileged"] and cr["arch"]["qhead"]
    assert n_params(CRITIC) == 4_877_448


def test_forward_pass():
    policy = load_policy(CK, "cpu")
    runner = Runner(4, 0, "cpu")
    team, battle, mask = runner.state()
    m = torch.from_numpy(mask[:, 0].copy())
    with torch.no_grad():
        logits, value = policy(observe_on(policy, team, battle, 0, "cpu"), m)
    assert logits.shape == (4, 12)
    assert torch.isfinite(value).all() and (value.abs() <= 1).all()
    best = logits.argmax(-1).numpy()
    assert mask[np.arange(4), 0, best].all()
