"""Play MetapodV0 in the Rust engine, no server needed, and print a turn-by-turn log.

MetapodV0 (side 0) is the spec2@200 policy with the exact one-turn search (tau 0.02, --infer, K = 8 guessed sets) and
one of its two searched teams (data/search_teams_v1). The opponent (side 1) brings a random team drawn from the
ladder set pool and is, by default, the same network without search (its most likely action).

    uv run python scripts/play_local.py                        # one game, log to stdout
    uv run python scripts/play_local.py --games 4 --opponent smart --out examples/demo.log
    uv run python scripts/play_local.py --opponent checkpoints/spec2_00200.pt --no-search   # policy vs policy

--opponent: "random", "greedy", "smart" (scripted bots, agent/bots.py) or a policy checkpoint path.
Everything runs on the CPU unless --device says otherwise.
"""
import argparse
import json
import warnings
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gen1py  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent.arena import Player  # noqa: E402
from agent.obs import ACTIVE, ALIVE, HP, MAX_HP, MOVES, PRESENT, SPECIES, STATUS  # noqa: E402
from agent.search_exact import ExactSearchPlayer  # noqa: E402
from agent.selfplay import sample_team  # noqa: E402
from agent.team_tournament import showdown_team  # noqa: E402

DEX = json.load(open(ROOT / "data/gen1.json"))
SPECIES_NAME = [s["name"].title() for s in DEX["species"]]
MOVE_NAME = {m["id"]: m["name"].removesuffix("_M").replace("_", " ").title() for m in DEX["moves"] if m}


def status_name(s):
    s = int(s)
    if s & 7:
        return "SLP"
    return next((n for bit, n in ((8, "PSN"), (16, "BRN"), (32, "FRZ"), (64, "PAR")) if s & bit), "")


def mon(t):
    st = status_name(t[STATUS])
    return f"{SPECIES_NAME[int(t[SPECIES])]} {int(t[HP])}/{int(t[MAX_HP])}{' ' + st if st else ''}"


def active(team_side):
    for j in range(6):
        if team_side[j, PRESENT] and team_side[j, ACTIVE]:
            return j
    return None


def action_name(a, team_side, mask_side):
    a = int(a)
    if not mask_side.any():
        return "-"
    if a < 4:
        j = active(team_side)
        return MOVE_NAME.get(int(team_side[j, MOVES][a]), f"move {a + 1}") if j is not None else f"move {a + 1}"
    if a == 4:
        return "Struggle"
    if a < 11:
        return f"switch to {SPECIES_NAME[int(team_side[a - 5, SPECIES])]}"
    return "(no choice)"


def team_line(team_side):
    return ", ".join(SPECIES_NAME[int(t[SPECIES])] + ("" if t[ALIVE] else " (fainted)") for t in team_side if t[PRESENT])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/spec2_00200.pt"))
    ap.add_argument("--teams", default=str(ROOT / "data/search_teams_v1"))
    ap.add_argument("--opponent", default=None, help="random | greedy | smart | CHECKPOINT (default: --ckpt, no search)")
    ap.add_argument("--no-search", action="store_true", help="MetapodV0's policy alone, without the search")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", help="also write the log to this file")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    rng = np.random.default_rng(a.seed)
    ours = [showdown_team(p.read_text()) for p in sorted(Path(a.teams).glob("*_team"))]
    me = Player(a.ckpt, a.device) if a.no_search else ExactSearchPlayer(a.ckpt, a.device, tau=0.02, infer=True, k=8, seed=a.seed)
    opp = Player(a.opponent or a.ckpt, a.device)
    me_name = "MetapodV0 (policy only)" if a.no_search else "MetapodV0 (search: tau 0.02, --infer, K=8)"
    opp_name = {"random": "random bot", "greedy": "greedy bot", "smart": "smart bot"}.get(a.opponent, f"{Path(a.opponent or a.ckpt).name} (policy only)")
    score = 0.0
    for game in range(a.games):
        env = gen1py.VecEnv(1, seed=int(rng.integers(1 << 31)))
        t0_team = list(ours[rng.integers(len(ours))])
        t1_team = sample_team(rng)
        env.reset(0, t0_team, t1_team)
        if hasattr(me, "past"):
            me.past.clear(); me.pending.clear()
        say(f"=== game {game + 1}: {me_name} vs {opp_name} ===")
        team = env.team_state().astype(np.int16)
        say(f"side 0 (MetapodV0): {team_line(team[0, 0])}")
        say(f"side 1 (opponent):  {team_line(team[0, 1])}")
        idx = np.array([0])
        t_search = []
        for turn in range(1, a.max_turns + 1):
            team, battle, mask = env.team_state().astype(np.int16), env.battle_bytes(), env.legal_mask()
            if env.results()[0] >= 0:
                break
            j0, j1 = active(team[0, 0]), active(team[0, 1])
            t = time.perf_counter()
            me.last = None
            a0 = int(me.actions(team, battle, mask, 0, rng, env=env, idx=idx)[0]) if mask[0, 0].any() else 11
            t_search.append(time.perf_counter() - t)
            a1 = int(opp.actions(team, battle, mask, 1, rng, env=env, idx=idx)[0]) if mask[0, 1].any() else 11
            info = ""
            last = getattr(me, "last", None)
            if last and a0 != 11:
                mix = sorted(zip(last["mix"], last["mine"]), reverse=True)
                mix = ", ".join(f"{action_name(r, team[0, 0], mask[0, 0])} {p:.2f}" for p, r in mix if p >= 0.01)
                info = f"   [value {last['value']:+.2f}; mix {mix}]"
            say(f"turn {turn:3d}: {mon(team[0, 0, j0]) if j0 is not None else '-':28s} vs {mon(team[0, 1, j1]) if j1 is not None else '-':28s}"
                f" | ours: {action_name(a0, team[0, 0], mask[0, 0])}; theirs: {action_name(a1, team[0, 1], mask[0, 1])}{info}")
            env.step(np.array([[a0, a1]], np.int64))
            if hasattr(me, "observe"):
                me.observe(np.array([a1]), env=env, idx=idx)
        r = int(env.results()[0])
        res = {0: "MetapodV0 wins", 1: "opponent wins", 2: "draw"}.get(r, f"unfinished after {a.max_turns} turns")
        score += {0: 1.0, 2: 0.5}.get(r, 0.0)
        team = env.team_state().astype(np.int16)
        say(f"result: {res} after {turn - 1} turns; left standing: ours {int(team[0, 0, :, ALIVE].sum())}, "
            f"theirs {int(team[0, 1, :, ALIVE].sum())}; mean decision time {np.mean(t_search):.2f} s (max {np.max(t_search):.2f} s)")
        say()
    say(f"score: {score:g} / {a.games}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
