"""Fixed benchmark opponents for the 6v6 agent, working on gen1py.VecEnv state.

- random_actions: a uniformly random legal action.
- greedy_actions: the legal move with the highest expected-damage proxy (power x same-type bonus x
  type effectiveness x accuracy) against the opposing active Pokemon. It never switches unless it
  must; with no damaging move it picks a random legal move.
- smart_actions: greedy, plus the basics of RBY play. It sees both full teams (a bot may cheat).
  - Replacing a fainted Pokemon, it sends the teammate with the best matchup: its best damage
    proxy against the opposing active minus the opponent's best against it. (Switching
    voluntarily on the same rating didn't beat greedy, so it doesn't.)
  - It puts an unstatused opponent to sleep when Sleep Clause allows, else paralyses it
    (Thunder Wave doesn't touch Ground types).
  - Below half HP it heals with Recover/Softboiled, or Rest below a third.
"""
import json
from pathlib import Path

import numpy as np

from agent.obs import ACTIVE, ALIVE, HP, MAX_HP, MOVES, SLEPT_BY_FOE, SPECIES, STATUS

ROOT = Path(__file__).resolve().parent.parent
DEX = json.load(open(ROOT / "data/gen1.json"))
TYPES = np.array([s["types"] for s in DEX["species"]])
EFF = np.ones((len(DEX["types"]), len(DEX["types"])))
for a, d, m in DEX["type_chart"]:
    EFF[a, d] = m / 10
MOVE = {m["id"]: m for m in DEX["moves"] if m}


def proxy(move_id, attacker, defender):
    m = MOVE.get(int(move_id))
    if not m:
        return 0.0
    if m["effect"] == "SPECIAL_DAMAGE_EFFECT":      # Seismic Toss / Night Shade: level-based
        power = 0 if (m["type"] == 8 and 0 in TYPES[defender]) else 75
        return power * m["accuracy"] / 255
    if m["power"] <= 1:
        return 0.0
    stab = 1.5 if m["type"] in TYPES[attacker] else 1.0
    eff = EFF[m["type"], TYPES[defender][0]] * (EFF[m["type"], TYPES[defender][1]] if TYPES[defender][1] != TYPES[defender][0] else 1)
    return m["power"] * stab * eff * m["accuracy"] / 255


def random_actions(mask, side, rng):
    m = mask[:, side]
    w = m / np.maximum(m.sum(-1, keepdims=True), 1)
    return (rng.random((len(m), 1)) > np.cumsum(w, -1)).sum(-1).clip(max=11)


def greedy_actions(mask, team, side, rng):
    out = random_actions(mask, side, rng)
    n = len(mask)
    me, opp = team[:, side], team[:, 1 - side]
    a = me[np.arange(n), me[..., ACTIVE].argmax(-1)]
    o = opp[np.arange(n), opp[..., ACTIVE].argmax(-1)]
    for i in range(n):
        legal = [k for k in range(4) if mask[i, side, k]]
        if not legal:
            continue  # replacing, asleep, recharging...: keep the random legal action
        scores = [proxy(a[i, MOVES][k], a[i, SPECIES], o[i, SPECIES]) for k in legal]
        out[i] = legal[int(np.argmax(scores))] if max(scores) > 0 else legal[rng.integers(len(legal))]
    return out



def best_damage(moves, attacker, defender):
    return max(proxy(m, attacker, defender) for m in moves)


GROUND, ELECTRIC = DEX["types"].index("GROUND"), DEX["types"].index("ELECTRIC")


def status_move(move_id, attacker, opp_team, defender):
    """The move's priority for smart_actions given the opposing team: 2 sleep, 1 paralysis, 0 none."""
    m = MOVE.get(int(move_id))
    if not m or defender[STATUS] != 0:
        return 0
    if m["effect"] == "SLEEP_EFFECT":
        return 0 if (opp_team[:, SLEPT_BY_FOE] > 0).any() else 2
    if m["effect"] == "PARALYZE_EFFECT":
        return 0 if m["type"] == ELECTRIC and GROUND in TYPES[defender[SPECIES]] else 1
    return 0


def smart_actions(mask, team, side, rng):
    out = greedy_actions(mask, team, side, rng)
    me, opp = team[:, side], team[:, 1 - side]
    for i in range(len(mask)):
        o = opp[i, opp[i, :, ACTIVE].argmax()]
        a = me[i, me[i, :, ACTIVE].argmax()]
        can_switch = [j for j in range(6) if mask[i, side, 5 + j]]
        if can_switch and (not mask[i, side, :5].any() or not a[ALIVE]):    # replacing a fainted Pokemon
            rate = lambda m: best_damage(m[MOVES], m[SPECIES], o[SPECIES]) - best_damage(o[MOVES], o[SPECIES], m[SPECIES])
            out[i] = 5 + max(can_switch, key=lambda j: rate(me[i, j]))
            continue
        legal = [k for k in range(4) if mask[i, side, k]]
        if not legal:
            continue
        hp = a[HP] / max(a[MAX_HP], 1)
        heal = [k for k in legal if MOVE.get(int(a[MOVES][k]), {}).get("effect") == "HEAL_EFFECT"
                and hp < (1 / 3 if MOVE[int(a[MOVES][k])]["name"] == "REST" else 1 / 2)]
        status = max(legal, key=lambda k: status_move(a[MOVES][k], a, opp[i], o))
        if heal:
            out[i] = heal[0]
        elif status_move(a[MOVES][status], a, opp[i], o) > 0:
            out[i] = status
    return out
