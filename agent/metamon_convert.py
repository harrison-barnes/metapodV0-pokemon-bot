"""Metamon's parsed Gen 1 OU human replays -> our observation arrays, for supervised training.

Source: Hugging Face `jakegrigsby/metamon-parsed-replays` (gen1ou.tar.gz, CC BY-NC 4.0, Metamon,
UT Austin RPL: credit them, and keep this project non-commercial). Each file is one battle from one
player's side: per decision a `UniversalState` (both active Pokemon, our bench, previous moves,
hp as a fraction, one visible "effect" per active Pokemon) and the action taken
(0-3: the active Pokemon's moves in alphabetical order, 4-8: the bench in alphabetical order,
-1: never revealed).

The encoding is the Showdown adapter's own (`OurPlayer.arrays` in agent/showdown_player.py): each
state is wrapped in stand-ins for the poke-env objects that code reads, so the network sees human
battles exactly as it sees live ones. What the replay format doesn't keep is filled in as the
adapter would: slots in team / appearance order, opposing moves in the order first seen, Sleep
Clause from status changes (asleep without the side's own Rest = asleep from the foe's move).

Actions use the engine's action space (agent/obs.py): 0-3 move slot, 4 Struggle, 5-10 switch to team
slot, 11 pass. Asleep or frozen, the engine has no move choice, so a move there becomes pass (the
same rule the Showdown search uses); a recharge turn (Hyper Beam last turn, target still standing)
is pass with nothing else legal. Those turns show no choice in the replay (-1), so they are filled in
as pass; other unseen choices (mostly full paralysis and flinches) stay -1.

Output, one .npz per shard: team (N, 2, 6, 18) int16, bb (N, 102) uint8, mask (N, 12) bool,
action (N,) int8 (-1 unknown), game (N,) int32, step (N,) int16; per game: result (+1 win / -1 loss),
rating (0 = unrated), year, tour (Smogon tournament game), files.

    uv run --with lz4 --group showdown python -m agent.metamon_convert data/metamon_replays/gen1ou \
        --out data/metamon_gen1ou --workers 8
"""
import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace as NS

import lz4.frame
import numpy as np
from poke_env.battle import Effect, Status

from agent.showdown_player import MOVE_IDX, PASS, SPECIES_IDX, SPECIES_MAX_HP, OurPlayer, Tracker, species_stats

STATUS = {"par": Status.PAR, "slp": Status.SLP, "frz": Status.FRZ, "psn": Status.PSN, "tox": Status.TOX,
          "brn": Status.BRN, "fnt": Status.FNT}
EFFECT = {"confusion": Effect.CONFUSION, "substitute": Effect.SUBSTITUTE, "leechseed": Effect.LEECH_SEED,
          "reflect": Effect.REFLECT}
NAME = re.compile(r"_(Unrated|\d+)_.*_(WIN|LOSS)\.json\.lz4$")


class Move(NS):
    pass


def mon(p, hp_max, key, active):
    """A poke-env Pokemon stand-in from a UniversalPokemon dict."""
    st = STATUS.get(p["status"])
    moves = {}
    for m in sorted(p["moves"], key=lambda m: m["name"]):      # a stable slot order
        if m["name"] in MOVE_IDX and len(moves) < 4:
            moves[m["name"]] = Move(current_pp=m["current_pp"])
    hp = 0 if st == Status.FNT else round(p["hp_pct"] * hp_max)
    if st != Status.FNT:
        hp = max(hp, 1) if p["hp_pct"] > 0 else 0
    boosts = {"atk": p["atk_boost"], "def": p["def_boost"], "spe": p["spe_boost"], "spa": p["spa_boost"],
              "accuracy": p["accuracy_boost"], "evasion": p["evasion_boost"]}
    eff = {EFFECT[p["effect"]]: 1} if p["effect"] in EFFECT else {}
    return NS(species=p["name"], moves=moves, current_hp=hp, max_hp=hp_max, status=None if st == Status.FNT else st,
              fainted=st == Status.FNT or hp == 0, active=active, current_hp_fraction=hp / hp_max if hp_max else 0,
              stats=None, boosts=boosts, effects=eff, status_counter=0, _key=key)


class Adapter:
    """Just enough of OurPlayer for `arrays` / `_side` / `legal`."""
    arrays, _side = OurPlayer.arrays, OurPlayer._side

    def __init__(self):
        self.trackers = {}

    @staticmethod
    def _recharging(battle):
        return battle._recharge


def convert(path):
    """One replay file -> dict of arrays (or None if it can't be used)."""
    m = NAME.search(path.name)
    d = json.loads(lz4.frame.open(path, "rb").read())
    states, actions = d["states"], d["actions"]
    if not states or len(states) != len(actions):
        return None
    s0 = states[0]
    roster = [s0["player_active_pokemon"]["name"]] + [p["name"] for p in s0["available_switches"]]
    if len(roster) != 6 or any(n not in SPECIES_IDX for n in roster):
        return None
    own_key = {n: f"p1: {n}" for n in roster}
    tr, ad = Tracker(), Adapter()
    ad.trackers["b"] = tr
    tr.own = [own_key[n] for n in roster]
    opp_last, opp_moves = {}, {}         # opponent name -> last seen dict, revealed moves (in order seen)
    own_last = {n: None for n in roster}
    status_prev = {}
    prev_opp_active = None
    out = {"team": [], "bb": [], "mask": [], "action": [], "step": []}
    for t, (s, a) in enumerate(zip(states, actions)):
        me, op = s["player_active_pokemon"], s["opponent_active_pokemon"]
        if me["name"] not in own_key or op["name"] not in SPECIES_IDX:
            return None
        # Our team: active + bench as shown; anyone missing has fainted (keep their last moves).
        team = {}
        shown = {me["name"]: me, **{p["name"]: p for p in s["available_switches"]}}
        for n in roster:
            sp = SPECIES_IDX[n]
            if n in shown:
                own_last[n] = shown[n]
                team[own_key[n]] = mon(shown[n], SPECIES_MAX_HP[sp], own_key[n], n == me["name"])
            else:
                p = dict(own_last[n] or me, name=n, status="fnt", hp_pct=0.0, effect="noeffect")
                team[own_key[n]] = mon(p, SPECIES_MAX_HP[sp], own_key[n], False)
        for k, v in team.items():
            if v.active:
                v.stats = dict(zip(("atk", "def", "spe", "spa"), species_stats(SPECIES_IDX[v.species])))
        tr.seen_own.add(own_key[me["name"]])
        # Opponent: everyone seen so far, last known state; revealed moves accumulate.
        okey = f"p2: {op['name']}"
        if okey not in tr.opp:
            tr.opp.append(okey)
        opp_last[op["name"]] = op
        seen = opp_moves.setdefault(okey, [])
        for mv in op["moves"]:
            i = MOVE_IDX.get(mv["name"])
            if i and i not in seen and len(seen) < 4:
                seen.append(i)
        tr.used[okey] = list(seen)
        opp_team = {}
        for k in tr.opp:
            n = k[4:]
            p = opp_last[n]
            if n != op["name"] and p["effect"] != "noeffect":
                p = dict(p, effect="noeffect", atk_boost=0, def_boost=0, spe_boost=0, spa_boost=0,
                         accuracy_boost=0, evasion_boost=0)
            opp_team[k] = mon(p, SPECIES_MAX_HP[SPECIES_IDX[n]], k, n == op["name"])
        # Moves we've used, last moves, volatiles the adapter tracks itself.
        own_active = own_key[me["name"]]
        pm, om = s["player_prev_move"]["name"], s["opponent_prev_move"]["name"]
        if MOVE_IDX.get(pm):
            u = tr.used.setdefault(own_active, [])
            if MOVE_IDX[pm] not in u:
                u.append(MOVE_IDX[pm])
            tr.last_move["p1"] = MOVE_IDX[pm]
        if MOVE_IDX.get(om):
            tr.last_move["p2"] = MOVE_IDX[om]
        tr.volatile = {own_active: {"lightscreen"}} if me["effect"] == "lightscreen" else {}
        if op["effect"] == "lightscreen":
            tr.volatile[okey] = {"lightscreen"}
        # Sleep Clause bookkeeping: newly asleep without the side's own Rest -> asleep from a foe's move.
        for key, p, own_move in ((own_active, me, pm), (okey, op, om)):
            was = status_prev.get(key)
            if p["status"] == "slp" and was != "slp" and own_move != "rest":
                tr.slept_by_foe.add(key)
            if p["status"] != "slp":
                tr.slept_by_foe.discard(key)
            status_prev[key] = p["status"]
        # Recharging: Hyper Beam used last turn (newly shown as the previous move, or last action), the target still
        # standing, and the turn's choice unseen or recorded as Metamon's recharge index 0.
        used_now = t > 0 and (states[t - 1]["player_prev_move"]["name"] != "hyperbeam"
                              or _was_move(states[t - 1], actions[t - 1], "hyperbeam"))
        recharge = (pm == "hyperbeam" and used_now and prev_opp_active == op["name"] and op["status"] != "fnt"
                    and not s["forced_switch"] and a in (-1, 0))
        prev_opp_active = op["name"]
        battle = NS(battle_tag="b", team=team, opponent_team=opp_team, active_pokemon=team[own_active],
                    opponent_active_pokemon=opp_team[okey], player_role="p1", opponent_role="p2",
                    side_conditions={}, opponent_side_conditions={}, _recharge=recharge)
        tarr, bb = ad.arrays(battle)
        mask, act = legal_and_action(s, a, tr, team, own_active, recharge)
        out["team"].append(tarr[0]); out["bb"].append(bb[0]); out["mask"].append(mask); out["action"].append(act)
        out["step"].append(t)
    if not out["team"]:
        return None
    arr = {k: np.asarray(v) for k, v in out.items()}
    rating = m.group(1) if m else "Unrated"
    arr["meta"] = dict(result=1 if (m and m.group(2) == "WIN") else -1, rating=0 if rating == "Unrated" else int(rating),
                       year=int(path.parent.parent.name), tour=path.name.startswith("smogtours"), file=path.name)
    return arr


def _was_move(s, a, name):
    if not 0 <= a <= 3:
        return False
    ms = sorted(m["name"] for m in s["player_active_pokemon"]["moves"])
    return a < len(ms) and ms[a] == name


def legal_and_action(s, a, tr, team, own_active, recharge):
    """Engine-space legal mask and action for our side."""
    mask = np.zeros(12, bool)
    me = s["player_active_pokemon"]
    active = team[own_active]
    slots = [k for k in active.moves]                       # our move slots, in team-array order
    bench = sorted(p["name"] for p in s["available_switches"])
    for n in bench:
        mask[5 + tr.own.index(f"p1: {n}")] = True
    act = -1
    if s["forced_switch"] or me["status"] == "fnt":
        if 4 <= a <= 8 and a - 4 < len(bench):
            act = 5 + tr.own.index(f"p1: {bench[a - 4]}")
        return mask, act
    if recharge:
        mask[:] = False
        mask[PASS] = True
        return mask, PASS
    if me["status"] in ("slp", "frz"):
        mask[PASS] = True
    else:
        usable = [i for i, k in enumerate(slots) if active.moves[k].current_pp > 0]
        for i in usable or []:
            mask[i] = True
        if not usable:
            mask[4] = True
    if a == -1 and me["status"] in ("slp", "frz"):
        act = PASS                                          # no move to show: the engine's pass
    if 0 <= a <= 3:
        if me["status"] in ("slp", "frz"):
            act = PASS
        elif not mask[:4].any():
            act = 4
        else:
            names = sorted(m["name"] for m in me["moves"] if m["name"] in MOVE_IDX)
            if a < len(names) and names[a] in slots:
                act = slots.index(names[a])
    elif 4 <= a <= 8 and a - 4 < len(bench):
        act = 5 + tr.own.index(f"p1: {bench[a - 4]}")
    if act >= 0 and not mask[act]:
        act = -1
    return mask, act


def _safe(path):
    try:
        return convert(path)
    except Exception as e:                     # count and skip malformed files
        return repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=10000, help="games per shard")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    files = sorted(Path(args.src).rglob("*.json.lz4"))[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors, skipped, decisions = {}, 0, 0
    with Pool(args.workers) as pool:
        for k in range(0, len(files), args.shard):
            games = []
            for r in pool.imap(_safe, files[k:k + args.shard], chunksize=64):
                if isinstance(r, str):
                    errors[r[:80]] = errors.get(r[:80], 0) + 1
                elif r is None:
                    skipped += 1
                else:
                    games.append(r)
            g = np.concatenate([np.full(len(x["action"]), i, np.int32) for i, x in enumerate(games)])
            meta = [x.pop("meta") for x in games]
            np.savez_compressed(out / f"shard{k // args.shard:03d}.npz",
                     **{key: np.concatenate([x[key] for x in games]) for key in ("team", "bb", "mask", "action", "step")},
                     game=g, result=np.array([m["result"] for m in meta], np.int8),
                     rating=np.array([m["rating"] for m in meta], np.int16), year=np.array([m["year"] for m in meta], np.int16),
                     tour=np.array([m["tour"] for m in meta]), files=np.array([m["file"] for m in meta]))
            decisions += len(g)
            print(f"shard {k // args.shard}: {len(games)} games, {len(g)} decisions; skipped {skipped}, errors {sum(errors.values())}",
                  flush=True)
    print("errors:", json.dumps(errors, indent=1))


if __name__ == "__main__":
    main()
