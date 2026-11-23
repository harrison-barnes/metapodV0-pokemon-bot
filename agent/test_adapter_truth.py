"""Ground-truth test of the whole Showdown-to-engine state rebuild (agent/showdown_player.py).

tools/showdown/gen1_truth.js plays Gen 1 OU battles in the Showdown simulator and writes, per decision point, the
messages each player would have received and Showdown's internal state. Here those messages are fed, offline, to the
real adapter (an OurPlayer that never connects: its poke-env Battle objects are built by the same message handlers as on
a server), once from each player's seat, and everything the adapter rebuilds at each decision is compared with the
truth, from the point of view of a player who sees only their own side's private information:

- "knowable" fields (what the protocol shows, or what can be counted from it) must match: a difference is a mismatch;
- hidden quantities (sleep, confusion, Disable, trapping, Thrash and Bide turns left, Substitute HP, the opponent's
  exact HP and unrevealed moves) are reported separately: for those the test checks that the truth lies inside what the
  search would draw for them (agent/search_exact.py _draw_hidden), given the adapter's counts;
- the legal action set of the engine, loaded with the rebuilt state, against Showdown's request (both sides).

    (cd tools/showdown && npm install && node gen1_truth.js --games 300 | gzip > ../../truth_exotic.jsonl.gz)
    uv run --group showdown python -m agent.test_adapter_truth truth_exotic.jsonl.gz
    ... --write-teams teams.jsonl --games 300     # realistic teams (ours vs human teams) for gen1_truth.js --teams
    ... --out table.json --bad bad.jsonl          # the table as JSON; every mismatch with its context
    ... --max-rate 0.001                          # exit 1 if any knowable field differs more often than this
"""
import argparse
import asyncio
import collections
import gzip
import json
import logging
import sys

import gen1py  # noqa: F401
import numpy as np
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.concurrency import POKE_LOOP

from agent.obs import ACTIVE, ALIVE, HP, MAX_HP, MOVES, MOVES_REVEALED, PP, REVEALED, SLEPT_BY_FOE, SPECIES, STATUS
from agent import showdown_player as ADAPTER
from agent.showdown_player import DEX, MOVE_DATA, MOVE_IDX, SPECIES_IDX, showdown_team, to_id

TYPE_IDX = {to_id(t.replace("_TYPE", "")): i for i, t in enumerate(DEX["types"])}
STATUS_BITS = {"": 0, "psn": 8, "tox": 8, "brn": 16, "frz": 32, "par": 64}
SIDE = 50
BS1 = {"bide": 1, "thrash": 2, "charging": 16, "trapping": 32, "invulnerable": 64, "confused": 128}
BS2 = {"mist": 2, "focus": 4, "sub": 16, "recharge": 32, "rage": 64, "seeded": 128}
BS3 = {"toxic": 1, "lightscreen": 2, "reflect": 4, "transformed": 8}
VOL = {"bide": "bide", "thrash": "lockedmove", "charging": "twoturnmove", "trapping": "partialtrappinglock",
       "invulnerable": "invulnerability", "confused": "confusion", "mist": "mist", "focus": "focusenergy", "sub": "substitute",
       "recharge": "mustrecharge", "rage": "rage", "seeded": "leechseed", "toxic": "residualdmg", "lightscreen": "lightscreen",
       "reflect": "reflect"}
PASS = 11


logging.getLogger("poke-env").setLevel(logging.ERROR)


class _Stub:
    """Stands in for the search player where the adapter hands it state (trap hits, turns asleep / confused)."""
    infer = False
    trap_uncertain = True        # so the adapter reports a trap that may go on, instead of drawing whether it does

    def __init__(self):
        self.trap, self.seen, self.past = {}, {}, {}


def offline_player(name, adapter=None, play=None):
    """The adapter (agent.showdown_player, or the module file `adapter`: an older copy, for before / after tables) fed
    from recorded messages. Every decision's rebuilt state is kept in .captured. play: OurPlayer arguments (ckpt,
    search_tau, ...): then every decision is also really made (policy, search, trace), and thrown away."""
    mod = ADAPTER
    if adapter:
        import importlib.util
        spec = importlib.util.spec_from_file_location("adapter_under_test", adapter)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return type("OfflinePlayer", (_Offline, mod.OurPlayer), {"Tracker": mod.Tracker, "Real": mod.OurPlayer})(name, play)


class _Offline:
    def __init__(self, name, play=None):
        self.play = play
        if play:
            self._init(name, **play)
            return
        try:
            self._init(name, None)
        except Exception:               # an adapter from before ckpt=None was allowed: any small checkpoint will do
            import glob
            self._init(name, "checkpoints/spec2_00200.pt")
        self.searcher = _Stub()

    def _init(self, name, ckpt, **kw):
        super().__init__(ckpt, "cpu", account_configuration=AccountConfiguration(name, None),
                         server_configuration=LocalhostServerConfiguration, battle_format="gen1ou", start_listening=False,
                         max_concurrent_battles=0, log_level=logging.ERROR, **kw)
        self.captured, self.sent, self.rebuild_errors, self.orders = None, [], [], []

        async def send(message, room=None, message_2=None):
            self.sent.append((room, message))
        self.ps_client.send_message = send

    def choose_move(self, battle):
        if self.play:               # the real decision, start to finish
            order = self.Real.choose_move(self, battle)
            self.orders.append(order.message)
            return order
        try:
            self.captured = self.capture(battle)
        except Exception as e:      # a rebuild that raises would cost the live bot this decision
            self.captured = None
            self.rebuild_errors.append((battle.battle_tag, battle.turn, repr(e)))
        return self.choose_default_move()

    def capture(self, battle):
        tr = self.trackers.setdefault(battle.battle_tag, self.Tracker())
        st = self.rebuild(battle) if hasattr(self, "rebuild") else None
        if st is None:              # the adapter before rebuild() existed
            team, bb = self.arrays(battle)
            mask, _ = self.legal(battle, tr)
            st = {"team": team[0], "bb": bb[0], "mask": mask, "trap": dict(self.searcher.trap.get(0, {}))}
        cap = dict(st, turn=battle.turn, force=bool(battle.force_switch), own=list(tr.own), opp=list(tr.opp))
        if "seen" not in cap:       # the search's own count (ExactSearchPlayer._count_seen), made only when it searches
            from agent.search_exact import ExactSearchPlayer
            seen = tr.__dict__.setdefault("seen", {})
            cap["seen"] = {k: list(v) for k, v in seen.items()}
            if cap["mask"].sum() > 1:
                self.searcher.seen = {0: seen}
                ExactSearchPlayer._count_seen(self.searcher, cap["team"][None], [cap["bb"]], [0])
        try:
            env = gen1py.VecEnv(1, seed=1)
            env.load(0, cap["team"].astype(np.int64), bytes(cap["bb"]), int(battle.turn))
            # as the search does before it reads the opponent's options: unseen opponents become guesses (full HP)
            env.set_members([(0, 1, j, "MAGIKARP:SPLASH") for j in range(6) if cap["team"][1, j, 17] and not cap["team"][1, j, REVEALED]])
            cap["emask"] = env.legal_mask()[0].copy()
        except Exception as e:
            cap["emask"], cap["load_error"] = None, repr(e)
        return cap

    def feed(self, tag, text):
        chunk = f">{tag}\n{text}"
        fut = asyncio.run_coroutine_threadsafe(self._handle_battle_message([m.split("|") for m in chunk.split("\n")]), POKE_LOOP)
        fut.result(timeout=30)


# -- the truth, in the engine's terms ------------------------------------------------------------------------------
def move_id(sd_id):
    return MOVE_IDX.get(sd_id, 0)


def side_truth(T, s):
    """Showdown's state of side s's active Pokemon as the engine's battle fields."""
    S = T["sides"][s]
    m = next((p for p in S["pokemon"] if p["active"]), None)
    if m is None:
        return None
    vol, mod, sto, b = m["volatiles"], m["modified"], m["stored"], m["boosts"]
    out = {
        "hp": m["hp"], "stats": [mod[k] for k in ("atk", "def", "spe", "spa")], "unmod": [sto[k] for k in ("atk", "def", "spe", "spa")],
        "status": STATUS_BITS.get(m["status"], 0), "asleep": m["status"] == "slp",
        "sleep_left": m["status_state"].get("time", 0) if m["status"] == "slp" else 0,
        "sleep_start": m["status_state"].get("startTime", 0) if m["status"] == "slp" else 0,
        "mods": [b[k] + 7 for k in ("atk", "def", "spe", "spa", "accuracy", "evasion")],
        "conf_left": vol.get("confusion", {}).get("time", 0),
        "toxic_n": vol.get("residualdmg", {}).get("counter", 0),
        "disable_slot": vol["disable"]["slotIndex"] + 1 if "disable" in vol else 0,
        "disable_left": vol.get("disable", {}).get("time", 0),
        "disable_move": move_id(vol["disable"]["move"]) if "disable" in vol else 0,
        "attacks_left": (vol["bide"]["time"] if "bide" in vol else vol["lockedmove"]["time"] if "lockedmove" in vol
                         else vol["partialtrappinglock"]["duration"] if "partialtrappinglock" in vol else 0),
        "trap_total": vol.get("partialtrappinglock", {}).get("totalDuration", 0),
        "sub_hp": vol["substitute"]["hp"] - 1 if "substitute" in vol else 0,
        "bide_dmg": vol.get("bide", {}).get("damage", 0),
        # what Counter reads (Showdown resolves it from the acting player's view): the side's last selected and last
        # announced move as the other player saw them
        "selected": {"nomove": 0, "cannotmove": 0xFF, None: 0}.get(S["last_enemy_selected_move"], move_id(S["last_enemy_selected_move"])),
        "last_move": move_id(S["last_enemy_move"]) if S["last_enemy_move"] else 0,
        "types": sorted(TYPE_IDX.get(to_id(t), 255) for t in (m["types"] * 2)[:2]),
        "moves": [move_id(x["id"]) for x in m["moves"]], "species": SPECIES_IDX[m["set_species"]],
        "transformed": m["transformed"], "mon": m,
    }
    for name, v in VOL.items():
        out[name] = v in vol
    out["transformed"] = m["transformed"]
    return out


def side_seen(bb, s):
    """The adapter's battle bytes of side s (0 own, 1 opponent) as the same fields."""
    o = bb[s * SIDE:(s + 1) * SIDE]
    u16 = lambda a, n=1: [int(x) for x in o[a:a + 2 * n].view("<u2")]
    out = {
        "hp": u16(0)[0], "stats": u16(2, 4), "unmod": u16(10, 4), "bide_dmg": u16(18)[0], "status": int(o[20]) & 0xF8, "asleep": bool(o[20] & 7),
        "sleep_left": int(o[20] & 7), "mods": [int(x) for x in o[21:27]], "conf_left": int(o[30]), "toxic_n": int(o[31]),
        "disable_slot": int(o[32]) >> 4, "disable_left": int(o[32]) & 15, "disable_move": int(o[33]), "attacks_left": int(o[34]),
        "sub_hp": int(o[35]), "selected": int(o[36]), "last_move": int(o[37]), "ram": [int(x) for x in o[37:41]],
        "types": sorted(int(x) for x in o[41:43]), "moves": [int(x) for x in o[43:47]], "species": int(o[47]), "clauses": int(o[49]),
    }
    for bits, byte in ((BS1, 27), (BS2, 28), (BS3, 29)):
        for name, bit in bits.items():
            out[name] = bool(o[byte] & bit)
    return out


FLAGS = list(BS1) + list(BS2) + list(BS3)
# Hidden from the player, for both sides: (field, flag that says it applies)
HIDDEN = {"sleep_left": "asleep", "conf_left": "confused", "disable_left": "disable_slot", "sub_hp": "sub"}


class Table:
    def __init__(self):
        self.n, self.bad, self.ex = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        self.all_bad = []

    def add(self, field, ok, ctx=None, got=None, want=None, kind="knowable"):
        key = (kind, field)
        self.n[key] += 1
        if not ok:
            self.bad[key] += 1
            row = dict(ctx or {}, field=field, kind=kind, got=got, want=want)
            self.all_bad.append(row)
            if len(self.ex[key]) < 3:
                self.ex[key].append(row)


def draw_support(field, seen_side, cap, who, truth_side, mon_key):
    """What agent/search_exact.py _draw_hidden could put in a hidden field, given the adapter's state."""
    slot = (who, cap["own" if who == 0 else "opp"].index(mon_key)) if mon_key in cap["own" if who == 0 else "opp"] else None
    e = cap["seen"].get(slot, [0, 0])
    if field == "sleep_left":
        j = slot[1] if slot else 0
        by_foe = bool(cap["team"][who, j, SLEPT_BY_FOE])
        return set(range(1, max(1, 7 - e[0]) + 1)) if by_foe else {max(1, 2 - e[0])}
    if field == "conf_left":
        return set(range(max(1, 2 - e[1]), max(1, 5 - e[1]) + 1))
    if field == "sub_hp":
        j = slot[1] if slot else 0
        if who == 0 and seen_side["sub_hp"]:
            return {seen_side["sub_hp"]}
        return {max(1, int(cap["team"][who, j, MAX_HP]) // 4)}
    if field == "attacks_left":       # trapping: 1-4 continuation hits after the first
        h = cap["trap"].get(who)
        if h is None:
            return {seen_side["attacks_left"]}
        return {n - h for n in range(1, 5) if n > h} or {1}
    if field == "disable_left":
        return set(range(1, 9)) if seen_side["disable_left"] else {0}
    return set()


def request_choices(req):
    """(free move ids or None when the move is forced / there is none to choose, forced?, may switch?)"""
    if req is None or req.get("wait"):
        return None
    if req.get("forceSwitch"):
        return {"force": True}
    act = req["active"][0]
    mv = act["moves"]
    forced = len(mv) == 1 and "pp" not in mv[0] and mv[0]["id"] != "struggle"
    return {"force": False, "forced": forced, "moves": [m["id"] for m in mv if not m.get("disabled")] if not forced else [mv[0]["id"]],
            "can_switch": not act.get("trapped")}


def compare(tab, row, cap, viewer, reqs):
    T = row["truth"]
    ctx0 = {"game": row["game"], "tag": row["tag"], "step": row["step"], "turn": T["turn"], "viewer": viewer}
    team, bb = cap["team"], cap["bb"]
    truth_sides = [side_truth(T, viewer), side_truth(T, 1 - viewer)]
    for who, name in ((0, "own"), (1, "opp")):
        S = T["sides"][viewer if who == 0 else 1 - viewer]
        keys = cap["own"] if who == 0 else cap["opp"]
        by_ident = {p["ident"]: p for p in S["pokemon"]}
        # ---- team rows
        if who == 1:
            shown = {p["ident"] for p in S["pokemon"] if p["ident"] in keys}
            tab.add("opp.team.count", len(keys) == len(shown), ctx0, len(keys), len(shown))
        for j, k in enumerate(keys[:6]):
            p, t = by_ident.get(k), team[who, j]
            ctx = dict(ctx0, mon=k)
            if p is None:
                tab.add(f"{name}.team.ident", False, ctx, k, sorted(by_ident))
                continue
            tab.add(f"{name}.team.species", int(t[SPECIES]) == SPECIES_IDX[p["base_species"]], ctx, int(t[SPECIES]), p["base_species"])
            if who == 0:
                tab.add("own.team.hp", int(t[HP]) == p["hp"], ctx, int(t[HP]), p["hp"])
                tab.add("own.team.max_hp", int(t[MAX_HP]) == p["maxhp"], ctx, int(t[MAX_HP]), p["maxhp"])
            else:
                tab.add("opp.team.max_hp", int(t[MAX_HP]) == p["maxhp"], ctx, int(t[MAX_HP]), p["maxhp"])
                lo, hi = (pct_bounds(p["hp"], p["maxhp"]))
                tab.add("opp.team.hp_in_shown_percent", lo <= int(t[HP]) <= hi, ctx, int(t[HP]), [lo, hi, p["hp"]])
                tab.add("opp.team.hp_exact", int(t[HP]) == p["hp"], ctx, int(t[HP]) - p["hp"], 0, kind="hidden")
            st = "" if p["fainted"] else p["status"]
            tab.add(f"{name}.team.status", (int(t[STATUS]) & 0xF8) == STATUS_BITS.get(st, 0) and bool(t[STATUS] & 7) == (st == "slp"), ctx,
                    int(t[STATUS]), st)
            tab.add(f"{name}.team.alive", bool(t[ALIVE]) == (not p["fainted"]), ctx, int(t[ALIVE]), not p["fainted"])
            if not p["fainted"]:        # a fainted active Pokemon stays "active" for the engine until it is replaced
                tab.add(f"{name}.team.active", bool(t[ACTIVE]) == p["active"], ctx, int(t[ACTIVE]), p["active"])
            tab.add(f"{name}.team.slept_by_foe", bool(t[SLEPT_BY_FOE]) == p["slept_by_foe"], ctx, int(t[SLEPT_BY_FOE]), p["slept_by_foe"])
            base = {move_id(x["id"]): x for x in p["base_moves"]}
            mv = [int(x) for x in t[MOVES]]
            pp = [int(x) for x in t[PP]]
            if who == 0:
                want = [move_id(x["id"]) for x in p["base_moves"]] + [0] * (4 - len(p["base_moves"]))
                tab.add("own.team.moves", mv == want, ctx, mv, want)
                # (while transformed the row carries the copied moves' PP: the engine reads PP by slot)
                src = p["moves"] if p["transformed"] else p["base_moves"]
                wpp = [x["pp"] for x in src] + [0] * (4 - len(src))
                tab.add("own.team.pp", mv != want or pp == wpp, ctx, pp, wpp)
            else:
                known = [x for x in mv if x]
                tab.add("opp.team.moves_are_its_own", all(x in base for x in known), ctx, known, sorted(base))
                used = sorted(move_id(x["id"]) for x in p["base_moves"] if x["pp"] < x["maxpp"])
                tab.add("opp.team.moves_all_used_shown", set(used) <= set(known), ctx, sorted(known), used)
                for x, q in zip(mv, pp):
                    if x and x in base:
                        tab.add("opp.team.pp_of_shown_moves", q == base[x]["pp"], dict(ctx, move=x), q, base[x]["pp"])
        # ---- the active Pokemon
        ts, ss = truth_sides[who], side_seen(bb, who)
        if ts is None or ts["mon"]["fainted"]:
            continue
        key = ts["mon"]["ident"]
        vols = sorted(ts["mon"]["volatiles"])
        ctx = dict(ctx0, mon=key, vol=vols, status=ts["mon"]["status"], transformed=ts["transformed"])
        if key not in keys or keys.index(key) > 5 or not team[who, keys.index(key), ACTIVE]:
            tab.add(f"{name}.active_is_right", False, ctx, None, key)
            continue
        tab.add(f"{name}.active_is_right", True)
        if who == 0:
            tab.add("own.hp", ss["hp"] == ts["hp"], ctx, ss["hp"], ts["hp"])
        for f in ("stats", "unmod", "mods", "status", "asleep", "types", "species", "toxic_n"):
            tab.add(f"{name}.{f}", ss[f] == ts[f], ctx, ss[f], ts[f])
        for f in FLAGS:
            if f == "trapping" and ss[f] and not ts[f] and cap.get("trap_uncertain"):
                # after the first continuation turn Showdown no longer says whether a trap goes on: the adapter keeps
                # the flag with the hits so far and the search draws whether it holds
                tab.add(f"{name}.trap_that_may_go_on_is_over", False, ctx, cap["trap"].get(who), ts["mon"]["volatiles"].get("fakepartiallytrapped"), kind="hidden")
                continue
            if f == "trapping" and ss[f] and ts[f] and cap.get("trap_uncertain"):
                tab.add(f"{name}.trap_that_may_go_on_is_over", True, kind="hidden")
            tab.add(f"{name}.flag.{f}", ss[f] == ts[f], ctx, ss[f], ts[f])
        if who == 0:
            tab.add("own.moves", ss["moves"] == ts["moves"] + [0] * (4 - len(ts["moves"])), ctx, ss["moves"], ts["moves"])
        else:
            known = [x for x in ss["moves"] if x]
            tab.add("opp.moves_are_its_own", all(x in ts["moves"] for x in known), ctx, known, ts["moves"])
        # Disable: which move is public (the -start line names it); the turns left are hidden
        if who == 0:
            tab.add("own.disable_slot", ss["disable_slot"] == ts["disable_slot"], ctx, ss["disable_slot"], ts["disable_slot"])
        tab.add(f"{name}.disable_move", ss["disable_move"] == ts["disable_move"], ctx, ss["disable_move"], ts["disable_move"])
        # last moves, for Counter and Mirror Move
        tab.add(f"{name}.last_selected_move", ss["selected"] == ts["selected"], ctx, ss["selected"], ts["selected"], kind="info")
        tab.add(f"{name}.last_announced_move", ss["last_move"] == ts["last_move"], ctx, ss["last_move"], ts["last_move"], kind="info")
        # Would a Counter aimed at this side succeed, and for how much? (its last move Normal / Fighting with power, not
        # Counter itself, and damage dealt last: the three things Counter reads)
        want = MOVE_DATA.get(ts["last_move"], {"power": 0, "type": 0})
        counterable = lambda power, typ, sel: bool(power) and typ in (0, 1) and sel != MOVE_IDX["counter"]
        dmg = int(bb[100:102].view("<u2")[0])
        got_c = 2 * dmg if counterable(ss["ram"][1], ss["ram"][2], ss["selected"]) else 0
        want_c = 2 * T["last_damage"] if counterable(want["power"], want["type"], ts["selected"]) else 0
        shown = cap.get("damage_shown", True)   # damage into a Substitute doesn't show
        tab.add(f"{name}.counter_against_it.succeeds", bool(got_c) == bool(want_c), dict(ctx, ram=ss["ram"], last=ts["last_move"]), got_c, want_c)
        if got_c and want_c and not shown:
            tab.add(f"{name}.counter_against_it.damage_after_substitute_hit", abs(got_c - want_c) <= max(8, want_c // 10), ctx, got_c, want_c, kind="hidden")
        elif got_c and want_c:
            # within 10%: damage we dealt shows only as a percentage of the opponent's HP
            tab.add(f"{name}.counter_against_it.damage", abs(got_c - want_c) <= max(8, want_c // 10),
                    dict(ctx, ram=ss["ram"], last=ts["last_move"]), got_c, want_c)
            tab.add(f"{name}.counter_against_it.damage_exact", got_c == want_c, ctx, got_c, want_c, kind="info")
        # hidden counters: is the truth inside what the search would draw?
        for f, flag in HIDDEN.items():
            if ts[flag] and ss[flag]:
                sup = draw_support(f, ss, cap, who, ts, key)
                tab.add(f"{name}.{f}", ts[f] in sup, dict(ctx, start=ts.get("sleep_start")), sorted(sup), ts[f], kind="hidden")
        if ts["trapping"] and ss["trapping"]:
            sup = draw_support("attacks_left", ss, cap, who, ts, key)
            tab.add(f"{name}.trap_turns_left", ts["attacks_left"] in sup, dict(ctx, total=ts["trap_total"]), sorted(sup), ts["attacks_left"], kind="hidden")
        if ts["thrash"] or ts["bide"]:
            tab.add(f"{name}.{'thrash' if ts['thrash'] else 'bide'}_turns_left", ss["attacks_left"] == ts["attacks_left"], ctx,
                    ss["attacks_left"], ts["attacks_left"], kind="hidden")
        if ts["bide"]:
            tab.add(f"{name}.bide_damage", ss["bide_dmg"] == ts["bide_dmg"], ctx, ss["bide_dmg"], ts["bide_dmg"], kind="hidden" if who else "info")
        # clause flags the engine reads for this side's sleep moves and freezes
        foes = T["sides"][1 - viewer if who == 0 else viewer]["pokemon"]
        want = (1 if any(p["hp"] and p["slept_by_foe"] for p in foes) else 0) | (2 if any(p["status"] == "frz" for p in foes) else 0) \
            | (4 if ts["mon"]["slept_by_foe"] else 0)
        tab.add(f"{name}.clauses", ss["clauses"] == want, ctx, ss["clauses"], want)
    dmg = int(bb[100:102].view("<u2")[0])
    tab.add("last_damage", dmg == T["last_damage"], dict(ctx0, vols=[sorted(t["mon"]["volatiles"]) if t else None for t in truth_sides]), dmg, T["last_damage"], kind="info")
    # ---- legal actions: Showdown's request against our mask, and against the engine loaded with the rebuilt state
    mine = request_choices(reqs[viewer])
    theirs = request_choices(reqs[1 - viewer])
    em = cap.get("emask")
    ctx = dict(ctx0, req=mine, vols=[sorted(t["mon"]["volatiles"]) if t else None for t in truth_sides],
               status=[t["mon"]["status"] if t else None for t in truth_sides])
    tab.add("engine.loads", em is not None, dict(ctx, err=cap.get("load_error")))
    if em is None or mine is None:
        return
    # A trap that may or may not go on: Showdown offers every move (used only if the trap has ended) where the engine,
    # in the world where it holds, offers none. Not comparable; the search plays both worlds.
    maybe_trap = bool(cap.get("trap_uncertain")) or any(side_seen(bb, w)["trapping"] for w in (0, 1)) and bool(
        (reqs[viewer].get("active") or [{}])[0].get("maybeLocked"))
    own_keys = cap["own"]
    S = T["sides"][viewer]
    alive_bench = sorted(5 + own_keys.index(p["ident"]) for p in S["pokemon"] if not p["active"] and not p["fainted"] and p["ident"] in own_keys)
    e_sw = sorted(int(a) for a in np.flatnonzero(em[0]) if 5 <= a <= 10)
    e_mv = sorted(int(a) for a in np.flatnonzero(em[0]) if a < 5 or a == PASS)
    a_sw = sorted(int(a) for a in np.flatnonzero(cap["mask"]) if 5 <= a <= 10)
    if mine["force"]:
        tab.add("legal.own.forced_switch", e_sw == alive_bench and not [a for a in e_mv if a != PASS] or not alive_bench, ctx, [e_mv, e_sw], alive_bench)
        tab.add("legal.own.adapter_mask", a_sw == alive_bench, ctx, a_sw, alive_bench)
    else:
        want_sw = alive_bench if mine["can_switch"] else []
        tab.add("legal.own.switches", e_sw == want_sw, ctx, e_sw, want_sw)
        tab.add("legal.own.adapter_switches", a_sw == want_sw, ctx, a_sw, want_sw)
        own_moves = [x["id"] for x in next(p for p in S["pokemon"] if p["active"])["moves"]]
        if mine["forced"]:
            want_mv = [PASS]
        elif mine["moves"] == ["struggle"]:
            want_mv = [4]
        else:
            want_mv = [i for i, x in enumerate(reqs[viewer]["active"][0]["moves"]) if not x.get("disabled")]
        if maybe_trap:
            tab.add("legal.own.moves_with_uncertain_trap", e_mv == want_mv, ctx, e_mv, want_mv, kind="info")
        else:
            tab.add("legal.own.moves", e_mv == want_mv, ctx, e_mv, want_mv)
    if theirs is None:
        return
    ctx = dict(ctx, req=theirs)
    o_sw = [int(a) for a in np.flatnonzero(em[1]) if 5 <= a <= 10]
    o_mv = [int(a) for a in np.flatnonzero(em[1]) if a < 5 or a == PASS]
    if theirs["force"]:
        tab.add("legal.opp.forced_switch", not [a for a in o_mv if a != PASS], ctx, [o_mv, o_sw], "switch")
    else:
        tab.add("legal.opp.may_switch", bool(o_sw) == theirs["can_switch"] or not any(
            not p["active"] and not p["fainted"] for p in T["sides"][1 - viewer]["pokemon"]), ctx, o_sw, theirs["can_switch"])
        if maybe_trap:
            tab.add("legal.opp.move_is_forced_with_uncertain_trap", (o_mv == [PASS]) == theirs["forced"], ctx, o_mv, theirs, kind="info")
        else:
            tab.add("legal.opp.move_is_forced", (o_mv == [PASS]) == theirs["forced"], ctx, o_mv, theirs)
            if theirs["moves"] == ["struggle"]:
                tab.add("legal.opp.struggle", o_mv == [4], ctx, o_mv, theirs, kind="info")   # needs all four moves seen at 0 PP


def pct_bounds(hp, maxhp):
    """The exact HP values that show as the same percentage as hp (Showdown's HP Percentage Mod)."""
    if hp == 0:
        return 0, 0
    shown = lambda h: 99 if (p := -(-h * 100 // maxhp)) == 100 and h < maxhp else p
    pct = shown(hp)
    hs = [h for h in range(1, maxhp + 1) if shown(h) == pct]
    return hs[0], hs[-1]


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        for line in f:
            yield json.loads(line)


def run(paths, viewers=(0, 1), limit=None, race_start=False, adapter=None):
    tab = Table()
    players = [offline_player("Truth One", adapter), offline_player("Truth Two", adapter)]
    decisions = games = 0
    for path in paths:
        game, first = None, True
        for row in read(path):
            if row["game"] != game:
                game, first = row["game"], True
                games += 1
                if limit and games > limit:
                    break
                tag = f"battle-gen1ou-{games}"
            reqs = [None, None]
            for v in viewers:
                players[v].captured = None
            for who, kind, text in row["msgs"]:
                v = int(who[1]) - 1
                if kind == "request" and "|request|" in text:     # (other private lines, e.g. |-activate|...|move: Struggle)
                    body = text.split("|request|", 1)[1]
                    reqs[v] = json.loads(body) if body else None
                if v in viewers:
                    if first and kind == "update":
                        # As a server delivers it: the room's opening lines (with both |player| lines), then the start.
                        # race_start: everything in one message, which happens now and then (1 of 160 local battles).
                        head, sep, rest = text.partition("\n|\n")
                        head = "|init|battle\n|title|Truth One vs. Truth Two\n" + head
                        if race_start or not sep:
                            text = head + sep + rest
                        else:
                            players[v].feed(tag, head)
                            text = "|" + "\n" + rest
                    players[v].feed(tag, text)
            first = False
            for v in viewers:
                cap = players[v].captured
                if cap is not None and not row["truth"]["ended"]:
                    decisions += 1
                    compare(tab, row, cap, v, reqs)
    errors = [e for p in players for e in p.rebuild_errors]
    return tab, games, decisions, errors


def feed_games(player, paths, viewer=0, limit=None, race_start=False):
    """Feeds one seat's messages of recorded games to player; yields (row, request) after each decision point."""
    games, game, first, tag = 0, None, True, None
    for path in paths:
        for row in read(path):
            if row["game"] != game:
                game, first = row["game"], True
                games += 1
                if limit and games > limit:
                    return
                tag = f"battle-gen1ou-{games}"
            req = None
            for who, kind, text in row["msgs"]:
                if int(who[1]) - 1 != viewer:
                    continue
                if kind == "request" and "|request|" in text:
                    body = text.split("|request|", 1)[1]
                    req = json.loads(body) if body else None
                if first and kind == "update":
                    head, sep, rest = text.partition("\n|\n")
                    head = "|init|battle\n|title|Truth One vs. Truth Two\n" + head
                    if race_start or not sep:
                        text = head + sep + rest
                    else:
                        player.feed(tag, head)
                        text = "|" + "\n" + rest
                player.feed(tag, text)
            first = False
            yield row, req


def play(paths, ckpt, limit=20, viewer=0, trace=None, **kw):
    """The whole bot, offline: every decision of one seat in recorded games is really made (rebuild, search, policy
    fallback, trace) on the CPU. The battles go on as recorded whatever it chooses; what is checked is that every
    decision gets a legal answer, and how often the search fails."""
    import time
    p = offline_player(["Truth One", "Truth Two"][viewer], play=dict(ckpt=ckpt, **kw))
    if trace:
        from pathlib import Path
        Path(trace).mkdir(parents=True, exist_ok=True)
        p.trace_dir = trace
    n = illegal = 0
    t0 = time.time()
    for row, req in feed_games(p, paths, viewer, limit):
        if req is None or req.get("wait") or row["truth"]["ended"] or not p.orders:
            continue
        n += 1
        msg = p.orders[-1]
        if req.get("forceSwitch"):
            ok = msg.startswith("/choose switch")
        else:
            act = req["active"][0]
            ids = [m["id"] for m in act["moves"] if not m.get("disabled")]
            ok = (msg.startswith("/choose switch") and not act.get("trapped")) or (msg.startswith("/choose move") and (msg.split()[2] in ids or msg.split()[2].isdigit())) \
                or msg == "/choose default"
        illegal += not ok
        if not ok:
            print("  not legal:", row["game"], row["step"], msg, json.dumps(req.get("active"))[:200])
        p.orders.clear()
    times = sorted(t for ts in p.times.values() for t in ts)
    out = {"decisions": n, "not_legal": illegal, "search_failures": p.search_errors, "rescued_by_safe_state": getattr(p, "safe_searches", None),
           "adapter_errors": getattr(p, "errors", 0), "tracker_lines_skipped": getattr(p, "tracker_errors", None),
           "last_resort_choices": getattr(p, "last_resorts", None), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out))
    return out


def report(tab, games, decisions, errors, out=None, bad=None):
    print(f"{games} games, {decisions} decisions compared; rebuild errors: {len(errors)}")
    for e in errors[:5]:
        print("   ", e)
    rows = []
    for kind in ("knowable", "hidden", "info"):
        print(f"\n== {kind} ==\n{'field':38s} {'differ':>8s} {'of':>8s} {'rate':>8s}")
        for (k, f), n in sorted(tab.n.items()):
            if k != kind:
                continue
            b = tab.bad[(k, f)]
            rows.append({"kind": k, "field": f, "bad": b, "n": n})
            print(f"{f:38s} {b:8d} {n:8d} {b / n:8.3%}" + (f"   e.g. {compact(tab.ex[(k, f)][0])}" if b else ""))
    if out:
        json.dump({"games": games, "decisions": decisions, "rebuild_errors": len(errors), "rows": rows}, open(out, "w"), indent=1)
    if bad:
        with open(bad, "w") as f:
            for r in tab.all_bad:
                f.write(json.dumps(r, default=str) + "\n")
    return rows


def compact(r):
    return {k: v for k, v in r.items() if k in ("game", "step", "viewer", "mon", "got", "want")}


def write_teams(path, n, seed=0):
    """Realistic teams for gen1_truth.js --teams: our searched teams or pool teams against human ladder teams."""
    from pathlib import Path
    from agent.selfplay import sample_team
    rng = np.random.default_rng(seed)
    root = Path(__file__).resolve().parent.parent
    ours = [p.read_text() for p in sorted((root / "data/search_teams_v1").glob("*_team"))]
    human = json.load(open(root / "data/human_teams_v1.json"))["opp"]
    with open(path, "w") as f:
        for g in range(n):
            a = ours[g % len(ours)] if g % 2 == 0 else showdown_team(sample_team(rng))
            b = showdown_team(human[rng.integers(len(human))])
            p1, p2 = (a, b) if g % 4 < 2 else (b, a)
            f.write(json.dumps({"p1": p1, "p2": p2, "tag": "real"}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("truth", nargs="*")
    ap.add_argument("--write-teams")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--bad")
    ap.add_argument("--max-rate", type=float)
    ap.add_argument("--race-start", action="store_true", help="deliver the battle's opening as one message")
    ap.add_argument("--limit", type=int, help="games to read")
    ap.add_argument("--adapter", help="test this copy of showdown_player.py instead (e.g. the one before the fixes)")
    ap.add_argument("--play", metavar="CKPT", help="instead of comparing states: really make every decision of one seat with this "
                    "checkpoint (exact search at --search-tau, on the CPU) and count failures")
    ap.add_argument("--search-tau", type=float, default=0.02)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--trace")
    args = ap.parse_args()
    if args.write_teams:
        write_teams(args.write_teams, args.games, args.seed)
        sys.exit(0)
    if args.play:
        r = play(args.truth, args.play, limit=args.limit or 10, trace=args.trace, search_tau=args.search_tau, infer=args.infer)
        sys.exit(1 if r["not_legal"] or r["adapter_errors"] else 0)
    rows = report(*run(args.truth, limit=args.limit, race_start=args.race_start, adapter=args.adapter), out=args.out, bad=args.bad)
    if args.max_rate is not None:
        worst = [r for r in rows if r["kind"] == "knowable" and r["bad"] / r["n"] > args.max_rate]
        sys.exit(1 if worst else 0)
