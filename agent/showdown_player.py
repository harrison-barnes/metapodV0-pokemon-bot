"""Our 6v6 agent as a Pokemon Showdown player (poke-env), for battles on a local server.

Each turn rebuilds what the network sees in our own engine (agent/obs.py) from poke-env's view of
the battle: the two (6, 18) team arrays (`gen1py.VecEnv.team_state` layout) and the 102-byte
battle block of the two active Pokemon (data/dataset_smogon_v1/schema.json, the cartridge's
battle RAM). We are always side 0. Gaps between what Showdown shows and what the engine tracks:
- Opposing HP is shown as a percentage; exact HP is estimated from the species' max HP (level
  100, DV 15, full stat experience, as in our teams). Opposing stats likewise come from the
  species' base stats; ours come from the request.
  A percentage p stands for the exact HP values that round up to it; the middle one is used.
- Counters hidden in game (sleep and confusion turns, Disable and trapping turns, Substitute HP
  of the opponent, Bide damage) are zeroed by the observation anyway, so they are left at placeholders
  here (a sleeper at 2 turns); the search draws sleep, confusion and Substitute HP for each of its guesses
  (agent/search_exact.py, hidden=True) from the turns each has already used, which are counted here from
  the protocol (one |cant|...|slp per sleep turn, one |-activate|...|confusion per confused turn).
- Battle-status flags are set from what Showdown reports (confusion, Substitute, Leech Seed, Reflect, Light
  Screen, Mist, Focus Energy, recharging, Rage, Bide, Thrash, a charging two-turn move, Transform with the
  copied stats, types and moves, Conversion's types, Disable and the move it hit, the Toxic counter).
- Wrap/Clamp/Bind/Fire Spin: the turn after the first hit both requests show that the trap goes on (the
  trapper may only repeat its move, the trapped side may only "fight"). After that Showdown no longer says
  (all moves are offered with "maybeLocked"): whether the trap continues is hidden, like a sleep counter.
  The trapper's flag stays set with the continuation hits so far (searcher.trap), and a search that knows
  how (searcher.trap_uncertain) draws for each guess whether it goes on; for one that doesn't, it is drawn
  here once per decision. The trapping hit's damage is in wDamage.
- wDamage otherwise holds the damage of the last damaging move, as far as it showed (exact on our side, from
  percentages on the other), and each side's last announced move: what Counter reads.
- Team slots are fixed for the battle: ours in team order, the opponent's in order of appearance.
- Everything rebuilt here is checked against Showdown's own state by agent/test_adapter_truth.py.

    uv run --group showdown python -m agent.showdown_player \
        --ckpt checkpoints/spec2_00200.pt --search-tau 0.02 --infer --teams data/search_teams_v1 --accept SOME_USER --battles 10
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration, ShowdownServerConfiguration
from poke_env.battle import Effect, SideCondition, Status
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder

from agent.arena import load_policy
from agent.features import DEX, MAX_HP as SPECIES_MAX_HP
from agent.matchups import default_device
from agent.obs import (ACTIVE, ALIVE, HP, MAX_HP, MAX_PP, MOVES, MOVES_REVEALED, PP, PRESENT, REVEALED, SLEPT_BY_FOE,
                       SPECIES, STATUS)
from agent.selfplay import observe_on, sample_team

ROOT = Path(__file__).resolve().parent.parent
to_id = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
SPECIES_IDX = {to_id(s["name"]): i for i, s in enumerate(DEX["species"])}
MOVE_IDX = {to_id(m["name"]): m["id"] for m in DEX["moves"] if m}
SD_MOVE = {"hijumpkick": "highjumpkick", "psychicm": "psychic", "vicegrip": "visegrip"}   # our name -> Showdown id
for ours, sd in SD_MOVE.items():
    MOVE_IDX[sd] = MOVE_IDX.pop(ours)
MOVE_DATA = {m["id"]: m for m in DEX["moves"] if m}
NAMES = json.load(open(ROOT / "data/showdown_names.json"))
STAGE = [(25, 100), (28, 100), (33, 100), (40, 100), (50, 100), (66, 100), (1, 1), (15, 10), (2, 1), (25, 10), (3, 1),
         (35, 10), (4, 1)]
STATUS_BITS = {Status.PSN: 8, Status.TOX: 8, Status.BRN: 16, Status.FRZ: 32, Status.PAR: 64, Status.SLP: 2}
# Battle-status bits (engine/src/state.rs).
CONFUSED, MIST, FOCUS, SUB, RECHARGE, RAGE, SEEDED = 1 << 7, 1 << 1, 1 << 2, 1 << 4, 1 << 5, 1 << 6, 1 << 7
TOXIC, LIGHT_SCREEN, REFLECT, TRANSFORMED = 1, 2, 4, 8
SLEEP_BLOCKED, FREEZE_BLOCKED, SLEPT_BY_FOE_BIT = 1, 2, 4
SIDE = 50   # bytes per side in the battle block; wDamage at 100
TRAPPING, ATTACKS_LEFT = 1 << 5, 34     # bs1 USING_TRAPPING_MOVE; turns left byte
BIDING, THRASHING, CHARGING, INVULNERABLE = 1, 2, 1 << 4, 1 << 6      # bs1
TRAP_MOVES = {MOVE_IDX[m] for m in ("wrap", "clamp", "bind", "firespin")}
MOVE_NAME = {v: k for k, v in MOVE_IDX.items()}     # engine move index -> Showdown move id (decision traces)
TWO_TURN = {MOVE_IDX[m] for m in ("razorwind", "skullbash", "skyattack", "solarbeam", "dig", "fly")}
THRASH_MOVES = {MOVE_IDX[m] for m in ("thrash", "petaldance")}
STRUGGLE, RAGE_MOVE, BIDE_MOVE, MIMIC, TRANSFORM = (MOVE_IDX[m] for m in ("struggle", "rage", "bide", "mimic", "transform"))
# moves that leave the last damage as it was (Showdown data/mods/gen1/scripts.ts SKIP_LASTDAMAGE)
KEEP_DAMAGE = {MOVE_IDX[m] for m in (
    "confuseray", "conversion", "counter", "focusenergy", "glare", "haze", "leechseed", "lightscreen", "mimic", "mist", "poisongas",
    "poisonpowder", "recover", "reflect", "rest", "softboiled", "splash", "stunspore", "substitute", "supersonic", "teleport",
    "thunderwave", "toxic", "transform")}
TYPE_IDX = {to_id(t.replace("_TYPE", "")): i for i, t in enumerate(DEX["types"])}
TRAP_HITS = np.array([3, 3, 1, 1]) / 8    # continuation hits after the first: 1, 2, 3 or 4
CRASH = {MOVE_IDX["jumpkick"], MOVE_IDX["highjumpkick"]}
DRAIN = {MOVE_IDX[m] for m in ("absorb", "megadrain", "leechlife", "dreameater")}
ABSENT = SPECIES_IDX["magikarp"]          # stands in (fainted) for team slots an opponent with fewer than six doesn't have


def max_pp(mv):
    """Showdown's maximum PP: base PP with three PP Ups, capped at 61."""
    return min(61, MOVE_DATA[mv]["pp"] // 5 * 8) if mv in MOVE_DATA else 0


def hp_from_percent(pct, max_hp):
    """The middle of the exact HP values Showdown shows as pct% (it rounds up, and shows 99 for anything short of full)."""
    if pct <= 0:
        return 0
    if pct >= 100:
        return int(max_hp)
    lo = (pct - 1) * int(max_hp) // 100 + 1
    hi = pct * int(max_hp) // 100 if pct < 99 else int(max_hp) - 1
    return max(1, (lo + max(lo, hi)) // 2)


def condition(text):
    """'664/703 par' -> (664, 703); '0 fnt' -> (0, None). The opponent's are out of 100."""
    hp, _, mx = text.split()[0].partition("/")
    return int(hp), int(mx) if mx else None


def calc_stat(base, level=100, dv=15):
    return ((base + dv) * 2 + 63) * level // 100 + 5


def species_stats(sp):
    b = DEX["species"][sp]["base"]
    return [calc_stat(b[k]) for k in ("atk", "def", "spe", "spc")]


def showdown_team(sets):
    """Showdown export text for our set strings ("SPECIES:MOVE,MOVE,...")."""
    out = []
    for s in sets:
        sp, moves = s.split(":")
        name = NAMES["species"][to_id(sp)]
        mv = [NAMES["moves"][SD_MOVE.get(to_id(m), to_id(m))] for m in moves.split(",")]
        out.append(f"{name}\nEVs: 252 HP / 252 Atk / 252 Def / 252 SpA / 252 SpD / 252 Spe\n" + "\n".join(f"- {m}" for m in mv))
    return "\n\n".join(out)


class PoolTeams(Teambuilder):
    """A fresh team from our ladder set pool every battle: independent usage draws (sample_team), or
    with smogon=True teams built from Smogon's teammate and lead statistics (agent/teams.py)."""

    def __init__(self, seed=0, smogon=False):
        self.rng = np.random.default_rng(seed)
        if smogon:
            from agent.teams import TeamGen
            self.draw = TeamGen()
        else:
            self.draw = sample_team

    def yield_team(self):
        return self.join_team(self.parse_showdown_team(showdown_team(self.draw(self.rng))))


class DirTeams(Teambuilder):
    """A random team file (Showdown export format) from a directory every battle."""

    def __init__(self, path, seed=0):
        self.teams = [self.join_team(self.parse_showdown_team(p.read_text())) for p in sorted(Path(path).rglob("*_team"))]
        self.rng = np.random.default_rng(seed)

    def yield_team(self):
        return self.teams[self.rng.integers(len(self.teams))]


class Tracker:
    """What poke-env doesn't keep: slot order, who saw what, sleep-clause facts, last moves, and the Gen 1 volatile
    state that Showdown only shows as events (locks, counters, Transform, Disable, last damage)."""

    def __init__(self):
        self.role = None                       # our side ("p1" / "p2"), from the |player| lines
        self.own, self.opp = [], []            # team keys ("p1: Name") in slot order
        self.seen_own = set()                  # our Pokemon the opponent has seen
        self.used = {}                         # key -> move ids it has used (revealed), its own moves only
        self.pp_used = {}                      # key -> {move id: PP spent}, as a player can count it
        self.own_pp = {}                       # our key -> {move id: PP left} as the last request with PP showed it
        self.spent = {}                        # key -> {move id: PP spent since that request}
        self.base_moves = {}                   # our key -> its move ids at the start (before Mimic / Transform)
        self.species = {}                      # key -> species index, from the switch lines
        self.teamsize = {}                     # role -> team size
        self.slept_by_foe = set()              # keys asleep from the other side's move
        self.slp = {}                          # key -> turns of its present sleep already used
        self.last_move = {}                    # role -> move id
        self.volatile = {}                     # key -> {"reflect", "lightscreen", "transform"} (Gen 1 volatiles)
        # role -> the active Pokemon's volatile state: conf_n (confused turns used), toxic (counter), disable
        # {mv, n}, charging (move), thrash (uses), bide (turns), rage, mimic (move), types (Conversion), transform {...}
        self.vol = {"p1": {}, "p2": {}}
        self.past = []                         # (team, battle bytes, opponent's action) at our decisions (search infer)
        self.pending = None                    # our last decision's state, until the opponent's choice shows
        self.recharge = set()                  # keys that must recharge (-mustrecharge), until they do
        self.trap = {}                         # trapper key -> {"h": continuation hits so far, "target": key, "dmg": HP lost}
        self.trap_last = None                  # trapper whose hit the next -damage belongs to
        self.hp = {}                           # key -> (hp, max) as last shown
        self.last_damage = 0                   # damage of the last damaging move, as far as it showed (Counter)
        self.damage_shown = True               # False when it went into a Substitute (the amount is then a guess)
        self.dmg_ctx = None                    # (user key, move) whose -damage lines set last_damage
        # Gen 1 stat penalties of the actives (see _track_stats): key -> [speed quarterings, attack halvings] in force
        self.drops, self.status, self.active, self.actor = {}, {}, {}, None
        self.t0 = time.time()                  # when the battle's first message came
        self.rated, self.rating = False, None  # |rated| seen; (before, after) from the ladder update at the end
        self.errors = 0                        # tracker lines that raised (each is skipped)
        self.search_fail, self.safe_used = 0, 0
        self.searched, self.forced = 0, 0      # decisions the search answered / with a single legal option
        self.await_opp = None                  # trace row waiting for the opponent's action
        self.requests, self.answered = 0, 0    # requests for a decision received / answered
        self.trap_now = ({}, False)            # the last rebuild's traps ({side: hits so far}) and whether one is uncertain
        self.clock = None                      # (seconds left this turn, in total) as the battle timer last told us

    def exact(self, key, cond):
        """Exact HP for a shown condition: ours as shown, the opponent's from its percentage."""
        hp, mx = cond
        if mx == 100 and self.species.get(key) is not None:
            return hp_from_percent(hp, SPECIES_MAX_HP[self.species[key]])
        return hp


class _WithholdLogin(logging.Filter):
    """Keeps the login exchange (the server's challenge and our signed reply) out of the log."""

    def filter(self, record):
        msg = record.getMessage()
        if "/trn " in msg or "challstr" in msg:
            record.msg, record.args = "(login message withheld)", ()
        return True


def key_of(ident):
    """'p1a: Tauros' -> 'p1: Tauros' (poke-env's team dict keys)."""
    return ident[:2] + ident[3:] if ident[3] != " " else ident


IDENT = re.compile(r"p[12][a-c]?: ")
PASS = 11   # engine action index: stay in without choosing a move


class OurPlayer(Player):
    def __init__(self, ckpt, device, sample=False, search_tau=None, search_k=8, teammates=False, critic=None, infer=False, rollouts=0, rollout_depth=2, opp_model=None, opp_mix=1.0, depth=1, width2=3, search_opts=None, **kw):
        # Before Player.__init__, which starts listening and may receive battle messages at once.
        self.policy = load_policy(ckpt, device) if ckpt else None    # None: state rebuild only (agent/test_adapter_truth.py)
        self.device, self.sample = device, sample
        self.searcher, self.search_errors, self.errors = None, 0, 0
        self.safe_searches, self.tracker_errors, self.last_resorts = 0, 0, 0
        self.dump_dir = None                         # --dump-errors: the state of every failed search
        self.times, self.results = {}, None          # decision seconds per battle; --results file
        self.state_dump, self.states = None, {}      # --dump-states: every decision's rebuilt state, per battle
        self.trace_dir, self.trace = None, {}        # --trace: every decision's options and choice, per battle
        self.budget = None                           # the search's seconds per decision as configured (--search-budget)
        self.rng = np.random.default_rng(int(kw.pop("adapter_seed", 0)) + 12345)
        if search_tau is not None:                   # agent/search_exact.py over the same checkpoint
            from agent.search_exact import ExactSearchPlayer
            self.searcher = ExactSearchPlayer(ckpt, device, critic=critic, k=search_k, tau=search_tau, teammates=teammates,
                                              infer=infer, rollouts=rollouts, rollout_depth=rollout_depth, opp_model=opp_model, opp_mix=opp_mix,
                                              depth=depth, width2=width2, **(search_opts or {}))
            self.budget = getattr(self.searcher, "budget", None)
        self.trackers = {}
        super().__init__(**kw)

    # -- message tracking ---------------------------------------------------------------------------
    async def _handle_battle_message(self, split_messages):
        """Our own tracking of every line, then poke-env's. Neither may cost the battle: a line our tracking chokes on
        is skipped, and if poke-env fails on a message that asked for a decision, Showdown's default choice is sent."""
        tag = split_messages[0][0].lstrip(">")
        tr = self.trackers.setdefault(tag, Tracker())
        if self.state_dump:
            tr.__dict__.setdefault("log", []).extend("|".join(m) for m in split_messages[1:])
        for m in split_messages[1:]:
            try:
                self._track(tr, tag, m)
            except Exception as e:
                tr.errors += 1
                self.tracker_errors += 1
                if tr.errors <= 3:
                    self.logger.error("tracking failed (%s) on %r: %r; line skipped", tag, "|".join(m)[:200], e)
        try:
            await super()._handle_battle_message(split_messages)
        except Exception as e:
            self.errors += 1
            self.logger.error("message handling failed (%s): %r", tag, e)
            if tr.answered < tr.requests:    # a request went unanswered: don't let the timer decide
                tr.answered = tr.requests
                self.last_resorts += 1
                await self.ps_client.send_message("/choose default", tag)

    def _track(self, tr, tag, m):
        if len(m) < 2:
            return
        kind = m[1]
        if kind == "rated":
            tr.rated = True
        elif kind == "raw" and len(m) > 2 and "rating:" in m[2]:
            r = re.search(re.escape(self.username) + r"'s rating: (\d+) &rarr; <strong>(\d+)</strong>", m[2])
            if r:
                tr.rating = (int(r.group(1)), int(r.group(2)))
        elif kind == "player" and len(m) > 3 and to_id(m[3]) == to_id(self.username):
            tr.role = m[2]
        elif kind == "request" and len(m) > 2 and m[2]:
            body = "|".join(m[2:])
            if '"wait":true' not in body:
                tr.requests += 1
            if tr.role is None:
                r = re.search(r'"side":\{[^{}]*?"id":"(p[12])"', body)
                if r:
                    tr.role = r.group(1)
        elif kind == "teamsize" and len(m) > 3:
            tr.teamsize[m[2]] = int(m[3])
        elif kind == "inactive" and len(m) > 2:
            r = re.search(r"Time left: (\d+) sec this turn \| (\d+) sec total", "|".join(m[2:]))
            if r:                                # sent to us alone, with each request, while the timer is on
                tr.clock = (int(r.group(1)), int(r.group(2)))
        elif kind == "inactiveoff":
            tr.clock = None
        if len(m) < 3:
            return
        self._track_locks(tr, kind, m)
        self._track_stats(tr, kind, m)
        self._track_state(tr, kind, m)
        own = lambda k: self._is_own(k, tag)
        if kind in ("switch", "drag"):
            k = key_of(m[2])
            for v in list(tr.volatile):
                if v[:2] == k[:2]:
                    tr.volatile.pop(v)          # switching out clears the old active's volatiles
            if own(k):
                tr.seen_own.add(k)
            elif k not in tr.opp:
                tr.opp.append(k)
        elif kind == "-status" and m[3] == "slp":
            if not any("move: Rest" in x for x in m[4:]):
                tr.slept_by_foe.add(key_of(m[2]))
        elif kind in ("-curestatus", "faint"):
            tr.slept_by_foe.discard(key_of(m[2]))
        elif kind in ("-start", "-end") and to_id(m[3].replace("move: ", "")) in ("reflect", "lightscreen"):
            vs = tr.volatile.setdefault(key_of(m[2]), set())
            (vs.add if kind == "-start" else vs.discard)(to_id(m[3].replace("move: ", "")))
        elif kind == "-transform":
            tr.volatile.setdefault(key_of(m[2]), set()).add("transform")
        # the opponent's action after our last decision: for the trace, and (in engine indices) for the search's inference
        if kind in ("move", "switch", "cant") and IDENT.match(m[2]) and not own(key_of(m[2])):
            k = key_of(m[2])
            called = kind == "move" and any(x.startswith("[from]") and to_id(x[6:]) != to_id(m[3]) for x in m[5:])
            mv = MOVE_IDX.get(to_id(m[3])) if kind == "move" else None
            a = None
            if kind == "move" and not called:
                used = tr.used.get(k, [])
                a = 4 if mv == STRUGGLE else (used.index(mv) if mv in used and used.index(mv) < 4 else None)
            elif kind == "switch":
                a = 5 + tr.opp.index(k) if k in tr.opp and tr.opp.index(k) < 6 else None
            if tr.await_opp is not None and not called:
                row, known = tr.await_opp
                row["opp_did"] = to_id(m[3]) if kind == "move" else ("switch:" + to_id(k[4:]) if kind == "switch" else "cant:" + to_id(m[3]))
                row["opp_did_i"] = a
                if kind == "move":       # a move it had not shown before our decision: no column of the search matches it
                    row["opp_did_new"] = mv not in known
                tr.await_opp = None
            if tr.pending is not None:
                if a is not None and not any("[from]" in x for x in m[4:]):
                    tr.past.append((*tr.pending, a))
                if a is not None or kind == "cant":
                    tr.pending = None
        elif kind == "faint" and IDENT.match(m[2]) and not own(key_of(m[2])):
            # it fainted before it acted: what it does next (a replacement) is no answer to our last decision
            tr.pending, tr.await_opp = None, None
        elif kind == "turn":
            tr.pending, tr.await_opp, tr.dmg_ctx = None, None, None

    @staticmethod
    def _track_locks(tr, kind, m):
        """Recharging and partial trapping, which poke-env doesn't keep for Gen 1."""
        if kind in ("switch", "drag"):
            k = key_of(m[2])
            tr.recharge = {x for x in tr.recharge if x[:2] != k[:2]}
            for t in [t for t, e in tr.trap.items() if k[:2] in (t[:2], e["target"][:2])]:
                tr.trap.pop(t)                   # a switch on either side ends the trap (SendOutMon)
            if len(m) > 3:
                tr.species[k] = SPECIES_IDX.get(to_id(m[3].split(",")[0]))
            if len(m) > 4:
                tr.hp[k] = condition(m[4])
            tr.dmg_ctx = None
        elif kind in ("-damage", "-heal", "-sethp") and len(m) > 3:
            k, new = key_of(m[2]), condition(m[3])
            old = tr.hp.get(k)
            lost = tr.exact(k, old) - tr.exact(k, new) if old is not None else 0
            frm = next((to_id(x[6:]) for x in m[4:] if x.startswith("[from]")), None)
            e = tr.trap.get(tr.trap_last)
            if kind == "-damage" and e is not None and e["target"] == k and old is not None:
                e["dmg"] = max(0, lost)
            tr.trap_last = None
            ctx = tr.dmg_ctx                     # a move's damage to its target (or the crash of a missed Jump Kick), or
            if kind == "-damage" and (frm == "confusion" or (frm is None and ctx is not None and (ctx[0] != k or ctx[1] in CRASH))):
                tr.last_damage = max(0, lost)    # the confusion self-hit: what Counter doubles
                tr.damage_shown = True
                if frm is None and ctx[1] in DRAIN and lost > 0:
                    tr.last_damage = max(1, lost // 2)   # after a draining move it is the amount drained
            if kind == "-damage" and frm == "confusion":
                tr.trap.pop(k, None)             # hurting itself ends its trapping move
            tr.hp[k] = new
        elif kind == "-mustrecharge":
            tr.recharge.add(key_of(m[2]))
        elif kind == "cant" and len(m) > 3:
            k = key_of(m[2])
            if m[3] in ("recharge", "flinch"):   # flinching replaces the recharge turn
                tr.recharge.discard(k)
            tr.trap.pop(k, None)                 # a trapper that can't move (full paralysis) loses its hold
            tr.dmg_ctx = None
        elif kind == "-status" and len(m) > 3 and m[3] == "slp":
            tr.recharge.discard(key_of(m[2]))    # put to sleep while it had to recharge: the recharge is gone
        elif kind == "faint":
            k = key_of(m[2])
            tr.recharge.discard(k)
            for t in [t for t, e in tr.trap.items() if k in (t, e["target"])]:
                tr.trap.pop(t)
        elif kind == "move" and len(m) > 3:
            k, mv = key_of(m[2]), MOVE_IDX.get(to_id(m[3]))
            if mv not in TRAP_MOVES:
                tr.trap.pop(k, None)
                return
            foe = tr.active.get("p2" if k[:2] == "p1" else "p1")
            tr.recharge.discard(foe)             # a trapping move cancels the target's recharge, even if it misses
            if len(m) < 5 or not m[4] or any("[miss]" in x for x in m[4:]):
                tr.trap.pop(k, None)             # a miss ends it (and clears the flag)
            else:
                tgt, e = key_of(m[4]), tr.trap.get(k)
                cont = any(x.startswith("[from]") and to_id(x[6:]) == to_id(m[3]) for x in m[5:])
                if e is not None and e["target"] == tgt and cont:
                    e["h"] += 1                  # the same trap goes on
                else:
                    tr.trap[k] = {"h": 0, "target": tgt, "dmg": None}     # a new one (also when re-used after it ended)
                tr.trap_last = k

    @staticmethod
    def _track_stats(tr, kind, m):
        """The paralysis (speed / 4) and burn (attack / 2) penalties as Gen 1 keeps them (Showdown's modifiedStats,
        data/mods/gen1/scripts.ts): applied once when the status lands or the Pokemon comes in with it; NOT undone when
        the status is cured (Rest); undone for a stat when that stat's stage changes (Agility, Swords Dance) or by
        Haze; and applied AGAIN to the Pokemon whose turn it isn't whenever a move changes a stat stage (Amnesia,
        Psychic's Special drop...) while it is paralysed / burned. Transform copies the target's."""
        if kind in ("switch", "drag"):
            k = key_of(m[2])
            tr.drops.pop(tr.active.get(k[:2]), None)
            st = (m[4].split() + [""])[1] if len(m) > 4 else ""
            tr.active[k[:2]], tr.status[k] = k, st
            tr.drops[k] = [int(st == "par"), int(st == "brn")]
        elif kind == "move":
            tr.actor = key_of(m[2])
        elif kind == "-status" and len(m) > 3:
            k = key_of(m[2])
            tr.status[k] = m[3]
            d = tr.drops.setdefault(k, [0, 0])
            d[0] += m[3] == "par"
            d[1] += m[3] == "brn"
        elif kind == "-curestatus":
            tr.status[key_of(m[2])] = ""
        elif kind in ("-boost", "-unboost") and len(m) > 4 and m[4] != "0":
            d = tr.drops.setdefault(key_of(m[2]), [0, 0])
            if m[3] == "spe":
                d[0] = 0
            elif m[3] == "atk":
                d[1] = 0
            # once per move (Special shows as spa + spd), and not for Rage's boost, which no move call makes
            if m[3] != "spd" and tr.actor and not any("[from]" in x for x in m[5:]):
                foe = tr.active.get("p2" if tr.actor[:2] == "p1" else "p1")
                if foe:
                    f = tr.drops.setdefault(foe, [0, 0])
                    f[0] += tr.status.get(foe) == "par"
                    f[1] += tr.status.get(foe) == "brn"
        elif kind == "-clearallboost":
            for k in tr.active.values():
                tr.drops[k] = [0, 0]
        elif kind == "-transform" and len(m) > 3:
            tr.drops[key_of(m[2])] = list(tr.drops.get(key_of(m[3]), [0, 0]))

    @staticmethod
    def _track_state(tr, kind, m):
        """The rest of the volatile state, from the events that show it (see Tracker.vol): which moves a Pokemon has
        shown as its own and the PP they cost, sleep and confusion turns used, the Toxic counter, Disable, two-turn
        moves, Thrash, Bide, Rage, Mimic, Conversion, Transform, and the last move's damage."""
        if not IDENT.match(m[2]):
            return
        k = key_of(m[2])
        role = k[:2]
        v = tr.vol.setdefault(role, {})
        name = to_id(m[3].replace("move: ", "")) if len(m) > 3 else ""
        if kind in ("switch", "drag", "faint"):
            tr.vol[role] = {}
        elif kind == "move":
            mv = MOVE_IDX.get(name)
            tags = m[5:]
            frm = next((to_id(x[6:]) for x in tags if x.startswith("[from]")), None)
            same, still = frm == name, "[still]" in tags
            called = frm is not None and not same        # by Metronome or Mirror Move: not a move of its own
            charging = still and mv in TWO_TURN
            if mv and mv != STRUGGLE and not called:
                # PP is spent when the move is chosen and gets used: not on a continuing turn (Wrap, Thrash, Rage), and
                # for a two-turn move on the turn it is released
                spend = (mv in TWO_TURN and v.get("charging") == mv) if same else not charging
                if "transform" in v:                     # the copied moves, 5 PP each; nothing of its own is shown
                    slot = mv
                    if spend:
                        u = v["transform"]["pp_used"]
                        u[mv] = u.get(mv, 0) + 1
                else:
                    slot = MIMIC if v.get("mimic") == mv and mv not in tr.used.get(k, []) and mv not in tr.base_moves.get(k, []) else mv
                    if slot not in tr.used.setdefault(k, []):
                        tr.used[k].append(slot)
                    if spend:
                        for book in (tr.pp_used, tr.spent):
                            c = book.setdefault(k, {})
                            c[slot] = c.get(slot, 0) + 1
            if mv:
                tr.last_move[role] = mv
                if "[miss]" in tags or (not charging and mv not in KEEP_DAMAGE and len(m) > 4 and m[4]):
                    tr.last_damage, tr.damage_shown = 0, True
                tr.dmg_ctx = (k, mv)
            if charging:
                v["charging"] = mv
            else:
                v.pop("charging", None)
            if mv in THRASH_MOVES:
                v["thrash"] = v.get("thrash", 0) + 1
            else:
                v.pop("thrash", None)
            v.pop("rage_new", None)
            if mv == RAGE_MOVE and "[miss]" not in tags and not v.get("rage"):
                v["rage"] = v["rage_new"] = True
            if "disable" in v:
                v["disable"]["n"] += 1
        elif kind == "cant":
            if name == "slp":
                tr.slp[k] = tr.slp.get(k, 0) + 1
            if name in ("par", "disable"):               # the turn is lost: a two-turn move, Thrash or Bide ends
                for f in ("charging", "thrash", "bide"):
                    v.pop(f, None)
                if "disable" in v:
                    v["disable"]["n"] += 1
        elif kind == "-prepare":
            mv = MOVE_IDX.get(name)
            if mv:
                v["charging"] = mv
        elif kind == "-activate":
            if name == "confusion":
                v["conf_n"] = v.get("conf_n", 0) + 1
            elif name == "bide":
                v["bide"] = v.get("bide", 1) + 1
            elif name == "substitute" and tr.dmg_ctx is not None:
                # the hit went into a Substitute: the amount doesn't show (a guess, for Counter)
                tr.last_damage = max(1, int(SPECIES_MAX_HP[tr.species[k]]) // 8) if tr.species.get(k) is not None else 40
                tr.damage_shown = False
        elif kind == "-start":
            if name == "confusion":
                v["conf_n"] = 0
                if "[silent]" in m[4:]:                  # the confusion that ends Thrash / Petal Dance
                    v.pop("thrash", None)
            elif name == "bide":
                v["bide"] = 1
            elif name == "disable" and len(m) > 4:
                mv = MOVE_IDX.get(to_id(m[4]))
                if mv:
                    v["disable"] = {"mv": mv, "n": 0}
                    if "transform" not in v and v.get("mimic") != mv and mv not in tr.used.setdefault(k, []):
                        tr.used[k].append(mv)            # the message names a move it has
            elif name == "mimic" and len(m) > 4:
                v["mimic"] = MOVE_IDX.get(to_id(m[4]))
            elif name == "typechange" and len(m) > 4:
                v["types"] = [TYPE_IDX[to_id(t)] for t in m[4].split("/") if to_id(t) in TYPE_IDX]
        elif kind == "-end":
            if name == "confusion":
                v.pop("conf_n", None)
            elif name in ("bide", "disable"):
                v.pop(name, None)
                if name == "bide":
                    tr.dmg_ctx = (k, BIDE_MOVE)          # the damage line that follows is Bide's
            elif name == "toxiccounter":                 # Haze
                v.pop("toxic", None)
            elif name == "substitute" and tr.dmg_ctx is not None:
                tr.last_damage = max(1, int(SPECIES_MAX_HP[tr.species[k]]) // 4) if tr.species.get(k) is not None else 80
                tr.damage_shown = False
                uv = tr.vol.get(tr.dmg_ctx[0][:2], {})
                if uv.pop("rage_new", None):             # a first Rage that breaks a Substitute doesn't start the rage
                    uv.pop("rage", None)
        elif kind == "-status" and len(m) > 3:
            if m[3] == "tox":
                v["toxic"] = 0
            elif m[3] == "slp":
                tr.slp[k] = 0
        elif kind == "-curestatus":
            tr.slp.pop(k, None)
        elif kind == "-damage":
            frm = next((to_id(x[6:]) for x in m[4:] if x.startswith("[from]")), None)
            if frm in ("psn", "brn", "leechseed") and "toxic" in v:
                v["toxic"] += 1                          # the counter rises with every poison, burn and Leech Seed hit
            elif frm == "confusion":
                for f in ("charging", "thrash", "bide"):
                    v.pop(f, None)
                if "disable" in v:
                    v["disable"]["n"] += 1
        elif kind == "-transform" and len(m) > 3 and IDENT.match(m[3]):
            into = key_of(m[3])
            fv = tr.vol.get(into[:2], {})
            v["transform"] = {"into": into, "species": tr.species.get(into), "types": list(fv["types"]) if "types" in fv else None,
                              "moves": None, "pp": None, "pp_used": {}}

    def _is_own(self, key, tag):
        tr = self.trackers.get(tag)
        role = tr.role if tr is not None and tr.role else None
        if role is None:
            b = self.battles.get(tag)
            role = b.player_role if b is not None else None
        return role is not None and key.startswith(role)

    # -- observation --------------------------------------------------------------------------------
    def arrays(self, battle):
        tr = self.trackers.setdefault(battle.battle_tag, Tracker())
        role, opp_role = battle.player_role, battle.opponent_role
        tr.role = tr.role or role
        if not tr.own:
            tr.own = list(battle.team.keys())
        stray = [k for k in tr.opp if k[:2] == role]      # ours, listed before we knew which side we were
        if stray:
            tr.opp = [k for k in tr.opp if k[:2] != role]
            tr.seen_own.update(stray)
        req = battle.last_request or {}
        for p in (req.get("side") or {}).get("pokemon", []):
            if p["ident"] not in tr.base_moves:
                tr.base_moves[p["ident"]] = [MOVE_IDX[i] for i in p.get("moves", []) if i in MOVE_IDX][:4]
        own_k = next((k for k in tr.own if k in battle.team and battle.team[k].active), None)
        self._own_request(battle, tr, req, own_k)
        v_own, v_opp = tr.vol.get(role, {}), tr.vol.get(opp_role, {})
        team = np.zeros((1, 2, 6, 18), np.int16)
        for j, k in enumerate(tr.own[:6]):
            mon = battle.team[k]
            t = team[0, 0, j]
            sp = tr.species.get(k)
            if sp is None:
                sp = SPECIES_IDX[to_id(mon.base_species)]
            mv = (tr.base_moves.get(k) or [MOVE_IDX[i] for i in mon.moves if i in MOVE_IDX])[:4]
            known, spent = tr.own_pp.get(k, {}), tr.spent.get(k, {})
            pp = [max(0, known.get(x, max_pp(x)) - spent.get(x, 0)) for x in mv]
            tf = v_own.get("transform") if mon.active else None
            if tf and tf.get("pp"):          # the engine reads PP by slot: while transformed, the copied moves' PP
                pp = list(tf["pp"])
            t[SPECIES], t[MOVES] = sp, mv + [0] * (4 - len(mv))
            t[HP], t[MAX_HP] = mon.current_hp or 0, mon.max_hp or SPECIES_MAX_HP[sp]
            t[STATUS] = STATUS_BITS.get(mon.status, 0) if not mon.fainted else 0
            t[PP] = (pp + [0] * 4)[:4]
            t[REVEALED] = k in tr.seen_own
            t[MOVES_REVEALED] = sum(1 << i for i, x in enumerate(mv) if x in tr.used.get(k, []))
            t[ACTIVE], t[ALIVE], t[PRESENT] = bool(mon.active), not mon.fainted, 1
            t[SLEPT_BY_FOE] = k in tr.slept_by_foe and mon.status == Status.SLP
        size = tr.teamsize.get(opp_role, 6)
        for j in range(6):
            t = team[0, 1, j]
            t[PRESENT], t[ALIVE] = 1, 1
            if j >= max(size, len(tr.opp)):  # a smaller team: the missing members count as fainted, not as unknown
                t[SPECIES], t[MAX_HP], t[ALIVE], t[REVEALED] = ABSENT, SPECIES_MAX_HP[ABSENT], 0, 1
                continue
            if j >= len(tr.opp) or tr.opp[j] not in battle.opponent_team:
                continue
            k = tr.opp[j]
            mon = battle.opponent_team[k]
            sp = tr.species.get(k)
            if sp is None:
                sp = SPECIES_IDX[to_id(mon.base_species)]
            tf = v_opp.get("transform") if mon.active else None
            if tf and tf["moves"] is None:   # transformed: its moves are those of the Pokemon it copied, ours, all known.
                tf["moves"] = list(self._current_moves(tr, tf["into"], battle))    # They go in the battle bytes (_side);
            mv = [x for x in tr.used.get(k, []) if x][:4]                          # its team row stays its own.
            pp = [max(0, max_pp(x) - tr.pp_used.get(k, {}).get(x, 0)) for x in mv]
            t[SPECIES], t[MOVES] = sp, mv + [0] * (4 - len(mv))
            t[MAX_HP] = SPECIES_MAX_HP[sp]
            t[HP] = 0 if mon.fainted else hp_from_percent(round(mon.current_hp_fraction * 100), SPECIES_MAX_HP[sp])
            t[STATUS] = STATUS_BITS.get(mon.status, 0) if not mon.fainted else 0
            t[PP] = pp + [0] * (4 - len(pp))
            t[REVEALED], t[MOVES_REVEALED] = 1, (1 << len(mv)) - 1
            t[ACTIVE], t[ALIVE] = bool(mon.active), not mon.fainted
            t[SLEPT_BY_FOE] = k in tr.slept_by_foe and mon.status == Status.SLP
        battle_bytes = np.zeros((1, 102), np.uint8)
        for s, mon, tm in ((0, battle.active_pokemon, team[0, 0]), (1, battle.opponent_active_pokemon, team[0, 1])):
            if mon is None:
                continue
            self._side(battle_bytes[0, s * SIDE:(s + 1) * SIDE], battle, mon, tm, s, tr)
        battle_bytes[0, 100:102] = np.array([min(65535, max(0, int(tr.last_damage)))], "<u2").view(np.uint8)
        self._locks(battle, tr, team[0], battle_bytes[0])
        return team, battle_bytes

    def _own_request(self, battle, tr, req, own_k):
        """What the request says about our active Pokemon overrides what was tracked: its PP, its moves while
        transformed, and whether it is locked into something."""
        if own_k is None or battle.force_switch or not req.get("active"):
            return
        v = tr.vol.setdefault(battle.player_role, {})
        rm = req["active"][0].get("moves", [])
        if rm and all("pp" in x for x in rm):
            cur = [MOVE_IDX.get(x["id"], 0) for x in rm][:4]
            if "transform" in v:
                v["transform"].update(moves=cur, pp=[int(x["pp"]) for x in rm][:4])
            else:
                base = tr.base_moves.get(own_k) or cur
                pp = tr.own_pp.setdefault(own_k, {})
                for i, x in enumerate(rm[:4]):           # by slot: a Mimic slot keeps Mimic's PP
                    pp[base[i] if i < len(base) else cur[i]] = int(x["pp"])
                tr.spent[own_k] = {}
                v["moves"] = cur
            for f in ("charging", "thrash", "bide", "rage"):
                v.pop(f, None)
            tr.recharge.discard(own_k)
        elif len(rm) == 1:
            mv = MOVE_IDX.get(rm[0].get("id"))
            if rm[0].get("id") == "recharge":
                tr.recharge.add(own_k)
            elif mv == RAGE_MOVE:
                v["rage"] = True
            elif mv == BIDE_MOVE:
                v.setdefault("bide", 1)
            elif mv in THRASH_MOVES:
                v.setdefault("thrash", 1)
            elif mv in TWO_TURN:
                v["charging"] = mv

    def _current_moves(self, tr, key, battle):
        """The moves of one of our Pokemon as they stand (Mimic's slot replaced)."""
        v = tr.vol.get(key[:2], {}) if key in battle.team and battle.team[key].active else {}
        cur = list(v.get("moves") or tr.base_moves.get(key, []))
        if v.get("mimic") and MIMIC in cur:
            cur[cur.index(MIMIC)] = v["mimic"]
        return cur

    def _locks(self, battle, tr, team, bb):
        """The opponent's recharge and a Wrap/Clamp/Bind/Fire Spin that goes on or may go on, into the battle bytes."""
        own_k = next((k for k in tr.own if k in battle.team and battle.team[k].active), None)
        opp_k = next((k for k in tr.opp if k in battle.opponent_team and battle.opponent_team[k].active), None)
        if opp_k in tr.recharge:
            bb[SIDE + 28] |= RECHARGE
        req = battle.last_request or {}
        act = (req.get("active") or [{}])[0]
        rm = act.get("moves", [])
        ids = [x.get("id") for x in rm]
        forced = len(rm) == 1 and "pp" not in rm[0]
        maybe = bool(act.get("maybeLocked"))
        own_mon = battle.team.get(own_k) if own_k else None
        # asleep, frozen, or locked into something of our own (Bide, Rage...): then the request says nothing about a trap
        helpless = (own_mon is not None and own_mon.status in (Status.SLP, Status.FRZ)) or (
            forced and ids != ["fight"] and MOVE_IDX.get(ids[0]) not in TRAP_MOVES)
        traps, uncertain = {}, False
        if not battle.force_switch:
            for side, k, other in ((0, own_k, opp_k), (1, opp_k, own_k)):
                e = tr.trap.get(k)
                if e is None:
                    continue
                if e["target"] != other or e["h"] >= 4:
                    tr.trap.pop(k)
                    continue
                if side == 0:                    # ours: the request shows only the move while it surely goes on
                    sure = forced and MOVE_IDX.get(ids[0]) in TRAP_MOVES
                    may = sure or maybe
                else:                            # theirs: we may only "fight" while it surely goes on
                    sure = e["h"] == 0 and (ids == ["fight"] or helpless)
                    may = sure or maybe or helpless
                if not may:
                    tr.trap.pop(k)               # a trap the request doesn't show is over
                    continue
                if not sure:
                    uncertain = True
                    if not getattr(self.searcher, "trap_uncertain", False):
                        # the search can't draw it: draw here whether it goes on, as likely as the hits so far make it
                        h = e["h"]
                        if self.rng.random() >= TRAP_HITS[h:].sum() / TRAP_HITS[max(0, h - 1):].sum():
                            continue
                o = side * SIDE
                bb[o + 27] |= TRAPPING
                bb[o + ATTACKS_LEFT] = 1             # placeholder: the search draws the turns left
                if e["dmg"] is not None:
                    bb[100:102] = np.array([min(65535, max(0, int(e["dmg"])))], "<u2").view(np.uint8)
                traps[side] = e["h"]
        tr.trap_now = (traps, uncertain)
        if self.searcher is not None:
            self.searcher.trap = {0: traps}

    def _side(self, out, battle, mon, tm, s, tr):
        a = int(np.argmax(tm[:, ACTIVE])) if tm[:, ACTIVE].any() else 0
        t = tm[a]
        sp = int(t[SPECIES])
        role = battle.player_role if s == 0 else battle.opponent_role
        v = tr.vol.get(role, {})
        tf = v.get("transform")
        key = next((k for k, x in (battle.team if s == 0 else battle.opponent_team).items() if x is mon), "")
        u16 = lambda off, vals: out[off:off + 2 * len(vals)].view("<u2").__setitem__(slice(None), vals)
        own_stats = lambda m: [m.stats["atk"], m.stats["def"], m.stats["spe"], m.stats["spa"]] if m is not None and m.stats and m.stats.get("atk") else None
        types = v.get("types") or DEX["species"][sp]["types"]
        if tf:                                   # Transform: the target's stats (before stages), types and moves
            into = tf["into"]
            unmod = (own_stats(battle.team.get(into)) if s == 1 else None) or species_stats(tf["species"] if tf["species"] is not None else sp)
            if tf["types"]:
                types = tf["types"]
            elif tf["species"] is not None:
                types = DEX["species"][tf["species"]]["types"]
        else:
            unmod = (own_stats(mon) if s == 0 else None) or species_stats(sp)
        b = mon.boosts
        mods = [b.get("atk", 0), b.get("def", 0), b.get("spe", 0), b.get("spa", 0), b.get("accuracy", 0), b.get("evasion", 0)]
        stats = [min(999, max(1, x * STAGE[m + 6][0] // STAGE[m + 6][1])) for x, m in zip(unmod, mods[:4])]
        # the paralysis / burn penalties in force (Tracker.drops); without tracked history, one each while the status lasts
        par_q, brn_h = tr.drops.get(key) or (int(mon.status == Status.PAR), int(mon.status == Status.BRN))
        for _ in range(int(par_q)):
            stats[2] = max(1, stats[2] // 4)
        for _ in range(int(brn_h)):
            stats[0] = max(1, stats[0] // 2)
        u16(0, [int(t[HP])])
        u16(2, stats)
        u16(10, unmod)
        out[20] = t[STATUS]
        out[21:27] = [m + 7 for m in mods]
        eff = mon.effects
        bs1 = ((CONFUSED if Effect.CONFUSION in eff else 0) | (BIDING if "bide" in v else 0) | (THRASHING if "thrash" in v else 0)
               | (CHARGING if "charging" in v else 0)
               | (INVULNERABLE if v.get("charging") in (MOVE_IDX["fly"], MOVE_IDX["dig"]) else 0))
        bs2 = ((SUB if Effect.SUBSTITUTE in eff else 0) | (SEEDED if Effect.LEECH_SEED in eff else 0)
               | (MIST if Effect.MIST in eff else 0) | (FOCUS if Effect.FOCUS_ENERGY in eff else 0)
               | (RAGE if Effect.RAGE in eff or v.get("rage") else 0)
               | (RECHARGE if Effect.MUST_RECHARGE in eff or (s == 0 and (self._recharging(battle) or key in tr.recharge)) else 0))
        conds = battle.side_conditions if s == 0 else battle.opponent_side_conditions
        vol = tr.volatile.get(key, set())
        bs3 = ((TOXIC if "toxic" in v else 0) | (TRANSFORMED if "transform" in vol or tf else 0)
               | (REFLECT if "reflect" in vol or Effect.REFLECT in eff or SideCondition.REFLECT in conds else 0)
               | (LIGHT_SCREEN if "lightscreen" in vol or SideCondition.LIGHT_SCREEN in conds else 0))
        out[27], out[28], out[29] = bs1, bs2, bs3
        out[31] = min(255, v.get("toxic", 0))
        if "thrash" in v or "bide" in v:         # two or three more turns from the first: the likelier count left
            out[ATTACKS_LEFT] = max(1, 3 - v.get("thrash", v.get("bide", 1)))
        # the active Pokemon's moves as they stand: ours from the request, the opponent's as far as shown
        if s == 0:
            cur = list(tf["moves"]) if tf and tf.get("moves") else list(self._current_moves(tr, key, battle))
        elif tf and tf.get("moves"):
            cur = list(tf["moves"])
        else:
            cur = [int(x) for x in t[MOVES]]
            if v.get("mimic") and MIMIC in cur:
                cur[cur.index(MIMIC)] = v["mimic"]
        cur = (cur + [0] * 4)[:4]
        d = v.get("disable")
        if d and d["mv"] in cur:                 # which move is public; the turns left (1-8 at first) are not
            out[32] = ((cur.index(d["mv"]) + 1) << 4) | max(1, (9 - min(d["n"], 7)) // 2)
            out[33] = d["mv"]
        last = tr.last_move.get(role, 0)
        out[36] = last
        if last:
            md = MOVE_DATA[last]
            out[37:41] = [last, md["power"], md["type"], md["accuracy"]]
        out[41:43] = (list(types) * 2)[:2]
        out[43:47] = cur
        out[47] = sp
        others = tr.own if s == 1 else tr.opp     # the side this one's sleep moves would hit
        out[48] = any(k in tr.slept_by_foe for k in others)
        foes = [k for k in (tr.opp if s == 0 else tr.own)]
        opp_team = battle.opponent_team if s == 0 else battle.team
        sleep_blocked = any(k in tr.slept_by_foe and k in opp_team and opp_team[k].status == Status.SLP for k in foes)
        freeze_blocked = any(k in opp_team and opp_team[k].status == Status.FRZ for k in foes)
        out[49] = (SLEEP_BLOCKED * sleep_blocked) | (FREEZE_BLOCKED * freeze_blocked) | (SLEPT_BY_FOE_BIT * bool(t[SLEPT_BY_FOE]))

    @staticmethod
    def _recharging(battle):
        req = battle.last_request or {}
        act = (req.get("active") or [{}])[0]
        return any(m.get("id") == "recharge" for m in act.get("moves", []))

    def _seen(self, battle, tr):
        """{(side, slot): [sleep turns used, confused turns used]} for the search's draws of the turns left."""
        seen = {}
        for side, keys, role in ((0, tr.own, battle.player_role), (1, tr.opp, battle.opponent_role)):
            team = battle.team if side == 0 else battle.opponent_team
            for j, k in enumerate(keys[:6]):
                active = k in team and team[k].active
                e = [tr.slp.get(k, 0), tr.vol.get(role, {}).get("conf_n", 0) if active else 0]
                if e[0] or e[1]:
                    seen[(side, j)] = e
        return seen

    def rebuild(self, battle):
        """Everything a decision starts from: the engine state rebuilt from what we observed (team (2, 6, 18), battle
        bytes (102,)), our legal actions in the engine's indices with what each sends to Showdown, and what the search
        needs for the hidden counters."""
        tr = self.trackers.setdefault(battle.battle_tag, Tracker())
        team, bb = self.arrays(battle)
        mask, target = self.legal(battle, tr)
        traps, uncertain = tr.trap_now
        return {"team": team[0], "bb": bb[0], "mask": mask, "target": target, "seen": self._seen(battle, tr), "trap": dict(traps),
                "trap_uncertain": uncertain, "damage_shown": tr.damage_shown}

    # -- decisions ----------------------------------------------------------------------------------
    def legal(self, battle, tr):
        """(mask (12,), {action: poke-env move or Pokemon}). Moves by their slot in the request; a single forced
        "move" (Fight when asleep, frozen or trapped; Recharge; the move a lock repeats) is the engine's pass."""
        mask, target = np.zeros(12, bool), {}
        if not battle.force_switch:
            rm = ((battle.last_request or {}).get("active") or [{}])[0].get("moves", [])
            avail = {}
            for m in battle.available_moves:
                avail.setdefault(m.id, m)
            if len(rm) == 1 and "pp" not in rm[0] and battle.available_moves:
                a = 4 if rm[0].get("id") == "struggle" else PASS
                mask[a], target[a] = True, battle.available_moves[0]
            else:
                for i, x in enumerate(rm[:4]):
                    if not x.get("disabled") and x.get("id") in avail:
                        a = 4 if x["id"] == "struggle" else i
                        mask[a], target[a] = True, avail[x["id"]]
                if not mask[:4].any():           # nothing matched the request: poke-env's own list, in its order
                    active = battle.active_pokemon
                    own_moves = [i for i in active.moves if i in MOVE_IDX][:4] if active is not None else []
                    for m in battle.available_moves:
                        a = 4 if m.id == "struggle" else PASS if m.id in ("fight", "recharge") else (own_moves.index(m.id) if m.id in own_moves else None)
                        if a is not None:
                            mask[a], target[a] = True, m
        for p in battle.available_switches:
            k = next((k for k, v in battle.team.items() if v is p), None)
            if k in tr.own:
                a = 5 + tr.own.index(k)
                mask[a], target[a] = True, p
        return mask, target

    def choose_move(self, battle):
        t0 = time.perf_counter()
        tr = self.trackers.setdefault(battle.battle_tag, Tracker())
        tr.answered = tr.requests
        try:
            return self._choose(battle, t0)
        except Exception as e:                       # never hang a battle on an adapter bug
            self.errors += 1
            self.logger.error("choose_move failed (%s): %r; playing its strongest move", battle.battle_tag, e)
            return self._last_resort(battle)
        finally:                                     # decision times, against the server's turn timer
            self.times.setdefault(battle.battle_tag, []).append(time.perf_counter() - t0)

    def _last_resort(self, battle):
        """Without a rebuilt state: the available move with the most power, else any legal choice."""
        self.last_resorts += 1
        try:
            moves = [m for m in battle.available_moves if not battle.force_switch]
            if moves:
                return self.create_order(max(moves, key=lambda m: (m.base_power or 0) * (m.accuracy if isinstance(m.accuracy, (int, float)) else 1)))
            return self.choose_random_move(battle)
        except Exception:
            return self.choose_default_move()

    @torch.no_grad()
    def _choose(self, battle, t0=None):
        t0 = time.perf_counter() if t0 is None else t0
        tr = self.trackers.setdefault(battle.battle_tag, Tracker())
        st = self.rebuild(battle)
        team, bb, mask, target = st["team"][None], st["bb"][None], st["mask"], st["target"]
        if self.state_dump:
            self.states.setdefault(battle.battle_tag, []).append((battle.turn, bool(battle.force_switch), team[0], bb[0]))
        if self.searcher is not None and not battle.force_switch:     # for the search's inference, and the belief's memory
            tr.pending = (team[0].astype(np.int64), bytes(bb[0]))
        if mask.sum() == 0:
            return self.choose_default_move()       # recharge or another forced action
        if mask.sum() == 1:
            tr.forced += 1
            return self.create_order(target[int(mask.argmax())])
        if self.searcher is not None:
            # A failed search costs this decision its search, not the battle its searches: a second try on a state
            # stripped of what most often poisons one (the opponent's shown moves, exotic locks, the inference
            # history); then the plain policy; and the next decision starts from scratch.
            for safe in (False, True):
                try:
                    a = self._search(battle, team, bb, mask, st, safe=safe)
                    self._trace(battle, tr, target, a, dict(self.searcher.last, safe=safe), t0)
                    self.safe_searches += safe
                    tr.safe_used += safe
                    tr.searched += 1
                    return self.create_order(target[a])
                except Exception as e:
                    if not safe:
                        self.search_errors += 1
                        tr.search_fail += 1
                    if tr.search_fail <= 3 or safe:
                        self.logger.error("search failed (%s, turn %s%s): %r; %s", battle.battle_tag, battle.turn, ", safe mode" if safe else "",
                                          e, "using the policy" if safe else "retrying on a simplified state")
                    if self.dump_dir and not safe:   # keep the state for debugging
                        try:
                            np.savez(Path(self.dump_dir) / f"{battle.battle_tag}_{battle.turn}_{tr.search_fail}.npz", team=team, bb=bb, mask=mask,
                                     request=json.dumps(battle.last_request), error=repr(e), own=np.array(tr.own), opp=np.array(tr.opp))
                        except Exception:
                            pass
        o = observe_on(self.policy, team, bb, 0, self.device)
        logits, _ = self.policy(o, torch.from_numpy(mask[None]).to(self.device))
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e9)
        a = (torch.distributions.Categorical(logits=logits).sample() if self.sample else logits.argmax(-1)).item()
        if not mask[a]:
            a = int(np.flatnonzero(mask)[0])
        self._trace(battle, tr, target, a, {"policy": torch.softmax(logits[0], -1).cpu().numpy()}, t0)
        return self.create_order(target[a])

    def _trace(self, battle, tr, target, a, info, t0=None):
        try:                                         # a trace that can't be written must not cost the decision
            self._trace_row(battle, tr, target, a, info, t0)
        except Exception as e:
            if not getattr(self, "_trace_failed", False):
                self._trace_failed = True
                self.logger.error("trace row failed (%s): %r", battle.battle_tag, e)

    def _trace_row(self, battle, tr, target, a, info, t0=None):
        """--trace: one row per decision (turn, our options with the search's equilibrium mix and values or the
        policy's probabilities, the choice), written per battle when it ends. Rows only ever gain keys (the dashboard
        reads turn, chose, active, opp_active, opp_last_move, value and options[].a / mix / p / mean_v / worst_v):
        - t: Unix time of the decision; ms: milliseconds it took; t0 (first row): when the battle started;
        - options[].q: the action's value against the opponent's equilibrium mix (Q y); options[].prior: the policy's
          probability of it among the candidates;
        - opp: the opponent's candidate actions (a: name as far as known, i: engine action) with their equilibrium
          probability y and prior; value is x Q y, the regularised game's value; maxmin: the best pure action's worst
          case; best_q: max Q y (best_q - q of the chosen action is the decision's regret against y); kl: KL(mix || prior);
          size: [our candidates, theirs]; worlds: guessed worlds averaged; secs: the search's own seconds;
        - opp_did / opp_did_i (filled in when it shows): what the opponent then did ("move id", "switch:species",
          "cant:reason"), and its engine action (the `i` of `opp`) when it has one; opp_did_new: a move it had not
          shown before, which no single column stands for (that slot held a different move in each guess): leave
          those rows out of a log loss;
        - safe: the search ran on the simplified state after failing on the full one."""
        if not self.trace_dir:
            return
        def name(k):                                 # target: engine action -> poke-env Move or Pokemon
            t = target.get(int(k))
            return f"a{k}" if t is None else (getattr(t, "id", None) or getattr(t, "species", None) or str(t))
        opp_k = next((k for k in tr.opp if k in battle.opponent_team and battle.opponent_team[k].active), None)
        shown = [x for x in tr.used.get(opp_k, []) if x][:4]
        def oname(k):                                # the opponent's engine action, as far as it can be named
            k = int(k)
            if k < 4:
                return MOVE_NAME.get(shown[k], f"move{k + 1}") if k < len(shown) else f"unseen-move{k + 1}"
            if k == 4:
                return "struggle"
            if k == PASS:
                return "pass"
            return "switch:" + (to_id(tr.opp[k - 5][4:]) if k - 5 < len(tr.opp) else f"unseen{k - 5}")
        rows = self.trace.setdefault(battle.battle_tag, [])
        row = {"turn": battle.turn, "chose": name(a), "active": getattr(battle.active_pokemon, "species", None),
               "opp_active": getattr(battle.opponent_active_pokemon, "species", None),
               "opp_last_move": MOVE_NAME.get(tr.last_move.get(battle.opponent_role)), "t": round(time.time(), 2)}
        if t0 is not None:
            row["ms"] = int(1000 * (time.perf_counter() - t0))
        if not rows:
            row["t0"] = round(tr.t0, 2)
        if "Q" in info:                              # search: our options, equilibrium mix, value vs their best reply
            Q, mine = np.asarray(info["Q"], np.float64), info["mine"]
            x, y, px, py, theirs = (info.get(f) for f in ("mix", "y", "px", "py", "theirs"))
            ok = y is not None and len(y) == Q.shape[1]
            qy = Q @ np.asarray(y, np.float64) if ok else None
            row["options"] = [{"a": name(k),
                               "mix": None if x is None else round(float(x[i]), 3),
                               "mean_v": round(float(Q[i].mean()), 3), "worst_v": round(float(Q[i].min()), 3)}
                              for i, k in enumerate(mine)]
            for i, o in enumerate(row["options"]):
                if qy is not None:
                    o["q"] = round(float(qy[i]), 3)
                if px is not None and len(px) == len(mine):
                    o["prior"] = round(float(px[i]), 3)
            if "value" in info:
                row["value"] = round(info["value"], 3)  # expected value of the regularised equilibrium, our view (-1..1)
            row["size"] = [int(Q.shape[0]), int(Q.shape[1])]
            row["maxmin"] = round(float(Q.min(1).max()), 3)
            if qy is not None:
                row["best_q"] = round(float(qy.max()), 3)
            if x is not None and px is not None and len(px) == len(mine):
                xs, ps = np.asarray(x, np.float64), np.maximum(np.asarray(px, np.float64), 1e-9)
                row["kl"] = round(float(np.sum(xs[xs > 0] * np.log(xs[xs > 0] / ps[xs > 0]))), 4)
            if theirs is not None and len(theirs) == Q.shape[1]:
                row["opp"] = [dict({"a": oname(k), "i": int(k)}, **({"y": round(float(y[j]), 3)} if ok else {}),
                                   **({"prior": round(float(py[j]), 3)} if py is not None and len(py) == len(theirs) else {}))
                              for j, k in enumerate(theirs)]
            for f in ("worlds", "secs"):         # guessed worlds the root game averaged, and the search's own seconds
                if info.get(f) is not None:
                    row[f] = round(float(info[f]), 3) if f == "secs" else int(info[f])
            if info.get("safe"):
                row["safe"] = True
        else:
            p = info["policy"]
            row["options"] = [{"a": name(k), "p": round(float(p[k]), 3)} for k in np.flatnonzero(p > 0.01)]
        rows.append(row)
        if not battle.force_switch:
            tr.await_opp = (row, set(shown))
        self._live(battle)

    def _live(self, battle, ended=False):
        """--trace: the battle in progress (or just ended) to TRACE/live.json for the dashboard's live view."""
        if not self.trace_dir:
            return
        live = {"tag": battle.battle_tag, "opponent": battle.opponent_username, "turn": battle.turn,
                "url": f"https://play.pokemonshowdown.com/{battle.battle_tag}", "updated": time.time(),
                "ended": ended, "won": battle.won if ended else None, "rows": self.trace.get(battle.battle_tag, [])}
        tmp = Path(self.trace_dir) / "live.json.tmp"
        tmp.write_text(json.dumps(live))
        tmp.replace(Path(self.trace_dir) / "live.json")

    def _search(self, battle, team, bb, mask, st=None, safe=False):
        """Our action from ExactSearchPlayer, on the engine battle rebuilt from what we observed. safe: on a state
        stripped down to what is certain to load and be guessable (see _safe_state), without the inference history."""
        from agent.search import gen1py
        tr = self.trackers[battle.battle_tag]
        st = st or {}
        if safe:
            team, bb = self._safe_state(team, bb)
        env = gen1py.VecEnv(1, seed=int(self.searcher.rng.integers(1 << 62)))
        env.load(0, team[0].astype(np.int64), bytes(bb[0]), int(battle.turn))
        # The engine's choices, not Showdown's: asleep or frozen, Showdown still asks for a move (which
        # won't be used) where the engine only allows staying in (pass) or switching.
        emask = env.legal_mask()[0, 0]
        uncertain = bool(st.get("trap_uncertain")) and not safe
        if uncertain and getattr(self.searcher, "trap_uncertain", False):
            ours = mask.copy()                       # the search maps a move to pass in the guesses where the trap holds
        else:
            ours = emask & (mask | (np.arange(12) == PASS))
        if not ours.any():
            raise ValueError(f"no action legal in both: engine {np.flatnonzero(emask)}, showdown {np.flatnonzero(mask)}")
        m2 = np.zeros((1, 2, 12), bool)
        m2[0, 0] = ours
        self.searcher.past = {0: [] if safe else tr.past}
        self.searcher.seen = {0: {} if safe else {k: list(e) for k, e in st.get("seen", {}).items()}}   # turns asleep / confused, for the search's draws
        self.searcher.trap = {0: {} if safe else dict(st.get("trap", {}))}
        self.searcher.opp_ctx = {0: {"opponent": battle.opponent_username, "tag": battle.battle_tag}}    # the belief's per-battle context
        if self.budget is not None:              # never think longer than the battle timer can afford
            self.searcher.budget = self.clock_budget(self.budget, tr.clock)
        a = int(self.searcher._search(team, m2, 0, env, [0])[0])
        if a == PASS and not mask[PASS]:
            moves = mask.copy()
            moves[5:] = False
            if uncertain and moves.sum() > 1 and (bb[0, 27] | bb[0, SIDE + 27]) & TRAPPING:
                # we stay in while a trap may or may not go on: the move we send is used if it has ended, so choose
                # it as if it had
                b2 = bb.copy()
                b2[0, 27] &= ~TRAPPING & 0xFF
                b2[0, SIDE + 27] &= ~TRAPPING & 0xFF
                keep = self.searcher.last
                try:
                    a = self._search(battle, team, b2, moves, dict(st, trap={}, trap_uncertain=False))
                except Exception:
                    a = int(np.flatnonzero(moves)[0])
                self.searcher.last = keep
            else:                                    # any move will do: it isn't used
                a = int(np.flatnonzero(mask[:5])[0]) if mask[:5].any() else int(np.flatnonzero(mask)[0])
        assert mask[a], a
        return a

    @staticmethod
    def clock_budget(budget, clock):
        """The search's seconds for this decision: its configured budget, cut to what the battle timer leaves (10
        seconds of margin on the turn's limit, and a fifteenth of the total, which gains 10 seconds a turn)."""
        if clock is None:
            return budget
        turn, total = clock
        return max(0.05, min(budget, turn - 10, total / 15))

    @staticmethod
    def _safe_state(team, bb):
        """A copy of the rebuilt state with only what is sure to load and to be guessable: the opponent's Pokemon keep
        species, HP and status but none of their shown moves (the guesser then draws whole sets), and both actives
        lose the rare locks and counters (Disable, Bide, Thrash, two-turn moves, trapping, Transform)."""
        team, bb = team.copy(), bb.copy()
        opp = team[0, 1]
        opp[:, MOVES], opp[:, PP], opp[:, MOVES_REVEALED] = 0, 0, 0
        for s in (0, 1):
            o = s * SIDE
            bb[0, o + 27] &= CONFUSED                # bs1: keep confusion only
            bb[0, o + 29] &= ~TRANSFORMED & 0xFF
            bb[0, o + 18:o + 20] = 0                 # Bide damage
            bb[0, o + 32:o + 35] = 0                 # Disable, attacks left
        bb[0, SIDE + 43:SIDE + 47] = 0
        act = team[0, 0, :, ACTIVE].astype(bool)
        if act.any():                                # our active's moves as its team row has them
            bb[0, 43:47] = team[0, 0, int(act.argmax()), MOVES]
        return team, bb

    def _end_battle(self, battle, tr):
        """What the battle showed of the opponent, to the search's set model (searcher.end_battle, agent/search.py):
        its team rows as they stand at the end, without the placeholder for a slot a smaller team doesn't have, and
        its choices at our decisions."""
        hook = getattr(self.searcher, "end_battle", None)
        if hook is None or tr is None:
            return
        try:
            self.trackers[battle.battle_tag] = tr            # (already taken out by the caller; arrays() needs it)
            opp = self.arrays(battle)[0][0, 1].copy()
            opp[max(tr.teamsize.get(battle.opponent_role, 6), len(tr.opp)):] = 0
            hook(opponent=battle.opponent_username, tag=battle.battle_tag, opp_team=opp, past=tr.past, won=battle.won, turns=battle.turn)
        except Exception as e:
            self.logger.error("end_battle failed (%s): %r", battle.battle_tag, e)
        finally:
            self.trackers.pop(battle.battle_tag, None)

    def _battle_finished_callback(self, battle):
        tr = self.trackers.pop(battle.battle_tag, None)
        rows = self.states.pop(battle.battle_tag, None)
        if self.state_dump and rows:
            np.savez(Path(self.state_dump) / f"{battle.battle_tag}_{battle.player_role}.npz", turn=np.array([r[0] for r in rows]),
                     force=np.array([r[1] for r in rows]), team=np.stack([r[2] for r in rows]), bb=np.stack([r[3] for r in rows]),
                     won=battle.won, own=np.array(tr.own if tr else []), opp=np.array(tr.opp if tr else []),
                     log=np.array(getattr(tr, "log", [])))
        if tr is not None:
            self.opp_choices = getattr(self, "opp_choices", 0) + len(tr.past)
        self._end_battle(battle, tr)
        self._live(battle, ended=True)
        rows = self.trace.pop(battle.battle_tag, None)
        if self.trace_dir and rows:
            with open(Path(self.trace_dir) / f"{battle.battle_tag}.jsonl", "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
        t = self.times.pop(battle.battle_tag, [0.0])
        if self.results:                             # one row per battle, e.g. for a ladder run's record
            with open(self.results, "a") as f:
                # fallbacks: searches that failed in all battles so far (as before); battle_*: this battle's own counts
                f.write(json.dumps({"tag": battle.battle_tag, "opponent": battle.opponent_username, "won": battle.won,
                                    "turns": battle.turn, "rating": battle.rating, "opp_rating": battle.opponent_rating,
                                    "max_s": round(max(t), 2), "mean_s": round(float(np.mean(t)), 2),
                                    "fallbacks": self.search_errors, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "t0": round(tr.t0, 2) if tr else None, "decisions": len(t),
                                    "searched": tr.searched if tr else None, "single_option": tr.forced if tr else None,
                                    "battle_search_failures": tr.search_fail if tr else None, "battle_safe_searches": tr.safe_used if tr else None,
                                    "battle_tracker_errors": tr.errors if tr else None, "rated": tr.rated if tr else None,
                                    "ladder_before": tr.rating[0] if tr and tr.rating else None,
                                    "ladder_after": tr.rating[1] if tr and tr.rating else None}) + "\n")
        if self.n_finished_battles % 10 == 0:
            print(f"  progress: won {self.n_won_battles} / {self.n_finished_battles}", flush=True)


SERVERS = {"local": LocalhostServerConfiguration,
           "showdown": ShowdownServerConfiguration,     # play.pokemonshowdown.com, the main public server
           "pokeagent": ServerConfiguration("wss://battling.pokeagentchallenge.com/showdown/websocket",
                                            "https://battling.pokeagentchallenge.com/action.php?")}


def parse_opts(pairs):
    """["k=8", "name=x"] -> {"k": 8, "name": "x"}: values are JSON where they parse (numbers, true / false, lists)."""
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k.replace("-", "_")] = json.loads(v)
        except json.JSONDecodeError:
            out[k.replace("-", "_")] = v
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/spec2_00200.pt")
    ap.add_argument("--name", default="metapodv0-local")
    ap.add_argument("--battles", type=int, default=10)
    ap.add_argument("--accept", help="accept challenges from this user")
    ap.add_argument("--challenge", help="challenge this user")
    ap.add_argument("--ladder", action="store_true", help="queue on the (local) ladder")
    ap.add_argument("--teams", default="pool", help="'pool' (our ladder sets, independent draws), 'smogon' (pool sets built into "
                    "teams by agent/teams.py) or a directory of Showdown team files")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--concurrent", type=int, default=1, help="battles played at once (on the ladder: keeps searching while it plays)")
    ap.add_argument("--timer", action="store_true", help="turn the battle timer on at the start of every battle")
    ap.add_argument("--log-level", type=int, help="poke-env log level (20 = every message sent and received; login "
                    "messages are withheld from the log)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="torch device (default: the best available; 'cpu' beside a bot that has the GPU)")
    ap.add_argument("--search-tau", type=float, help="play with agent/search_exact.py at this tau")
    ap.add_argument("--search-k", type=int, default=8)
    ap.add_argument("--critic", help="search leaf evaluator (default: the checkpoint's own _critic.pt)")
    ap.add_argument("--teammates", action="store_true", help="search guesses unseen opponents with teammate statistics")
    ap.add_argument("--infer", action="store_true", help="search weights its guesses by the opponent's past choices")
    ap.add_argument("--rollouts", type=int, default=0, help="search scores each leaf by this many policy playouts")
    ap.add_argument("--rollout-depth", type=int, default=2, help="turns per leaf playout")
    ap.add_argument("--opp-model", help="checkpoint search models the opponent with (default: --ckpt), e.g. a human-imitation net")
    ap.add_argument("--opp-mix", type=float, default=1.0, help="with --opp-model: its share of the opponent's move prior, the rest our own policy's (DiL-piKL)")
    ap.add_argument("--search-depth", type=int, default=1, help="turns searched (later turns narrower)")
    ap.add_argument("--search-width2", type=int, default=3, help="candidate actions per side after the first turn")
    # agent/search_exact.py's newer options, passed on only when given (its defaults are the live behaviour)
    ap.add_argument("--search-calib", help="leaf-value calibration (JSON file with a, b)")
    ap.add_argument("--search-rounds", type=int, help="rounds of --search-k guessed worlds the root game averages")
    ap.add_argument("--search-budget", type=float, help="seconds per decision: rounds are added while they fit (cut to what the "
                    "battle timer leaves, when the timer is on)")
    ap.add_argument("--search-max-rounds", type=int)
    ap.add_argument("--search-width", type=int, help="our candidate actions")
    ap.add_argument("--search-max-out", type=int, help="chance outcomes kept per action pair")
    ap.add_argument("--search-opp-width", type=int, help="the opponent's candidate replies")
    ap.add_argument("--search-opp-eps", type=float, help="uniform floor on the opponent's reply prior")
    ap.add_argument("--search-opp-tau", type=float, help="the opponent's regularization weight in the root solve")
    ap.add_argument("--search-bayes", action="store_true", default=None, help="solve the root as a Bayesian game over opponent types")
    ap.add_argument("--search-opt", action="append", metavar="KEY=VALUE", help="any other ExactSearchPlayer argument (repeatable; the "
                    "value is JSON where it parses), e.g. --search-opt buckets=5 --search-opt hidden=false")
    ap.add_argument("--engine-rules", help="the engine ruleset of every VecEnv the search makes (default: smogon). 'smogon-sd' follows "
                    "Showdown where battle6 didn't (re-sleeping a Pokemon that woke this turn; a trap ended by a switch)")
    ap.add_argument("--dump-states", help="directory to save every decision's rebuilt team arrays and battle bytes, per battle")
    ap.add_argument("--dump-errors", help="directory to save the rebuilt state of every decision whose search failed")
    ap.add_argument("--server", choices=list(SERVERS), default="local",
                    help="'showdown': the main Pokemon Showdown server; 'pokeagent': the PokeAgent Challenge's AI ladder (both public; need an account: --creds or SHOWDOWN_USERNAME / SHOWDOWN_PASSWORD)")
    ap.add_argument("--creds", help="file with the account's username and password on two lines (overrides --name); "
                    "without it, the environment variables SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD are used when set")
    ap.add_argument("--results", help="append one JSON row per finished battle to this file")
    ap.add_argument("--save-replays", help="directory for each battle's full Showdown log as an HTML replay")
    ap.add_argument("--trace", help="directory for each battle's decision trace (options, search mix and values, choice) as JSONL")
    args = ap.parse_args()
    if args.engine_rules:
        import gen1py
        gen1py.VecEnv(1, rules=args.engine_rules)        # an older build doesn't know the ruleset: fail here, loudly
        os.environ["GEN1PY_RULES"] = args.engine_rules   # read by every VecEnv made from now on (engine/py/src/lib.rs)
    name, password = args.name, None
    if args.creds:
        name, password = Path(args.creds).expanduser().read_text().splitlines()[:2]
    elif os.environ.get("SHOWDOWN_USERNAME"):          # the public servers: account from the environment
        name, password = os.environ["SHOWDOWN_USERNAME"], os.environ.get("SHOWDOWN_PASSWORD")
    teams = PoolTeams(args.seed, smogon=args.teams == "smogon") if args.teams in ("pool", "smogon") else DirTeams(args.teams, args.seed)
    logging.getLogger(name).addFilter(_WithholdLogin())   # before the player exists: it logs in as soon as it is made
    opts = {k: v for k, v in (("calib", args.search_calib), ("rounds", args.search_rounds), ("budget", args.search_budget),
                              ("max_rounds", args.search_max_rounds), ("width", args.search_width), ("max_out", args.search_max_out),
                              ("opp_width", args.search_opp_width), ("opp_eps", args.search_opp_eps), ("opp_tau", args.search_opp_tau),
                              ("bayes", args.search_bayes)) if v is not None}
    opts.update(parse_opts(args.search_opt))
    p = OurPlayer(args.ckpt, args.device or default_device(), sample=args.sample, search_tau=args.search_tau, search_k=args.search_k, teammates=args.teammates, critic=args.critic, infer=args.infer,
                  rollouts=args.rollouts, rollout_depth=args.rollout_depth, opp_model=args.opp_model, opp_mix=args.opp_mix,
                  depth=args.search_depth, width2=args.search_width2, search_opts=opts, adapter_seed=args.seed,
                  account_configuration=AccountConfiguration(name, password),
                  server_configuration=SERVERS[args.server], battle_format="gen1ou", team=teams,
                  max_concurrent_battles=args.concurrent, start_timer_on_battle_start=args.timer, log_level=args.log_level,
                  save_replays=args.save_replays or False)
    p.results = args.results
    for attr, d in (("trace_dir", args.trace), ("state_dump", args.dump_states), ("dump_dir", args.dump_errors)):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            setattr(p, attr, d)
    if args.accept:
        await p.accept_challenges(args.accept, args.battles)
    elif args.challenge:
        await p.send_challenges(args.challenge, args.battles)
    else:
        await p.ladder(args.battles)
    print(f"{name}: won {p.n_won_battles} / {p.n_finished_battles} (ties {p.n_tied_battles}), "
          f"adapter errors {p.errors}, search fallbacks {p.search_errors} (rescued by the simplified state: {p.safe_searches}), "
          f"tracker lines skipped {p.tracker_errors}, last-resort choices {p.last_resorts}, opponent choices seen {getattr(p, 'opp_choices', 0)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
