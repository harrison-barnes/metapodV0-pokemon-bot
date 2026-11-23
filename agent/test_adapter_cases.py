"""One small case per bug the ground-truth test (agent/test_adapter_truth.py) found in the Showdown adapter, on
hand-written protocol lines, so each fix has a test that needs no simulator. Also the fail-soft guards.

    uv run --group showdown python -m agent.test_adapter_cases
"""
import logging
import sys

import gen1py  # noqa: F401
import numpy as np

from agent.showdown_player import (KEEP_DAMAGE, MIMIC, MOVE_IDX, OurPlayer, PASS, SIDE, SPECIES_IDX, STRUGGLE, TRAPPING, Tracker,
                                   hp_from_percent, max_pp)
from agent.test_adapter_truth import offline_player

P1, P2 = "p1: Tauros", "p2: Snorlax"


def track(tr, text):
    """The three symmetric trackers on protocol lines (no player needed)."""
    for line in text.strip().split("\n"):
        m = line.strip().split("|")
        if len(m) >= 3:
            OurPlayer._track_locks(tr, m[1], m)
            OurPlayer._track_stats(tr, m[1], m)
            OurPlayer._track_state(tr, m[1], m)
    return tr


def start():
    return track(Tracker(), """
|switch|p1a: Tauros|Tauros|353/353
|switch|p2a: Snorlax|Snorlax|100/100""")


def case_recharge_ends_with_sleep():
    """Hyper Beam, then put to sleep before recharging: the recharge is gone (it stayed set for the rest of the
    Pokemon's stay: no switching, and a lost turn on waking, in the search's eyes)."""
    tr = track(start(), """
|move|p2a: Snorlax|Hyper Beam|p1a: Tauros
|-damage|p1a: Tauros|200/353
|-mustrecharge|p2a: Snorlax""")
    assert P2 in tr.recharge
    track(tr, "|move|p1a: Tauros|Sing|p2a: Snorlax\n|-status|p2a: Snorlax|slp|[from] move: Sing")
    assert P2 not in tr.recharge


def case_recharge_ends_with_flinch_or_trap():
    tr = track(start(), "|-mustrecharge|p2a: Snorlax\n|cant|p2a: Snorlax|flinch")
    assert P2 not in tr.recharge
    tr = track(start(), "|-mustrecharge|p2a: Snorlax\n|move|p1a: Tauros|Wrap|p2a: Snorlax|[miss]\n|-miss|p1a: Tauros")
    assert P2 not in tr.recharge, "a trapping move cancels the recharge even when it misses"


def case_sleep_turns_are_counted_from_the_protocol():
    """One |cant|slp per sleep turn used: not one per decision of ours (a forced switch of ours is a decision without a
    sleep turn; a Pokemon put to sleep before it moved has used a turn by our next decision)."""
    tr = track(start(), """
|move|p1a: Tauros|Sing|p2a: Snorlax
|-status|p2a: Snorlax|slp|[from] move: Sing
|cant|p2a: Snorlax|slp""")
    assert tr.slp[P2] == 1
    track(tr, "|switch|p2a: Chansey|Chansey|100/100\n|switch|p2a: Snorlax|Snorlax|100/100 slp")
    assert tr.slp[P2] == 1, "benched turns don't count"
    track(tr, "|cant|p2a: Snorlax|slp\n|-curestatus|p2a: Snorlax|slp|[msg]")
    assert P2 not in tr.slp


def case_confused_turns():
    tr = track(start(), "|-start|p2a: Snorlax|confusion\n|-activate|p2a: Snorlax|confusion\n|-activate|p2a: Snorlax|confusion")
    assert tr.vol["p2"]["conf_n"] == 2
    track(tr, "|-end|p2a: Snorlax|confusion")
    assert "conf_n" not in tr.vol["p2"]


def case_trap_hits():
    tr = track(start(), "|move|p1a: Tauros|Wrap|p2a: Snorlax\n|-damage|p2a: Snorlax|95/100")
    assert tr.trap[P1]["h"] == 0 and tr.trap[P1]["dmg"] > 0
    track(tr, "|move|p1a: Tauros|Wrap|p2a: Snorlax|[from] Wrap\n|-damage|p2a: Snorlax|90/100")
    assert tr.trap[P1]["h"] == 1
    track(tr, "|move|p1a: Tauros|Wrap|p2a: Snorlax\n|-damage|p2a: Snorlax|85/100")
    assert tr.trap[P1]["h"] == 0, "chosen again after it ended: a new trap, not a third hit"
    track(tr, "|cant|p1a: Tauros|par")
    assert P1 not in tr.trap, "a fully paralysed trapper lets go"


def case_pp_and_shown_moves():
    tr = track(start(), """
|move|p2a: Snorlax|Skull Bash||[still]
|-prepare|p2a: Snorlax|Skull Bash""")
    sb = MOVE_IDX["skullbash"]
    assert tr.used[P2] == [sb] and tr.pp_used.get(P2, {}).get(sb, 0) == 0, "the charging turn costs no PP"
    track(tr, "|move|p2a: Snorlax|Skull Bash|p1a: Tauros|[from] Skull Bash\n|-damage|p1a: Tauros|200/353")
    assert tr.pp_used[P2][sb] == 1
    track(tr, "|move|p2a: Snorlax|Wrap|p1a: Tauros\n|-damage|p1a: Tauros|190/353\n|move|p2a: Snorlax|Wrap|p1a: Tauros|[from] Wrap")
    assert tr.pp_used[P2][MOVE_IDX["wrap"]] == 1, "a continuing turn costs no PP"
    track(tr, "|move|p2a: Snorlax|Metronome|p2a: Snorlax\n|move|p2a: Snorlax|Surf|p1a: Tauros|[from] Metronome")
    assert MOVE_IDX["surf"] not in tr.used[P2], "a move Metronome called is not one of its moves"
    track(tr, "|move|p2a: Snorlax|Struggle|p1a: Tauros")
    assert STRUGGLE not in tr.used[P2]
    assert max_pp(MOVE_IDX["growl"]) == 61, "Showdown's cap for 40-PP moves"


def case_toxic_counter():
    tr = track(start(), "|-status|p2a: Snorlax|tox\n|-damage|p2a: Snorlax|94/100 tox|[from] psn\n|-damage|p2a: Snorlax|82/100 tox|[from] psn")
    assert tr.vol["p2"]["toxic"] == 2
    track(tr, "|move|p2a: Snorlax|Rest|p2a: Snorlax\n|-status|p2a: Snorlax|slp|[from] move: Rest")
    assert tr.vol["p2"]["toxic"] == 2, "Rest keeps the counter (it then scales burn and poison)"
    track(tr, "|switch|p2a: Chansey|Chansey|100/100")
    assert "toxic" not in tr.vol["p2"]


def case_last_damage():
    tr = track(start(), "|move|p2a: Snorlax|Body Slam|p1a: Tauros\n|-damage|p1a: Tauros|253/353")
    assert tr.last_damage == 100 and tr.damage_shown
    track(tr, "|move|p1a: Tauros|Body Slam|p2a: Snorlax\n|-damage|p2a: Snorlax|80/100")
    mx = 523
    assert abs(tr.last_damage - 0.2 * mx) <= 0.011 * mx + 1, tr.last_damage      # from percentages: within a point
    track(tr, "|move|p2a: Snorlax|Reflect|p2a: Snorlax\n|-start|p2a: Snorlax|Reflect")
    assert tr.last_damage > 0 and MOVE_IDX["reflect"] in KEEP_DAMAGE, "Reflect leaves the last damage for Counter"
    before = tr.last_damage
    track(tr, "|move|p1a: Tauros|Substitute|p1a: Tauros\n|-start|p1a: Tauros|Substitute\n|-damage|p1a: Tauros|165/353")
    assert tr.last_damage == before, "a Substitute's cost is not move damage"
    track(tr, "|move|p2a: Snorlax|Body Slam|p1a: Tauros\n|-activate|p1a: Tauros|Substitute|[damage]")
    assert not tr.damage_shown
    track(tr, "|move|p2a: Snorlax|Body Slam|p1a: Tauros|[miss]\n|-miss|p2a: Snorlax")
    assert tr.last_damage == 0


def case_hp_from_percent():
    """A shown percentage is rounded up: 7% of 313 is 19-21 HP (the adapter used to say 22)."""
    for mx in (313, 353, 523, 703):
        for hp in range(1, mx + 1):
            pct = -(-hp * 100 // mx)
            pct = 99 if pct == 100 and hp < mx else pct
            est = hp_from_percent(pct, mx)
            assert -(-est * 100 // mx) in (pct, 100 if pct == 99 else pct), (mx, hp, pct, est)
            assert abs(est - hp) <= mx // 100 + 1
    assert hp_from_percent(7, 313) in (19, 20, 21) and hp_from_percent(100, 313) == 313 and hp_from_percent(0, 313) == 0


OPENING = """|init|battle
|title|Truth One vs. Truth Two
|gametype|singles
|player|p1|Truth One||
|player|p2|Truth Two||
|gen|1
|tier|[Gen 1] OU
|
|teamsize|p1|6
|teamsize|p2|5
|start
|switch|p1a: Tauros|Tauros|353/353
|switch|p2a: Snorlax|Snorlax|100/100
|turn|1"""
MOVES = ["bodyslam", "hyperbeam", "earthquake", "blizzard"]
SIDE_JSON = ('"side":{"name":"Truth One","id":"p1","pokemon":[{"ident":"p1: Tauros","details":"Tauros","condition":"353/353","active":true,'
             '"stats":{"atk":298,"def":288,"spa":238,"spd":238,"spe":318},"moves":["bodyslam","hyperbeam","earthquake","blizzard"],'
             '"baseAbility":"noability","item":"","pokeball":"pokeball"},{"ident":"p1: Chansey","details":"Chansey","condition":"703/703",'
             '"active":false,"stats":{"atk":108,"def":108,"spa":308,"spd":308,"spe":198},"moves":["softboiled","thunderwave","icebeam","thunderbolt"],'
             '"baseAbility":"noability","item":"","pokeball":"pokeball"}]}')


def request(maybe_locked=False, pp=(24, 8, 16, 8)):
    mv = ",".join('{"move":"%s","id":"%s","pp":%d,"maxpp":%d,"target":"normal","disabled":false}' % (m, m, q, q) for m, q in zip(MOVES, pp))
    return '|request|{"active":[{"moves":[%s]%s}],%s}' % (mv, ',"maybeLocked":true' if maybe_locked else "", SIDE_JSON)


def new_player():
    p = offline_player("Truth One")
    p.logger.setLevel(logging.CRITICAL)      # the cases below provoke errors on purpose
    p.choose_move = lambda battle: (setattr(p, "state", p.rebuild(battle)), p.choose_default_move())[1]
    return p


def case_opening_in_one_message():
    """The room's opening and the first switches in one message (it happens: 1 of 160 local battles): our own lead used
    to be listed as the opponent's first Pokemon for the whole battle, shifting its real team by one slot."""
    p = new_player()
    p.feed("battle-gen1ou-9001", OPENING)
    p.feed("battle-gen1ou-9001", request())
    tr = p.trackers["battle-gen1ou-9001"]
    assert tr.opp == [P2] and tr.own[0] == P1, (tr.own, tr.opp)
    team = p.state["team"]
    assert team[1, 0, 0] == SPECIES_IDX["snorlax"] and team[1, 0, 12] == 1
    assert team[1, 5, 15] == 0 and team[1, 5, 12] == 1, "a five-Pokemon team: the sixth slot is not an unknown Pokemon"
    assert list(team[0, 0, 8:12]) == [24, 8, 16, 8]


def case_trap_that_may_go_on():
    """After the first continuation turn Showdown offers every move with "maybeLocked": the trap may or may not go on.
    The flag stays, with the hits so far, for the search to draw (it used to be dropped: wrong 5 times in 8)."""
    p = new_player()
    tag = "battle-gen1ou-9002"
    p.feed(tag, OPENING)
    p.feed(tag, request())
    p.feed(tag, "|\n|move|p2a: Snorlax|Wrap|p1a: Tauros\n|-damage|p1a: Tauros|340/353\n|cant|p1a: Tauros|partiallytrapped\n|\n|upkeep\n|turn|2")
    p.feed(tag, '|request|{"active":[{"moves":[{"move":"Fight","id":"fight"}]}],%s}' % SIDE_JSON)
    st = p.state
    assert st["bb"][SIDE + 27] & TRAPPING and st["trap"] == {1: 0} and not st["trap_uncertain"] and st["mask"][PASS]
    p.feed(tag, "|\n|move|p2a: Snorlax|Wrap|p1a: Tauros|[from] Wrap\n|-damage|p1a: Tauros|327/353\n|\n|upkeep\n|turn|3")
    p.feed(tag, request(maybe_locked=True))
    st = p.state
    assert st["bb"][SIDE + 27] & TRAPPING and st["trap"] == {1: 1} and st["trap_uncertain"] and st["mask"][:4].all()
    assert int(st["bb"][100]) == 13, "the trap's damage is the last damage"
    p.feed(tag, "|\n|move|p2a: Snorlax|Body Slam|p1a: Tauros\n|-damage|p1a: Tauros|227/353\n|move|p1a: Tauros|Body Slam|p2a: Snorlax\n|-damage|p2a: Snorlax|80/100\n|\n|upkeep\n|turn|4")
    p.feed(tag, request(pp=(23, 8, 16, 8)))
    st = p.state
    assert not (st["bb"][SIDE + 27] & TRAPPING) and st["trap"] == {}
    assert list(st["team"][0, 0, 8:12]) == [23, 8, 16, 8], "our PP is the request's"


def case_fainted_before_acting_is_no_choice():
    """We knock the opponent out before it moves; its replacement is not its answer to our decision (it used to go
    into the search's inference history as a voluntary switch)."""
    p = new_player()
    tag = "battle-gen1ou-9003"
    p.feed(tag, OPENING)
    p.feed(tag, request())
    tr = p.trackers[tag]
    tr.pending = (np.zeros((2, 6, 18), np.int64), bytes(102))
    p.feed(tag, "|\n|move|p1a: Tauros|Hyper Beam|p2a: Snorlax\n|-damage|p2a: Snorlax|0 fnt\n|faint|p2a: Snorlax\n|\n|upkeep")
    p.feed(tag, "|\n|switch|p2a: Chansey|Chansey|100/100\n|turn|2")
    assert tr.past == [] and tr.pending is None


def case_timer_cuts_the_search_budget():
    p = new_player()
    tag = "battle-gen1ou-9007"
    p.feed(tag, OPENING)
    p.feed(tag, "|inactive|Time left: 150 sec this turn | 290 sec total")
    assert p.trackers[tag].clock == (150, 290)
    assert OurPlayer.clock_budget(30.0, (150, 290)) == 290 / 15 and OurPlayer.clock_budget(2.0, (150, 290)) == 2.0
    assert OurPlayer.clock_budget(30.0, (12, 20)) == 20 / 15 and OurPlayer.clock_budget(5.0, None) == 5.0
    assert OurPlayer.clock_budget(5.0, (8, 300)) == 0.05, "under ten seconds on the turn: answer at once"


def case_end_of_battle_goes_to_the_set_model():
    """searcher.end_battle gets the opponent's rows as the battle left them (it used to get an empty team: the
    tracker had already been put away), without the stand-in for the slot a five-Pokemon team doesn't have."""
    p = new_player()
    tag, got = "battle-gen1ou-9008", []
    p.searcher.end_battle = lambda **kw: got.append(kw)
    p.feed(tag, OPENING)
    p.feed(tag, request())
    p.feed(tag, "|\n|move|p2a: Snorlax|Body Slam|p1a: Tauros\n|-damage|p1a: Tauros|253/353\n|\n|upkeep\n|turn|2")
    p.feed(tag, "|\n|win|Truth One")
    assert len(got) == 1 and got[0]["opponent"] == "Truth Two" and got[0]["won"] is True, got
    opp = got[0]["opp_team"]
    assert opp[0, 0] == SPECIES_IDX["snorlax"] and opp[0, 12] == 1 and opp[0, 1] == MOVE_IDX["bodyslam"] and opp[0, 13] == 1
    assert not opp[5].any() and opp[4, 17] == 1 and opp[4, 12] == 0, "slot 6 of a five-Pokemon team is empty; slot 5 is unseen"
    assert tag not in p.trackers


def case_tracker_error_costs_one_line():
    """A line our tracking chokes on is skipped; poke-env still gets the message and the request is still answered."""
    p = new_player()
    tag = "battle-gen1ou-9004"
    orig = OurPlayer._track_state
    try:
        type(p)._track_state = staticmethod(lambda tr, kind, m: 1 / 0)
        p.feed(tag, OPENING)
        p.feed(tag, request())
    finally:
        type(p)._track_state = orig
    assert p.tracker_errors > 0 and p.battles[tag].turn == 1 and any("/choose" in m for _, m in p.sent), p.sent


def case_unreadable_request_is_answered():
    """If poke-env fails on a request, Showdown's default choice is sent instead of letting the timer run out."""
    p = new_player()
    tag = "battle-gen1ou-9005"
    p.feed(tag, OPENING)
    p.feed(tag, '|request|{"active":[{"moves":"nonsense"}],"side":{"id":"p1","pokemon":7}}')
    assert (tag, "/choose default") in p.sent and p.last_resorts == 1, p.sent


def case_search_failure_costs_one_search():
    """The search raising on the full state: the same decision is searched again on the simplified state; if that
    fails too, the policy answers; either way the next decision starts fresh. Needs the checkpoint (CPU)."""
    ckpt = "checkpoints/spec2_00200.pt"
    p = offline_player("Truth One", play=dict(ckpt=ckpt, search_tau=0.02, infer=True))
    tag = "battle-gen1ou-9006"
    real = p.searcher._search
    calls = []

    def flaky(team, mask, side, env, games):
        calls.append(len(p.searcher.past[0]) if p.searcher.past.get(0) is not None else None)
        if len(calls) in (1, 3, 4):          # decision 1: full fails, safe works; decision 2: both fail; decision 3: works
            raise ValueError("injected")
        return real(team, mask, side, env, games)
    p.searcher._search = flaky
    p.feed(tag, OPENING)
    p.feed(tag, request())
    assert p.search_errors == 1 and p.safe_searches == 1 and p.orders[-1].startswith("/choose"), (p.search_errors, p.safe_searches)
    p.feed(tag, "|\n|move|p1a: Tauros|Body Slam|p2a: Snorlax\n|-damage|p2a: Snorlax|80/100\n|move|p2a: Snorlax|Body Slam|p1a: Tauros\n|-damage|p1a: Tauros|253/353\n|\n|upkeep\n|turn|2")
    p.feed(tag, request(pp=(23, 8, 16, 8)))
    assert p.search_errors == 2 and p.safe_searches == 1 and len(p.orders) == 2 and p.orders[-1].startswith("/choose")
    p.feed(tag, "|\n|move|p1a: Tauros|Body Slam|p2a: Snorlax\n|-damage|p2a: Snorlax|60/100\n|move|p2a: Snorlax|Body Slam|p1a: Tauros\n|-damage|p1a: Tauros|153/353\n|\n|upkeep\n|turn|3")
    p.feed(tag, request(pp=(22, 8, 16, 8)))
    assert p.search_errors == 2 and len(p.orders) == 3 and p.errors == 0 and p.last_resorts == 0


if __name__ == "__main__":
    slow = "--fast" not in sys.argv
    cases = [(n, f) for n, f in sorted(globals().items()) if n.startswith("case_") and (slow or n != "case_search_failure_costs_one_search")]
    failed = 0
    for n, f in cases:
        try:
            f()
            print("ok  ", n)
        except Exception as e:
            failed += 1
            print("FAIL", n, repr(e))
    print(f"{len(cases) - failed} of {len(cases)} cases pass")
    sys.exit(1 if failed else 0)
