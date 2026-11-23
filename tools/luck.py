"""Luck counts from a Showdown battle log: every chance trial with a known probability (crits, misses, full paralysis,
freezes, secondary paralysis), per side, with the observed and expected number of events. agent/next_eval_play.py stores
them in each result row and agent/next_eval_analyze.py regresses results on them (the luck adjustment).

    from tools.luck import counts
    counts(log_lines, "our_username")        # {"us_crit": [trials, observed, expected], ...}
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOT = "metapodv0"
norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())


# ---------- replay parsing: chance trials with known probabilities

_D = json.loads((ROOT / "data/gen1.json").read_text())
SPE = {norm(s["name"]): s["base"]["spe"] for s in _D["species"]}
TYPES = {norm(s["name"]): set(s["types"]) for s in _D["species"]}
MOV = {norm(m["name"]): m for m in _D["moves"] if m}
for _a, _b in [("psychic", "psychicm"), ("hijumpkick", "highjumpkick"), ("vicegrip", "visegrip")]:
    if _b in MOV and _a not in MOV:
        MOV[_a] = MOV[_b]
TN = {t: i for i, t in enumerate(_D["types"])}
HIGHCRIT = {"slash", "razorleaf", "crabhammer", "karatechop"}
FIXED = {"seismictoss", "nightshade", "superfang", "dragonrage", "sonicboom", "psywave", "counter", "bide", "fissure", "horndrill", "guillotine"}
NEVERMISS = {"swift"}
SEC = {"blizzard": ("frz", 0.10, "ICE"), "icebeam": ("frz", 0.10, "ICE"), "bodyslam": ("par", 0.30, "NORMAL"),
       "thunderbolt": ("par", 0.10, "ELECTRIC"), "thunder": ("par", 0.10, "ELECTRIC")}
KINDS = ("crit", "miss", "fpar", "frz", "par")
# Does an observed event of this kind, on this side (the attacker; for fpar the paralysed mon's side), help us?
HELPS = {("crit", "us"): True, ("crit", "them"): False, ("miss", "us"): False, ("miss", "them"): True,
         ("fpar", "us"): False, ("fpar", "them"): True, ("frz", "us"): True, ("frz", "them"): False,
         ("par", "us"): True, ("par", "them"): False}
LABEL = {("crit", "us"): "Our crits", ("crit", "them"): "Their crits", ("miss", "us"): "Our misses", ("miss", "them"): "Their misses",
         ("fpar", "us"): "We're fully paralysed", ("fpar", "them"): "They're fully paralysed", ("frz", "us"): "We freeze them",
         ("frz", "them"): "They freeze us", ("par", "us"): "We paralyse them", ("par", "them"): "They paralyse us"}

_PARSED = {}


def replay_lines(path):
    txt = Path(path).read_text(errors="ignore")
    m = re.search(r'class="battle-log-data">(.*?)</script>', txt, re.S)
    return [l for l in (m.group(1) if m else txt).split("\n") if l.startswith("|")]


def parse(path):
    """Chance trials per side, chance events by turn, ratings and how it ended; cached per file (replays are immutable)."""
    if path in _PARSED:
        return _PARSED[path]
    lines = replay_lines(path)
    me, ratings, names = None, {}, {}
    for l in lines:
        p = l.split("|")
        if p[1] == "player" and len(p) > 3 and p[3]:
            names[p[2]] = p[3]
            if len(p) > 5 and p[5].isdigit():
                ratings[p[2]] = int(p[5])
            if norm(p[3]) == BOT:
                me = p[2]
    opp = "p2" if me == "p1" else "p1"
    side = lambda ident: "us" if ident[:2] == me else "them"
    nick = lambda ident: ident.split(": ", 1)[1] if ": " in ident else ident
    st, nick2sp, C, events = {}, {}, defaultdict(float), []
    turn, won, ff = 0, None, None

    def trial(s, kind, prob, hit, t, text):
        C[(s, kind, "n")] += 1
        C[(s, kind, "obs")] += hit
        C[(s, kind, "exp")] += prob
        C[(s, kind, "var")] += prob * (1 - prob)
        if hit:
            events.append({"turn": t, "kind": kind, "side": s, "text": text, "helps": HELPS[(kind, s)]})

    def setst(ident, hp):
        parts = hp.split(" ")
        st[(side(ident), nick(ident))] = parts[1] if len(parts) > 1 else None

    sp_name = lambda s, n: nick2sp.get((s, n), n)
    who = lambda s: "our" if s == "us" else "their"
    for i, line in enumerate(lines):
        p = line.split("|")
        k = p[1]
        if k == "turn":
            turn = int(p[2])
        elif k in ("switch", "drag") and len(p) > 4:
            s, n = side(p[2]), nick(p[2])
            nick2sp[(s, n)] = p[3].split(",")[0]
            setst(p[2], p[4])
        elif k in ("-damage", "-heal") and len(p) > 3 and "[from]" not in (p[4] if len(p) > 4 else ""):
            if p[3] != "0 fnt":
                setst(p[2], p[3])
        elif k == "-status" and len(p) > 3:
            st[(side(p[2]), nick(p[2]))] = p[3]
        elif k == "-curestatus" and len(p) > 2:
            st[(side(p[2]), nick(p[2]))] = None
        elif k == "cant" and len(p) > 3 and p[3] == "par":
            s = side(p[2])
            trial(s, "fpar", 0.25, 1, turn, f"{who(s)} {sp_name(s, nick(p[2]))} was fully paralysed")
        elif k == "move" and len(p) > 3 and "[from]" not in line and "[still]" not in line:
            s, n = side(p[2]), nick(p[2])
            mv, tgt = norm(p[3]), (p[4] if len(p) > 4 else "")
            if st.get((s, n)) == "par":
                trial(s, "fpar", 0.25, 0, turn, "")
            blk = []
            for nxt in lines[i + 1:]:
                if nxt.split("|")[1] in ("move", "cant", "turn", "switch", "drag", "upkeep"):
                    break
                blk.append(nxt)
            M = MOV.get(mv)
            tside = side(tgt) if tgt else None
            hits_opp = bool(tgt) and tside != s
            missed = "[miss]" in line or any(b.startswith("|-miss|") for b in blk)
            failed = any(b.startswith(("|-fail|", "|-immune|")) for b in blk)
            mon, tmon = sp_name(s, n), sp_name(tside, nick(tgt)) if tgt else ""
            if M and hits_opp and not failed and mv not in NEVERMISS:
                acc = min(M["accuracy"], 255) / 256
                trial(s, "miss", 1 - acc, int(missed), turn, f"{who(s)} {mon}'s {p[3]} missed")
            dmg = any(b.startswith("|-damage|") and b.split("|")[2][:2] == tgt[:2] for b in blk) if tgt else False
            crit = any(b.startswith("|-crit|") for b in blk)
            if M and hits_opp and M["power"] > 1 and mv not in FIXED and (dmg or crit) and not missed:
                b = SPE.get(norm(mon), 80) // 2
                trial(s, "crit", min(255, b * 8 if mv in HIGHCRIT else b) / 256, int(crit), turn, f"{who(s)} {mon}'s {p[3]} crit {tmon}")
            if mv in SEC and hits_opp and dmg and not missed:
                stt, ch, imm = SEC[mv]
                tsp = norm(tmon)
                fainted = any(b.startswith("|faint|") or (b.startswith("|-damage|") and "0 fnt" in b) for b in blk)
                subhit = any("Substitute" in b for b in blk)
                if st.get((tside, nick(tgt))) is None and TN[imm] not in TYPES.get(tsp, set()) and not fainted and not subhit:
                    got = any(b.startswith("|-status|") and b.split("|")[3] == stt for b in blk)
                    trial(s, stt, ch, int(got), turn, f"{who(s)} {p[3]} {'froze' if stt == 'frz' else 'paralysed'} {tmon}")
        elif k == "win":
            won = norm(p[2]) == BOT
        elif k == "-message" and len(p) > 2 and " forfeited" in p[2]:
            ff = "us" if norm(p[2].split(" forfeited")[0]) == BOT else "them"
    out = {"r_us": ratings.get(me), "r_opp": ratings.get(opp), "won": won, "ff": ff, "C": dict(C), "events": events}
    _PARSED[path] = out
    return out


# ---------- small statistics helpers


def counts(lines, me):
    """Chance trials of one battle (its Showdown log lines), from player `me`'s side, by side and kind:
    {"us_crit": [trials, observed, expected], ...}. observed - expected has mean zero whatever either player does,
    so agent/next_eval_analyze.py can subtract its fitted effect from the result (fewer games for the same interval)."""
    global BOT, replay_lines
    BOT, key = norm(me), object()
    old, replay_lines = replay_lines, lambda _path: lines
    try:
        C = parse(key)["C"]
    finally:
        replay_lines = old
        _PARSED.pop(key, None)
    return {f"{s}_{k}": [int(C.get((s, k, "n"), 0)), int(C.get((s, k, "obs"), 0)), round(C.get((s, k, "exp"), 0.0), 3)]
            for s in ("us", "them") for k in KINDS if C.get((s, k, "n"))}
