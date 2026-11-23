"""One side of one evaluation pair for the paired harness (agent/next_eval_box.py, scripts/eval_metamon.sh).

A thin wrapper around agent.showdown_player: every flag it doesn't know is passed straight through to
`agent.showdown_player.main()`, so a variant is whatever flags (and environment) that module understands. What it adds:

- a team schedule shared across arms: battle j of (opponent, replica) plays the j-th team of a seeded, balanced
  schedule (every team equally often, in shuffled blocks), the same in every arm, so arms meet an opponent with the same
  teams in the same order. `--eval-skip` resumes a schedule after a restart;
- one JSON row per battle, written the moment it ends (`--eval-out`): arm, opponent, replica, index, result, turns,
  the team we actually played (file name), what the opponent showed (species and moves), decision count and times,
  adapter errors and search fallbacks so far;
- a watchdog: if no battle message arrives for `--eval-stall` seconds the row {"stalled": true} is written and the
  process exits with status 3, so the box runner can restart the pair instead of hanging;
- NEXT_DEVICE (cpu / cuda / mps) overrides the device, NEXT_THREADS (default 1) caps torch's CPU threads: a box
  shows every host core to every process, and dozens of processes each starting that many threads thrash.

    uv run --group showdown python -m agent.next_eval_play \
        --eval-out agent/runs/next_x/games.jsonl --eval-arm live --eval-opp bc --eval-rep 0 --eval-sched-seed 7 \
        --ckpt checkpoints/spec2_00200.pt --teams data/search_teams_v1 --search-tau 0.02 --infer \
        --name evA0 --challenge evB0 --battles 20
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NEXT_THREADS", "1")))

import agent.showdown_player as sp  # noqa: E402

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--eval-out", required=True)
ap.add_argument("--eval-arm", default="")
ap.add_argument("--eval-opp", default="")
ap.add_argument("--eval-rep", type=int, default=0)
ap.add_argument("--eval-role", default="arm", choices=["arm", "opp"])
ap.add_argument("--eval-sched-seed", type=int, default=0, help="team schedule seed: the same for every arm of an (opponent, replica)")
ap.add_argument("--eval-skip", type=int, default=0, help="schedule entries already played (a restarted pair)")
ap.add_argument("--eval-stall", type=float, default=900.0, help="seconds without any battle message before giving up")
ap.add_argument("--eval-logs", help="append every finished battle's Showdown log lines to this .jsonl.gz file")
EV, REST = ap.parse_known_args()
sys.argv = [sys.argv[0]] + REST

_dev = sp.default_device
sp.default_device = lambda: os.environ.get("NEXT_DEVICE") or _dev()
PROGRESS = {"t": time.time(), "tag": None, "done": 0, "last_end": time.time()}


def team_key(mons):
    """A team's identity: its species and their moves (ids), order-free."""
    return tuple(sorted((sp.to_id(sp_), tuple(sorted(sp.to_id(m) for m in moves))) for sp_, moves in mons))


class ScheduledTeams(sp.DirTeams):
    """A directory's teams in a seeded balanced order: shuffled blocks holding every team once."""
    files = {}                                     # team key -> file name, over every team directory seen

    def __init__(self, path, seed=0):
        paths = sorted(Path(path).rglob("*_team"))
        self.teams = [self.join_team(self.parse_showdown_team(p.read_text())) for p in paths]
        self.names = [f"{Path(path).name}/{p.name}" for p in paths]
        for p, n in zip(paths, self.names):
            mons = [(m.species or m.nickname, m.moves) for m in self.parse_showdown_team(p.read_text())]
            ScheduledTeams.files[team_key(mons)] = n
        self.rng = np.random.default_rng([EV.eval_sched_seed, 0x7EA])
        self.j, self.block, self.given = 0, [], []
        for _ in range(EV.eval_skip):
            self._next()

    def _next(self):
        if not self.block:
            self.block = list(self.rng.permutation(len(self.teams)))
        self.j += 1
        return int(self.block.pop())

    def yield_team(self):
        i = self._next()
        self.given.append(self.names[i])
        return self.teams[i]


sp.DirTeams = ScheduledTeams
_finished = sp.OurPlayer._battle_finished_callback
_handle = sp.OurPlayer._handle_battle_message


LINES = {}                                         # battle tag -> its Showdown log lines (for the luck counts)


def luck_counts(lines, me):
    """Chance trials of the battle, by side and kind (tools/luck.py counts)."""
    from tools.luck import counts
    return counts(lines, me)


async def handle(self, split_messages):
    PROGRESS["t"], PROGRESS["tag"] = time.time(), split_messages[0][0].lstrip(">")
    if EV.eval_role == "arm":
        LINES.setdefault(PROGRESS["tag"], []).extend("|".join(m) for m in split_messages[1:] if len(m) > 1)
    return await _handle(self, split_messages)


def finished(self, battle):
    t = list(self.times.get(battle.battle_tag, []))
    try:
        ours = [(m.species, list(m.moves)) for m in battle.team.values()]
        row = {"arm": EV.eval_arm, "opp": EV.eval_opp, "rep": EV.eval_rep, "role": EV.eval_role,
               "idx": EV.eval_skip + PROGRESS["done"], "tag": battle.battle_tag, "me": self.username,
               "them": battle.opponent_username, "won": battle.won, "turns": battle.turn,
               "team": ScheduledTeams.files.get(team_key(ours)),
               "opp_seen": sorted(f"{m.species}:{','.join(sorted(m.moves))}" for m in battle.opponent_team.values()),
               "n_dec": len(t), "mean_s": round(float(np.mean(t)), 3) if t else None, "max_s": round(max(t), 2) if t else None,
               "errors": getattr(self, "errors", 0), "fallbacks": self.search_errors,
               "secs": round(time.time() - PROGRESS["last_end"], 1), "t": round(time.time(), 1)}
        lines = LINES.pop(battle.battle_tag, [])
        if EV.eval_logs:
            import gzip
            with gzip.open(EV.eval_logs, "at") as f:
                f.write(json.dumps({"tag": battle.battle_tag, "me": self.username, "lines": lines}) + "\n")
        try:
            row["luck"] = luck_counts(lines, self.username)
        except Exception as e:
            row["luck_error"] = repr(e)[:200]
        with open(EV.eval_out, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:                           # the log must never break a battle
        print(f"eval row failed: {e!r}", flush=True)
    PROGRESS["done"] += 1
    PROGRESS["last_end"] = PROGRESS["t"] = time.time()
    return _finished(self, battle)


_accept = sp.OurPlayer.accept_challenges


async def accept(self, *a, **k):
    await self.ps_client.logged_in.wait()
    print("eval ready", flush=True)                  # the box runner starts the challenger once it sees this
    return await _accept(self, *a, **k)


sp.OurPlayer.accept_challenges = accept
sp.OurPlayer._handle_battle_message = handle
sp.OurPlayer._battle_finished_callback = finished


async def watchdog():
    while True:
        await asyncio.sleep(10)
        if time.time() - PROGRESS["t"] > EV.eval_stall:
            with open(EV.eval_out, "a") as f:
                f.write(json.dumps({"arm": EV.eval_arm, "opp": EV.eval_opp, "rep": EV.eval_rep, "role": EV.eval_role,
                                    "idx": EV.eval_skip + PROGRESS["done"], "tag": PROGRESS["tag"], "stalled": True,
                                    "t": round(time.time(), 1)}) + "\n")
            print(f"stalled: no battle message for {EV.eval_stall:.0f}s (battle {PROGRESS['tag']}); exiting", flush=True)
            os._exit(3)


async def main():
    Path(EV.eval_out).parent.mkdir(parents=True, exist_ok=True)
    w = asyncio.create_task(watchdog())
    print("eval player starting", flush=True)
    await sp.main()
    w.cancel()


if __name__ == "__main__":
    asyncio.run(main())
