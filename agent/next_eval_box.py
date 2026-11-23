"""Runs a paired A/B evaluation on one machine: every arm against every opponent at the same time, on the same local
Showdown server (127.0.0.1:8000, the one Metamon expects), with the same team schedules. scripts/eval_metamon.sh wraps it.

The plan comes from the environment (scripts/eval_metamon.sh sets an example):
  ARMS="live noinfer"                     arm names (letters, digits, underscore)
  ARM_live="--ckpt checkpoints/spec2_00200.pt --teams data/search_teams_v1 --search-tau 0.02 --infer"
                                          the arm's agent.showdown_player flags (any flag that module accepts)
  ENV_live="NEXT_X=1 NEXT_Y=a"            optional environment for that arm's processes
                                          ({unit} in flags or environment values becomes opponent_replica_arm, {out} the
                                          absolute OUT directory: a file per pair)
  DEV_live=cpu                            optional device for that arm (default ARM_DEVICE, default cpu)
  OPPS="TaurosV0:6:50 Kakuna:6:50"        opponent:replicas:battles - each arm plays replicas x battles against it
  SLOTS=4  GPU_SLOTS=2                    pairs running at once in total / of those, against Metamon bots
  TIME_LIMIT_MIN=90  PAIR_LIMIT_MIN=60    stop everything after / give one pair attempt at most
  OUT=agent/runs/eval_x
  METAMON_DIR=../metamon                  a clone of github.com/UT-Austin-RPL/metamon, installed per its README
  METAMON_PYTHON=$METAMON_DIR/.venv/bin/python   the interpreter that has metamon (and amago) installed
  METAMON_CACHE_DIR                       Metamon's own cache (checkpoints, team files); its default if unset

Opponents: a name in OWN (see OPP_* below) is one of ours (played by agent.next_eval_play on the CPU); any other name is
a Metamon agent (TaurosV0, Kakuna, TaurosEnsemble, KakunaEnsemble, ...) run by `python -m metamon.rl.evaluate --eval_type
challenge` in METAMON_DIR ("TaurosV0@cpu" runs it without the GPU; "Kakuna@fix2" gives each unit's bot only 2 of its
competitive teams, the same 2 in every arm: a regular opponent). Our own stand-ins are defined with environment
variables, e.g. OPP_nosearch="--ckpt checkpoints/spec2_00200.pt --teams pool" (the network without search, pool teams).

Pairing: a unit is (opponent, replica). All arms of a unit are queued next to each other, so they run at the same time
under the same load, and use the same team schedule seed (ours, and the opponent's when it is one of ours). Metamon
bots draw their own team at random each battle.

Every finished battle is one row in OUT/rows/*.jsonl the moment it ends; OUT/games.jsonl (all rows so far) and
OUT/status.json are rewritten every 30 s. A pair whose arm stalls (no battle message for STALL_S, default 600 s) or
exceeds PAIR_LIMIT_MIN is stopped and restarted for its remaining battles, at most twice. Score the result with
agent/next_eval_analyze.py.
"""
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = os.environ
OUT = Path(E.get("OUT", "agent/runs/next_eval_local"))
OWN = {}  # our own stand-in opponents: flags for agent.next_eval_play (role opp)
for k, v in E.items():   # e.g. OPP_nosearch="--ckpt checkpoints/spec2_00200.pt --teams pool"
    if k.startswith("OPP_"):
        OWN[k[4:]] = v
ARMS = E["ARMS"].split()
OPPS = [(o, int(r), int(n)) for o, r, n in (x.split(":") for x in E["OPPS"].split())]
SLOTS, GPU_SLOTS = int(E.get("SLOTS", "4")), int(E.get("GPU_SLOTS", "8"))
T_END = time.time() + 60 * float(E.get("TIME_LIMIT_MIN", "60"))
PAIR_LIMIT = 60 * float(E.get("PAIR_LIMIT_MIN", "60"))
STALL = E.get("STALL_S", "600")
PY = sys.executable
PROCS, LOCK, STATE = set(), threading.Lock(), {}
# SAVE_LOGS=1: every battle's Showdown log lines go to OUT/logs/battles_<unit>.jsonl.gz (they return in logs.tar.gz)
LOGS = lambda key: ["--eval-logs", str((OUT / "logs" / f"battles_{key}.jsonl.gz").resolve())] if E.get("SAVE_LOGS") else []


def rows_of(path):
    if not Path(path).exists():
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def spawn(cmd, log, env=None, cwd=ROOT):
    p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, env={**E, **(env or {})}, cwd=cwd,
                         start_new_session=True)
    with LOCK:
        PROCS.add(p)
    return p


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
    with LOCK:
        PROCS.discard(p)


def wait_for(log, text, p, limit):
    t0 = time.time()
    while time.time() - t0 < limit and time.time() < T_END:
        if p.poll() is not None:
            return False
        if Path(log).exists() and text in open(log, errors="replace").read():
            return True
        time.sleep(3)
    return False


def run_unit(arm, opp, rep, n, uid):
    """One arm's battles against one opponent replica, restarted after a stall until done (3 attempts)."""
    key = f"{opp}_{rep}_{arm}"
    rows, orows = OUT / "rows" / f"arm_{key}.jsonl", OUT / "rows" / f"opp_{key}.jsonl"
    seed = zlib.crc32(f"{opp}/{rep}".encode()) % 100000
    STATE[key] = {"done": 0, "n": n, "state": "starting", "attempts": 0}
    for attempt in range(3):
        done = sum(1 for r in rows_of(rows) if "won" in r)
        STATE[key].update(done=done, attempts=attempt + 1)
        if done >= n or time.time() > T_END - 60:
            break
        left = n - done
        a, b = f"e{uid}a{attempt}", f"e{uid}b{attempt}"
        blog, alog = OUT / "logs" / f"opp_{key}_{attempt}.log", OUT / "logs" / f"arm_{key}_{attempt}.log"
        thr = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        sub = lambda x: x.replace("{unit}", key).replace("{out}", str(OUT.resolve()))   # per-pair files, e.g. a memory file
        if opp in OWN:
            bot = spawn([PY, "-m", "agent.next_eval_play", "--eval-out", str(orows), "--eval-arm", arm, "--eval-opp", opp,
                         "--eval-rep", str(rep), "--eval-role", "opp", "--eval-sched-seed", str(seed + 1), "--eval-skip", str(done),
                         "--eval-stall", "100000", *shlex.split(OWN[opp]), "--name", b, "--accept", a, "--battles", str(left),
                         "--seed", str(seed)], blog, {**thr, "NEXT_DEVICE": "cpu"})
            ready = wait_for(blog, "eval ready", bot, 600)
        else:
            mm = Path(E.get("METAMON_DIR", ROOT.parent / "metamon")).resolve()
            cache = Path(E["METAMON_CACHE_DIR"]).resolve() if E.get("METAMON_CACHE_DIR") else None
            agent, _, where = opp.partition("@")          # "TaurosV0@cpu": the Metamon bot without the GPU
            team_set = "competitive"
            if where.startswith("fix"):                   # "Kakuna@fix2": a regular opponent - this unit's bot plays only
                import random                             # 2 of its competitive teams, the same 2 for every arm
                if cache is None:
                    raise RuntimeError("@fix opponents need METAMON_CACHE_DIR (where Metamon keeps its team files)")
                src = cache / "teams/competitive/gen1ou"
                team_set = f"fx_{agent}_{rep}_{where}"
                dst = cache / "teams" / team_set / "gen1ou"
                with LOCK:
                    if not dst.exists():
                        dst.mkdir(parents=True)
                        pick = random.Random(seed).sample(sorted(f for f in src.iterdir() if f.name.endswith("_team")), int(where[3:]))
                        for f in pick:
                            (dst / f.name).write_bytes(f.read_bytes())
                        (dst / "index.csv").write_text("filename\n" + "".join(f.name + "\n" for f in pick))
            bot = spawn([E.get("METAMON_PYTHON", str(mm / ".venv/bin/python")), "-m", "metamon.rl.evaluate", "--eval_type", "challenge",
                         "--agent", agent, "--username", b, "--opponent_username", a, "--role", "acceptor", "--gens", "1",
                         "--formats", "ou", "--total_battles", str(left), "--team_set", team_set,
                         "--save_results_to", str((OUT / "bots" / f"{key}_{attempt}").resolve()), "--battle_backend", "metamon"], blog,
                        {**thr, **({"METAMON_CACHE_DIR": str(cache)} if cache else {}), "PYTHONPATH": str(mm), "TORCHDYNAMO_DISABLE": "1",
                         "OMP_NUM_THREADS": E.get("BOT_THREADS", "2"), **({"CUDA_VISIBLE_DEVICES": ""} if where == "cpu" else {})}, cwd=mm)
            ready = wait_for(blog, "Made Challenge Env", bot, 1500)
            time.sleep(20 if ready else 0)
        if not ready:
            STATE[key]["state"] = "opponent never ready"
            stop(bot)
            continue
        env = {**thr, "NEXT_DEVICE": E.get(f"DEV_{arm}", E.get("ARM_DEVICE", "cpu"))}
        env.update(sub(kv).split("=", 1) for kv in E.get(f"ENV_{arm}", "").split())
        me = spawn([PY, "-m", "agent.next_eval_play", "--eval-out", str(rows.resolve()), "--eval-arm", arm, "--eval-opp", opp,
                    "--eval-rep", str(rep), "--eval-sched-seed", str(seed), "--eval-skip", str(done), "--eval-stall", STALL,
                    *LOGS(key), *shlex.split(sub(E[f"ARM_{arm}"])), "--name", a, "--challenge", b, "--battles", str(left), "--seed", str(seed)], alog, env)
        STATE[key]["state"] = "playing"
        t0 = time.time()
        while me.poll() is None and time.time() - t0 < PAIR_LIMIT and time.time() < T_END:
            time.sleep(5)
            STATE[key]["done"] = sum(1 for r in rows_of(rows) if "won" in r)
        STATE[key]["state"] = "finished" if me.poll() == 0 else f"stopped (exit {me.poll()})"
        stop(me)
        stop(bot)
    STATE[key]["done"] = sum(1 for r in rows_of(rows) if "won" in r)


def worker(q):
    while True:
        try:
            u = q.get_nowait()
        except queue.Empty:
            return
        try:
            run_unit(*u)
        except Exception as e:
            STATE[f"{u[1]}_{u[2]}_{u[0]}"] = {"state": f"runner error {e!r}"}


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def gpu_procs():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.split()
        return sorted(int(x) for x in o)
    except Exception:
        return None


def snapshot(t0):
    allrows = [r for f in sorted((OUT / "rows").glob("*.jsonl")) for r in rows_of(f)]
    tmp = OUT / "games.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in allrows))
    tmp.replace(OUT / "games.jsonl")
    n = sum(1 for r in allrows if r.get("role") == "arm" and "won" in r)
    status = {"minutes": round((time.time() - t0) / 60, 1), "arm_games": n, "games_per_min": round(n / max(1e-9, (time.time() - t0) / 60), 1),
              "stalls": sum(1 for r in allrows if r.get("stalled")), "gpu": gpu(), "gpu_procs": gpu_procs(), "load": os.getloadavg()[0], "units": STATE}
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    return n


def main():
    global GPU_SLOTS
    for d in ("rows", "logs", "bots"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    qs = {True: queue.Queue(), False: queue.Queue()}
    uid = base = int(E.get("UID_BASE", "0"))        # a second runner on the same server needs different user names
    for rep in range(max(r for _, r, _ in OPPS)):
        for opp, reps, n in OPPS:
            if rep < reps:
                for arm in ARMS:
                    uid += 1
                    qs[opp in OWN].put((arm, opp, rep, n, uid))
    g = gpu()
    if g and "BOT_MB" in E:                                   # no more GPU bots than the card's free memory holds
        used, total = (int(x.split()[0]) for x in g.split(",")[1:3])
        GPU_SLOTS = min(GPU_SLOTS, (total - used) // int(E["BOT_MB"]))
    n_gpu = min(GPU_SLOTS, qs[False].qsize())
    n_own = min(max(0, SLOTS - n_gpu), qs[True].qsize())
    print(f"{uid - base} units; {n_gpu} pairs at once against Metamon bots, {n_own} against our own stand-ins", flush=True)
    threads = [threading.Thread(target=worker, args=(qs[False],), daemon=True) for _ in range(n_gpu)]
    threads += [threading.Thread(target=worker, args=(qs[True],), daemon=True) for _ in range(n_own)]
    t0 = time.time()
    for i, t in enumerate(threads):
        t.start()
        time.sleep(float(E.get("STAGGER_S", "2")))            # don't start every process in the same second
    while any(t.is_alive() for t in threads) and time.time() < T_END:
        time.sleep(30)
        n = snapshot(t0)
        print(f"{(time.time() - t0) / 60:.1f} min: {n} arm games", flush=True)
    with LOCK:
        left = list(PROCS)
    for p in left:
        stop(p)
    print(f"done: {snapshot(t0)} arm games in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
