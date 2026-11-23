"""v4 training (agent/train_v4.py) with the games played by CPU processes while the GPU trains.

In train_v4 one process alternates: it plays a rollout (the GPU mostly waits on the engine and the
feature code), then trains on it (the CPU waits). Here --actors CPU processes play continuously,
each with --actor-envs battles and only the policy, and send each --steps-turn chunk to the
learner. The learner, on the GPU, takes --envs battles' worth of chunks per iteration and does
everything that needs the critic:
- the critic's value of every step (and of each chunk's last state, to bootstrap GAE);
- the counterfactual Q targets (--qhead): positions are rebuilt in a VecEnv (VecEnv.load) from the
  chunk's arrays and every pair of legal actions is played out there, as in train_v4;
- the PPO / magnet / tablebase update (train_v4.update, unchanged).
After each update it publishes the policy's weights (--weights file, version-numbered); actors pick
them up between chunks, so a chunk is played by a policy at most a version or two old. The PPO ratio
is taken against the policy that actually played (its log-probs travel with the chunk), and chunks
more than --max-lag versions old are dropped.

The matchup tables (670 MB) are loaded once and shared with the actors (torch shared memory), and
frozen league opponents on the same table reuse them.

Options are train_v4's (same names, same meaning) plus --actors, --actor-envs, --actor-threads and
--max-lag. --envs is the battles per iteration across all actors (a multiple of --actor-envs).

Search distillation (--distill-workers N with --distill-coef > 0): N more processes run the exact
search on a random share of positions of their own policy games (agent/search_targets.py) and the
update pulls the policy toward the search's mix. The critic is untouched by it. The searching
checkpoint pair (RUN/search.pt, search_critic.pt) is refreshed every --distill-refresh iterations.

    uv run python -m agent.actor_learner --init ... --actors 12 --actor-envs 64 [train_v4 options]
"""
import argparse
import copy
import json
import os
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from agent.selfplay import BOTH, ROOT, gae, evaluate, head_to_head
from agent.selfplay import observe_views
from agent.train_v4 import ALIVE, Counterfactual, TBStream, add_args, critic_obs, portable, reveal, update


def shared_tables(paths):
    """{table file name: (Matchups, Senses or None)} on the CPU, tensors in shared memory, for every
    matchup table the checkpoints `paths` use (plus the training table, paths[0])."""
    from agent.matchups import Matchups, has_hp_left, table_path
    from agent.senses import Senses
    out = {}
    for p in paths:
        stored = str(p) if str(p).endswith(".npz") else torch.load(p, map_location="cpu", weights_only=True).get("matchups")
        if not stored:                       # a network without matchup inputs (v1)
            continue
        tp = table_path(stored)
        if tp.name in out or not tp.is_file():
            continue
        mu = Matchups("cpu", tp)
        sense = Senses(mu, tp) if has_hp_left(tp) else None
        for obj in (mu, sense):
            for k, v in (vars(obj) if obj is not None else {}).items():
                if torch.is_tensor(v):
                    v.share_memory_()
        out[tp.name] = (mu, sense)
    return out


def seed_tables(tables, paths):
    """Make agent.arena.load_policy reuse the shared tables for every checkpoint in `paths`."""
    from agent import arena
    from agent.matchups import table_path
    for p in paths:
        stored = torch.load(p, map_location="cpu", weights_only=True).get("matchups")
        tp = table_path(stored) if stored else None
        if tp is not None and tp.name in tables:
            mu, sense = tables[tp.name]
            arena._TABLES[(str(tp), "cpu")] = mu
            if sense is not None:
                arena._TABLES[("senses", str(tp), "cpu")] = sense


def load_weights(policy, path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(ck["policy"])
    return ck["version"], ck.get("league", [])


def actor(rank, args, arch, tables, out, stop):
    """Plays chunks of --steps turns in --actor-envs battles forever, sending each to `out`."""
    torch.set_num_threads(args.actor_threads)
    from agent.policy import Policy
    from agent.selfplay import League, Runner, observe_on
    seed_tables(tables, list(args.league) + list(args.human))
    mu, sense = tables[Path(args.matchups).name]
    policy = Policy(**arch)
    policy.mu, policy.sense = mu, sense
    policy.eval()
    version, league_paths = load_weights(policy, args.weights)
    teams = None
    if args.teams == "smogon":
        from agent.teams import TeamGen
        teams = TeamGen()
    elif args.teams.startswith(("mix:", "mix1:")):             # specialisation on our own teams (agent/teams.TeamMix)
        from agent.teams import TeamMix
        teams = TeamMix(args.teams)
    n = args.actor_envs
    runner = Runner(n, args.seed + 7919 * (rank + 1), "cpu", teams)
    league = (League(league_paths, args.league_frac, n, "cpu", runner.rng, args.pfsp, args.human, args.human_frac, args.league_recent)
              if args.league_frac > 0 or args.human_frac > 0 else None)

    @torch.no_grad()
    def act(team, battle, mask):
        m = np.concatenate([mask[:, 0], mask[:, 1]])
        m[~m.any(-1), 11] = True
        o = [observe_on(policy, team, battle, s, "cpu") for s in (0, 1)]
        logits, _ = policy({k: torch.cat([o[0][k], o[1][k]]) for k in o[0]}, torch.from_numpy(m))
        d = torch.distributions.Categorical(logits=logits)
        a = d.sample()
        return a.numpy().reshape(2, n).T, d.log_prob(a).numpy().reshape(2, n).T

    while not stop.is_set():
        if os.path.getmtime(args.weights) > getattr(act, "loaded", 0):
            act.loaded = os.path.getmtime(args.weights)
            version, paths = load_weights(policy, args.weights)
            if league is not None:
                for p in paths[int((~league.is_human).sum()):]:
                    seed_tables(tables, [p])
                    league.add(p)
        buf = {k: [] for k in ("team", "battle", "mask", "action", "logp", "reward", "done", "learner_side")}
        finished, final_alive = [], []
        for t in range(args.steps):
            team, battle, mask = runner.state()
            a, logp = act(team, battle, mask)
            side = league.side.copy() if league is not None else np.full(n, BOTH)
            if league is not None:
                for s in (0, 1):
                    frozen = (side != s) & (side != BOTH)
                    if frozen.any():
                        a[:, s] = np.where(frozen, league.actions(team, battle, mask, s, frozen), a[:, s])
            runner.env.step(a.astype(np.int64))
            res = runner.env.results()
            done = res >= 0
            r0 = np.where(res == 0, 1.0, np.where(res == 1, -1.0, 0.0)) * done
            fa = np.zeros((n, 2, 6), np.float32)
            if done.any():
                fa[done] = runner.env.team_state()[done][..., ALIVE]
            for k, v in zip(buf, (team, battle, mask, a, logp, np.stack([r0, -r0], 1).astype(np.float32), done, side)):
                buf[k].append(v)
            final_alive.append(fa)
            finished += [(int(res[i]), int(runner.env.turns()[i])) for i in np.flatnonzero(done)]
            runner.finish(done)
            if league is not None and done.any():
                league.record(done, res)
                league.redraw(done)
        chunk = {k: np.stack(v) for k, v in buf.items()}
        chunk["last_team"], chunk["last_battle"], chunk["last_mask"] = runner.state()
        fa, done = np.stack(final_alive), chunk["done"]          # survivors at the end of each step's battle
        alive, known = np.zeros_like(fa), np.zeros(done.shape, bool)
        cur, ok = np.zeros(fa.shape[1:], np.float32), np.zeros(n, bool)
        for t in reversed(range(len(done))):
            cur[done[t]], ok[done[t]] = fa[t][done[t]], True
            alive[t], known[t] = cur, ok
        chunk["alive"], chunk["alive_known"] = alive, known
        chunk["version"], chunk["finished"], chunk["actor"] = version, finished, rank
        if league is not None:
            chunk["league_score"] = (league.score.copy(), league.games.copy())
        while not stop.is_set():
            try:
                out.put(chunk, timeout=5)
                break
            except queue.Full:
                pass


def observation_cache(policy, critic, buf, device, block=16384):
    """Every step's policy and critic observations (both sides), built once per iteration in big batches
    on the device; returns get(side, t, e) -> (policy obs, critic obs) for train_v4.update, plus the
    critic's (T, n, 2) values computed from the same tensors. Building per minibatch instead spends most
    of the learner's time in the table features' many small GPU ops."""
    T, n = buf["action"].shape[:2]
    team, battle = buf["team"].reshape(T * n, 2, 6, -1), buf["battle"].reshape(T * n, -1)
    mask = buf["mask"].reshape(T * n, 2, 12)
    cache, values = {}, np.zeros((T * n, 2), np.float32)
    for side in (0, 1):
        parts = []
        for k in range(0, T * n, block):
            sl = slice(k, k + block)
            parts.append(observe_views([(policy, team[sl], battle[sl], side), (critic, reveal(team[sl]), battle[sl], side)], device))
        po = {key: torch.cat([p[0][key] for p in parts]) for key in parts[0][0]}
        co = {key: torch.cat([p[1][key] for p in parts]) for key in parts[0][1]}
        cache[side] = (po, co)
        for k in range(0, T * n, block):
            m = mask[k:k + block, side].copy()
            m[~m.any(-1), 11] = True
            _, v = critic({key: x[k:k + block] for key, x in co.items()}, torch.from_numpy(m).to(device))
            values[k:k + block, side] = v.float().cpu().numpy()

    def get(side, t, e):
        idx = torch.from_numpy(t * n + e).to(device)
        po, co = cache[side]
        return {k: v[idx] for k, v in po.items()}, {k: v[idx] for k, v in co.items()}
    return get, values.reshape(T, n, 2)


def values_of(critic, team, battle, mask, device, block=8192):
    """(len, 2) critic values of positions from both sides' views, with their legal masks as in train_v4.act."""
    v = np.zeros((len(team), 2), np.float32)
    for side in (0, 1):
        for k in range(0, len(team), block):
            sl = slice(k, k + block)
            m = mask[sl, side].copy()
            m[~m.any(-1), 11] = True
            _, val = critic(critic_obs(critic, team[sl], battle[sl], side, device), torch.from_numpy(m).to(device))
            v[sl, side] = val.float().cpu().numpy()
    return v


def critic_values(critic, buf, device):
    """(T, n, 2) critic values of every step from both sides' views, and (n, 2) of the last states."""
    T, n = buf["action"].shape[:2]
    team = np.concatenate([buf["team"].reshape(T * n, *buf["team"].shape[2:]), buf["last_team"]])
    battle = np.concatenate([buf["battle"].reshape(T * n, -1), buf["last_battle"]])
    mask = np.concatenate([buf["mask"].reshape(T * n, 2, 12), buf["last_mask"]])
    v = values_of(critic, team, battle, mask, device)
    return v[:T * n].reshape(T, n, 2), v[T * n:]


def counterfactual(cf, critic, buf, args, rng):
    """train_v4's Q targets for up to (steps / cf_every) * cf_envs positions of the iteration."""
    import gen1py
    T, n = buf["action"].shape[:2]
    both = (buf["mask"][..., 0, :].sum(-1) > 1) & (buf["mask"][..., 1, :].sum(-1) > 1)
    cand = np.argwhere(both)
    k = min(len(cand), (T // args.cf_every) * args.cf_envs)
    if k == 0:
        return {}
    pick = cand[rng.choice(len(cand), k, replace=False)]
    env = gen1py.VecEnv(k, seed=int(rng.integers(1 << 31)))
    env.clear()
    for i, (t, e) in enumerate(pick):
        env.load(i, buf["team"][t, e].astype(np.int64), bytes(buf["battle"][t, e]), 0)
    mask = env.legal_mask()
    q = cf.targets(critic, SimpleNamespace(env=env), mask, np.arange(k))
    return {"cf_t": pick[:, 0], "cf_e": pick[:, 1], "cf_q": q, "cf_mask_agree": float((mask == buf["mask"][pick[:, 0], pick[:, 1]]).all((1, 2)).mean())}


def chance_targets(critic, buf, args, rng, device):
    """E over each turn's chance outcomes of r + gamma * V(next), from both sides' views, for a random
    args.chance_frac of the iteration's steps: the position is rebuilt (VecEnv.load), the joint action
    actually played is branched over its outcomes (damage rolls in args.chance_buckets groups, at most
    args.chance_max outcomes), and the critic values each. Returns {"ev": (T, n, 2), "ev_ok": (T, n)}."""
    import gen1py
    T, n = buf["action"].shape[:2]
    live = np.argwhere(buf["mask"].reshape(T, n, -1).any(-1))
    k = int(len(live) * args.chance_frac)
    ev, ok = np.zeros((T, n, 2), np.float32), np.zeros((T, n), bool)
    if k == 0:
        return {"ev": ev, "ev_ok": ok}
    pick = live[rng.choice(len(live), k, replace=False)]
    chunk = 256
    src = gen1py.VecEnv(chunk, seed=int(rng.integers(1 << 31)))
    sims = gen1py.VecEnv(chunk * args.chance_max, seed=int(rng.integers(1 << 31)))
    for c in range(0, k, chunk):
        part = pick[c:c + chunk]
        src.clear()
        for i, (t, e) in enumerate(part):
            src.load(i, buf["team"][t, e].astype(np.int64), bytes(buf["battle"][t, e]), 0)
        jobs = [(i, int(buf["action"][t, e, 0]), int(buf["action"][t, e, 1])) for i, (t, e) in enumerate(part)]
        try:
            idx, prob = sims.branch(src, jobs, buckets=args.chance_buckets, max_outcomes=args.chance_max)
        except ValueError:                       # an action the rebuilt position calls illegal: skip the chunk
            continue
        idx, prob, m = np.asarray(idx), np.asarray(prob), len(idx)
        res = np.asarray(sims.results())[:m]
        v = values_of(critic, sims.team_state()[:m].astype(np.int16), sims.battle_bytes()[:m], sims.legal_mask()[:m], device)
        r0 = np.where(res == 0, 1.0, np.where(res == 1, -1.0, 0.0))
        for s, sign in ((0, 1.0), (1, -1.0)):
            y = np.where(res >= 0, sign * r0, args.gamma * v[:, s])
            ev[part[:, 0], part[:, 1], s] = np.bincount(idx, weights=prob * y, minlength=len(part))
        ok[part[:, 0], part[:, 1]] = True
    return {"ev": ev, "ev_ok": ok}


def chance_check(buf, gamma):
    """Expected (ev) minus realized one-step targets on the marked steps: the mean should be ~0 (the
    luck removed averages out), the spread is the variance taken out of the advantages."""
    v, r, d = buf["value"], buf["reward"], buf["done"][..., None].astype(np.float32)
    nxt = np.concatenate([v[1:], buf["last_value"][None]])
    real = r + gamma * nxt * (1 - d)
    diff = (buf["ev"] - real)[buf["ev_ok"]]
    return {"chance_bias": round(float(diff.mean()), 4), "chance_luck_sd": round(float(diff.std()), 4)} if diff.size else {}


def main():
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--actors", type=int, default=4)
    ap.add_argument("--actor-envs", type=int, default=64)
    ap.add_argument("--actor-threads", type=int, default=1)
    ap.add_argument("--max-lag", type=int, default=2, help="drop chunks played by a policy more than this many versions old")
    ap.add_argument("--cache-obs", type=int, default=0, help="build every step's observations once per iteration (1) or per minibatch (0, faster on a 4090: the cache costs more to build than it saves)")
    ap.add_argument("--weights", help="where the learner publishes policy weights (default: the run directory)")
    ap.add_argument("--chance-frac", type=float, default=0.0, help="share of steps whose GAE target averages over the turn's chance outcomes (chance_targets)")
    ap.add_argument("--chance-buckets", type=int, default=3, help="damage-roll groups for --chance-frac")
    ap.add_argument("--chance-max", type=int, default=16, help="chance outcomes kept per step for --chance-frac")
    ap.add_argument("--pfsp", type=float, default=0.0, help="league opponents drawn with weight (1 - learner's score) ** PFSP + .05 (0: uniform)")
    ap.add_argument("--league-watch", help="a directory (another run's snapshots): every new policy there joins the league (AlphaStar main agent <-> exploiter)")
    ap.add_argument("--no-self-league", action="store_true", help="own snapshots don't join the league (an exploiter trains only against --league / --league-watch)")
    ap.add_argument("--league-recent", type=int, default=0, help="draw league opponents only from the newest N members (0: all); an exploiter plays the latest mains")
    ap.add_argument("--human", nargs="*", default=[], help="frozen human-imitation players with their own share --human-frac of the envs")
    ap.add_argument("--human-frac", type=float, default=0.0)
    ap.add_argument("--adv-filter", type=float, default=0.0, help="drop decisions whose |normalised advantage| is below this from the update (Ataraxos advantage filtering), keeping a random --adv-keep of them")
    ap.add_argument("--adv-keep", type=float, default=0.25)
    ap.add_argument("--distill-workers", type=int, default=0, help="search-target processes (agent/search_targets.py); needs --distill-coef")
    ap.add_argument("--distill-games", type=int, default=32, help="battles per search-target process")
    ap.add_argument("--distill-frac", type=float, default=0.05, help="share of real decisions a search-target process searches")
    ap.add_argument("--distill-buffer", type=int, default=50000, help="latest search targets kept for sampling")
    ap.add_argument("--distill-refresh", type=int, default=5, help="iterations between refreshes of the searching checkpoint pair")
    ap.add_argument("--distill-gpu-frac", type=float, default=0.08, help="share of the GPU's memory each search-target process may use")
    ap.add_argument("--distill-block", type=int, default=16, help="battles per search call in a search-target process")
    ap.add_argument("--distill-tau", type=float, default=0.05, help="the search's pull toward the policy's prior (larger: softer targets; the Showdown player uses .02)")
    args = ap.parse_args()
    assert args.envs % args.actor_envs == 0, "--envs must be a multiple of --actor-envs"
    from agent.policy import Policy
    device = args.device
    out = ROOT / "agent/runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    args.weights = args.weights or str(out / "actor_weights.pt")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    init = torch.load(args.init, map_location="cpu", weights_only=True)
    tables = shared_tables([args.matchups] + list(args.league) + list(args.human))
    mu_cpu, sense_cpu = tables[Path(args.matchups).name]
    if device == "cpu":
        mu, sense = mu_cpu, sense_cpu
    else:
        from agent.matchups import Matchups
        from agent.senses import Senses
        mu = Matchups(device, args.matchups)
        sense = Senses(mu, args.matchups)

    def build(ck, **extra):
        net = Policy(**{**dict(ck["arch"]), "senses": args.senses, **extra}).to(device)
        missing, _ = net.load_state_dict(ck["policy"], strict=False)
        net.mu, net.sense = mu, sense
        return net, len(missing)
    policy, n_new = build(init, aux=False, qhead=False, **({"opp_pp": True} if args.opp_pp else {}))
    c_init = torch.load(args.init_critic, map_location="cpu", weights_only=True) if args.init_critic else init
    critic, c_new = build(c_init, aux=args.aux_coef > 0, qhead=args.qhead, **({"opp_pp": True} if args.opp_pp else {}))
    print(f"init from {args.init}: policy {n_new} new tensors, critic {c_new}; "
          f"{sum(p.numel() for p in policy.parameters()):,} + {sum(p.numel() for p in critic.parameters()):,} parameters; device {device}", flush=True)
    league_paths = list(args.league)

    def publish(version):
        tmp = args.weights + ".tmp"
        torch.save({"policy": {k: v.cpu() for k, v in policy.state_dict().items()}, "version": version, "league": league_paths}, tmp)
        os.replace(tmp, args.weights)

    def save(stem, it):
        common = {"width": init["arch"]["d"], "iter": it, "matchups": portable(args.matchups)}
        torch.save({"policy": policy.state_dict(), "arch": policy.config, **common}, out / f"{stem}.pt")
        torch.save({"policy": critic.state_dict(), "arch": critic.config, "privileged": True, **common}, out / f"{stem}_critic.pt")

    def save_search():
        """RUN/search.pt + search_critic.pt for the search-target processes; the critic lands first,
        since they reload when the policy file changes."""
        common = {"width": init["arch"]["d"], "matchups": portable(args.matchups)}
        torch.save({"policy": critic.state_dict(), "arch": critic.config, "privileged": True, **common}, out / "search_critic.tmp")
        os.replace(out / "search_critic.tmp", out / "search_critic.pt")
        torch.save({"policy": policy.state_dict(), "arch": policy.config, **common}, out / "search.tmp")
        os.replace(out / "search.tmp", out / "search.pt")

    version = 0
    publish(version)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=args.lr)
    log = open(out / "log.jsonl", "a")
    log.write(json.dumps({"args": vars(args), "actor_learner": True}) + "\n")
    tbs = TBStream(args.tb_batch, args.seed) if args.tb_batch > 0 else None
    cf = Counterfactual(args.cf_seeds, device) if args.qhead else None
    ctx = mp.get_context("spawn")
    per_iter = args.envs // args.actor_envs
    chunks, stop = ctx.Queue(maxsize=per_iter), ctx.Event()   # actors run at most about a version ahead
    arch = dict(policy.config)
    procs = [ctx.Process(target=actor, args=(r, args, arch, tables, chunks, stop), daemon=True) for r in range(args.actors)]
    distill, dq, dprocs = None, None, []
    if args.distill_workers > 0 and args.distill_coef > 0:
        from agent.search_targets import DistillBuffer, worker as search_worker
        save_search()
        distill, dq = DistillBuffer(args.distill_buffer, args.seed), ctx.Queue(maxsize=64)
        dprocs = [ctx.Process(target=search_worker, args=(r, str(out / "search.pt"), device, args.distill_games, args.distill_frac,
                                                            256, args.seed, args.teams, dq, stop, {"tau": args.distill_tau, "block": args.distill_block},
                                                            None, args.distill_gpu_frac), daemon=True)
                  for r in range(args.distill_workers)]
    for p in procs + dprocs:
        p.start()
    magnet = None
    t0, ent0 = time.time(), args.ent_coef
    try:
        for it in range(1, args.iters + 1):
            frac = (it - 1) / max(args.iters - 1, 1)
            if args.lr_final is not None:
                for g in opt.param_groups:
                    g["lr"] = args.lr + (args.lr_final - args.lr) * frac
            if args.ent_final is not None:
                args.ent_coef = ent0 + (args.ent_final - ent0) * frac
            if args.magnet_coef > 0 and (magnet is None or (it - 1) % args.magnet_every == 0):
                magnet = copy.deepcopy(policy).eval()
                for p in magnet.parameters():
                    p.requires_grad_(False)
            t1 = time.time()
            got, dropped, lags = [], 0, []
            while len(got) < per_iter:
                try:
                    c = chunks.get(timeout=60)
                except queue.Empty:
                    dead = [p.pid for p in procs if not p.is_alive()]
                    if dead or time.time() - t1 > 1800:   # e.g. killed for memory: don't wait out the rental
                        raise RuntimeError(f"no games for {time.time() - t1:.0f}s; dead actors {dead} "
                                           f"(exit codes {[p.exitcode for p in procs if not p.is_alive()]})")
                    continue
                if version - c["version"] > args.max_lag:
                    dropped += 1
                    continue
                lags.append(version - c["version"])
                got.append(c)
            t2 = time.time()
            keys = ("team", "battle", "mask", "action", "logp", "reward", "done", "learner_side", "alive", "alive_known")
            buf = {k: np.concatenate([c[k] for c in got], 1) for k in keys}
            buf["last_team"] = np.concatenate([c["last_team"] for c in got])
            buf["last_battle"] = np.concatenate([c["last_battle"] for c in got])
            buf["last_mask"] = np.concatenate([c["last_mask"] for c in got])
            finished = [f for c in got for f in c["finished"]]
            league_rec = {}
            if any("league_score" in c for c in got):     # the actors' running scores in their player order
                names = list(args.league) + list(args.human) + league_paths[len(args.league):]
                sc = [c["league_score"] for c in got if "league_score" in c]
                for k, nm in enumerate(names):
                    g = sum(x[1][k] for x in sc if len(x[1]) > k)
                    if g > 0:
                        league_rec["/".join(x for x in Path(nm).with_suffix("").parts[-3:] if x != "snapshots")] = round(sum(x[0][k] * x[1][k] for x in sc if len(x[1]) > k) / g, 3)
            with torch.no_grad():
                critic.eval()
                cache = None
                if args.cache_obs:
                    cache, buf["value"] = observation_cache(policy, critic, buf, device)
                    buf["last_value"] = values_of(critic, buf["last_team"], buf["last_battle"], buf["last_mask"], device)
                else:
                    buf["value"], buf["last_value"] = critic_values(critic, buf, device)
                cfd = counterfactual(cf, critic, buf, args, rng) if cf is not None else {}
                t_ch = time.time()
                if args.chance_frac > 0:
                    buf.update(chance_targets(critic, buf, args, rng, device))
                t_ch = time.time() - t_ch
                critic.train()
            agree = cfd.pop("cf_mask_agree", None)
            buf.update(cfd)
            if args.league_watch:                         # the other learner's new snapshots join the league
                for f in sorted(Path(args.league_watch).glob("[0-9]*.pt")):
                    if not f.name.endswith("_critic.pt") and str(f) not in league_paths and time.time() - f.stat().st_mtime > 30:
                        league_paths.append(str(f))
            if args.league_frac <= 0 and args.human_frac <= 0:
                del buf["learner_side"]
            adv, ret = gae(buf, args.gamma, args.lam)
            d_new = 0
            while dq is not None:
                try:
                    c = dq.get_nowait()
                except queue.Empty:
                    break
                distill.add(c)
                d_new += len(c["pi"])
            if dprocs and not any(p.is_alive() for p in dprocs):
                raise RuntimeError(f"all search-target processes died (exit codes {[p.exitcode for p in dprocs]})")
            t3 = time.time()
            st = update(policy, critic, opt, buf, adv, ret, args, device, magnet, tbs, obs_cache=cache, distill=distill)
            del cache
            if device == "mps":
                torch.mps.empty_cache()
            version += 1
            rec = {"iter": it, "time": round(time.time() - t0), "wait_s": round(t2 - t1, 1), "prep_s": round(t3 - t2, 1),
                   "update_s": round(time.time() - t3, 1), "games": len(finished),
                   "turns": float(np.mean([f[1] for f in finished])) if finished else None,
                   "lag": float(np.mean(lags)), "dropped": dropped, "cf_mask_agree": agree,
                   **({"chance_s": round(t_ch, 1), "chance_n": int(buf["ev_ok"].sum()), **chance_check(buf, args.gamma)} if "ev_ok" in buf else {}),
                   **{k: round(v, 4) for k, v in st.items()}, "lr": opt.param_groups[0]["lr"], "ent_coef": round(args.ent_coef, 5)}
            if league_rec:
                rec["league"] = league_rec
            if distill is not None:
                rec.update(distill_new=d_new, distill_buf=len(distill))
                if it % args.distill_refresh == 0:
                    save_search()
            if it % args.snapshot_every == 0 or it == 1:
                (out / "snapshots").mkdir(exist_ok=True)
                save(f"snapshots/{it:05d}", it)
                if args.league_frac > 0 and it > 1 and not args.no_self_league:
                    league_paths.append(str(out / f"snapshots/{it:05d}.pt"))
            publish(version)
            if it % args.eval_every == 0 or it == 1:
                policy.eval()
                rec["eval"] = evaluate(policy, device, args.eval_games, 10_000 + it)
                policy.train()
                save("latest", it)
            if args.eval_vs and (it % args.eval_vs_every == 0 or it == 1):
                rec["vs"] = head_to_head(policy, args.eval_vs, args.eval_vs_games, 20_000 + it, device)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    finally:
        stop.set()
        for p in procs + dprocs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
