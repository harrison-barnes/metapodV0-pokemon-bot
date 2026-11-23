"""Self-play PPO for the 6v6 agent (Smogon RBY OU rules, teams sampled from the ladder set pool).

Both sides of every battle are played by the current network. A rollout stores raw states (team
arrays, battle bytes, legal masks), not observations, which are rebuilt per minibatch to keep
memory small. Reward is +1 for a win, -1 for a loss and 0 for a draw, at the end of the battle.
Decisions where a side has a single legal action (waiting while the opponent replaces a
fainted Pokemon, recharging...) carry no policy gradient.

Every --eval-every iterations the greedy policy (most likely action) plays the benchmark bots
(agent/bots.py), half the games on each side, and latest.pt is updated. Every --snapshot-every
iterations a copy goes to snapshots/ITER.pt, for agent/arena.py to rate against each other.

Options added for v3:
- --league CKPT ... --league-frac F: a fraction F of the envs play a frozen opponent drawn from
  the listed checkpoints plus this run's own snapshots (added as they are saved), and only the
  learner's side of those games trains. The rest stay pure self-play. Each finished battle
  redraws its env's opponent and sides.
- --kickstart CKPT: an extra loss, KL(teacher || learner) on decisions with a choice, pulls a new
  (e.g. bigger) network toward a trained one, fading linearly to zero over --kick-iters.
- --lr-final, --ent-final: the learning rate and entropy bonus anneal linearly to these values.
- --eval-vs CKPT ...: every --eval-vs-every iterations, head-to-head scores against fixed agents.
- --layers, --heads, --direct: architecture (agent/policy.py); checkpoints store it as "arch".

Usage: uv run python -m agent.selfplay [--envs 512] [--steps 128] [--iters 1000] [--name run]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import gen1py
from agent.bots import greedy_actions, random_actions, smart_actions
from agent.obs import observe
from agent.matchups import Matchups, default_device
from agent.policy import Policy

ROOT = Path(__file__).resolve().parent.parent
POOL = json.load(open(ROOT / "data/sets_ladder.json"))
# Showdown's gen1ou bans Dig and Fly (data/mods/gen1/rulesets.ts); three low-weight heuristic sets in the
# pool carry one. They stay in the file, whose positions index the matchup tables, but are never drawn.
GEN1OU_BANNED_MOVES = {"DIG", "FLY", "DOUBLE_TEAM", "MINIMIZE", "FISSURE", "HORN_DRILL", "GUILLOTINE"}
WEIGHTS = np.array([0.0 if GEN1OU_BANNED_MOVES & set(p["moves"]) else p["weight"] for p in POOL])
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def sample_team(rng, size=6):
    """Six different species, each set drawn with its ladder usage weight."""
    team, used = [], set()
    while len(team) < size:
        s = POOL[rng.choice(len(POOL), p=WEIGHTS)]
        if s["species"] not in used:
            used.add(s["species"])
            team.append(s["species"] + ":" + ",".join(s["moves"]))
    return team


def to_device(o, device):
    return {k: torch.from_numpy(v).to(device) for k, v in o.items()}


def observe_on(policy, team, battle, side, device):
    """One side's observation as device tensors, with matchup features if the policy uses them."""
    o = to_device(observe(team, battle, side, getattr(policy, "opp_pp", False)), device)
    if policy.matchups:
        o.update(policy.mu.features(team, side))
    if getattr(policy, "senses", False):
        o.update(policy.sense.features(team, side))
    return o


def observe_views(views, device):
    """Several (net, team, battle, side) views at once: [obs dict per view]. The table features (the costly
    part: hundreds of small GPU ops each) run in one batched call per feature table for all the views
    that share it, since every row depends only on its own team array; the result equals calling
    observe_on once per view."""
    obs = [to_device(observe(t, b, s, getattr(net, "opp_pp", False)), device) for net, t, b, s in views]
    n = [len(t) for _, t, _, _ in views]
    for flag, attr in (("matchups", "mu"), ("senses", "sense")):
        groups = {}
        for i, (net, _, _, _) in enumerate(views):
            if getattr(net, flag, False):
                groups.setdefault(id(getattr(net, attr)), (getattr(net, attr), []))[1].append(i)
        for tab, idx in groups.values():
            # side s's view of team t is side 0's view of t with the sides swapped
            teams = np.concatenate([views[i][1] if views[i][3] == 0 else views[i][1][:, ::-1] for i in idx])
            f = tab.features(np.ascontiguousarray(teams), 0)
            off = 0
            for i in idx:
                for k, v in f.items():
                    obs[i][k] = v[off:off + n[i]]
                off += n[i]
    return obs


class Runner:
    def __init__(self, n, seed, device, teams=None):
        """teams: a function rng -> team (six set strings); default sample_team (independent usage
        draws), agent.teams.TeamGen() for teams built from Smogon's teammate and lead statistics."""
        self.n, self.device = n, device
        self.rng = np.random.default_rng(seed)
        self.teams = teams or sample_team
        self.env = gen1py.VecEnv(n, seed=seed)
        for i in range(n):
            self.env.reset(i, self.teams(self.rng), self.teams(self.rng))

    def state(self):
        return self.env.team_state().astype(np.int16), self.env.battle_bytes(), self.env.legal_mask()

    def act(self, policy, team, battle, mask, greedy=False):
        """Actions and log-probs for both sides from one batched forward pass."""
        obs = [observe_on(policy, team, battle, s, self.device) for s in (0, 1)]
        o = {k: torch.cat([obs[0][k], obs[1][k]]) for k in obs[0]}
        m = np.concatenate([mask[:, 0], mask[:, 1]])
        m[~m.any(-1), 11] = True  # finished battles have no legal action; the env ignores theirs
        m = torch.from_numpy(m).to(self.device)
        with torch.no_grad():
            logits, value = policy(o, m)
            dist = torch.distributions.Categorical(logits=logits)
            a = logits.argmax(-1) if greedy else dist.sample()
            logp = dist.log_prob(a)
        n = self.n
        to = lambda t: t.cpu().numpy().reshape(2, n).T
        return to(a), to(logp), to(value)

    def finish(self, done):
        for i in np.flatnonzero(done):
            self.env.reset(int(i), self.teams(self.rng), self.teams(self.rng))


BOTH = 2  # learner_side value: both sides are the learner (self-play)


def rollout(runner, policy, steps, opponent=None, learner_side=None, on_finish=None):
    """`steps` decisions in every env. With an opponent (a function (team, battle, mask, side,
    envs) -> actions, used for the envs flagged in `envs`), env e's side learner_side[e] is played
    by policy and the other by the opponent; learner_side[e] == BOTH means self-play. Data for the
    opponent's moves is stored too but carries no gradient (see ppo_update). on_finish(done) runs
    after finished envs reset."""
    buf = {k: [] for k in ("team", "battle", "mask", "action", "logp", "value", "reward", "done")}
    finished, sides = [], []
    for _ in range(steps):
        team, battle, mask = runner.state()
        a, logp, value = runner.act(policy, team, battle, mask)
        if opponent is not None:
            for s in (0, 1):
                frozen = (learner_side != s) & (learner_side != BOTH)
                if frozen.any():
                    a[:, s] = np.where(frozen, opponent(team, battle, mask, s, frozen), a[:, s])
        runner.env.step(a.astype(np.int64))
        res = runner.env.results()
        done = res >= 0
        r0 = np.where(res == 0, 1.0, np.where(res == 1, -1.0, 0.0)) * done
        for k, v in zip(buf, (team, battle, mask, a, logp, value, np.stack([r0, -r0], 1), done)):
            buf[k].append(v)
        if learner_side is not None:
            sides.append(learner_side.copy())           # before on_finish redraws sides
        finished += [(int(res[i]), int(runner.env.turns()[i])) for i in np.flatnonzero(done)]
        runner.finish(done)
        if on_finish is not None and done.any():
            on_finish(done)
    team, battle, mask = runner.state()
    _, _, last_value = runner.act(policy, team, battle, mask)
    out = {k: np.stack(v) for k, v in buf.items()}            # (T, n, ...)
    out["last_value"] = last_value
    if learner_side is not None:
        out["learner_side"] = np.stack(sides)                    # (T, n)
    return out, finished


def gae(buf, gamma, lam):
    """GAE advantages and returns, (T, n, 2). If buf has "ev" (T, n, 2) with "ev_ok" (T, n), the one-step
    target r + gamma * V(next) of the marked steps is replaced by its expectation over the turn's chance
    outcomes (agent/actor_learner.chance_targets): the dice of that turn then drop out of every
    advantage (Mahjong's reward variance reduction; the luck term telescopes away)."""
    v, r, d = buf["value"], buf["reward"], buf["done"][..., None].astype(np.float32)
    ev, ok = buf.get("ev"), buf.get("ev_ok")
    T = len(v)
    adv = np.zeros_like(v)
    next_v, last = buf["last_value"], np.zeros_like(v[0])
    for t in reversed(range(T)):
        nonterminal = 1 - d[t]
        target = r[t] + gamma * next_v * nonterminal
        if ev is not None:
            target = np.where(ok[t][:, None], ev[t], target)
        delta = target - v[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        next_v = v[t]
    return adv, adv + v


def ppo_update(policy, opt, buf, adv, ret, args, device, learner_side=None, teacher=None, kick=0.0):
    """PPO epochs over the rollout's decisions: both sides', or only env e's side learner_side[e]
    (per env, or per step and env when shaped (T, n), since league envs change opponents mid-
    rollout; BOTH means both). With a teacher, adds kick * KL(teacher || policy)."""
    T, n = buf["action"].shape[:2]
    idx = np.stack(np.meshgrid(np.arange(T), np.arange(n), np.arange(2), indexing="ij"), -1).reshape(-1, 3)
    if learner_side is not None:
        ls = learner_side[idx[:, 0], idx[:, 1]] if learner_side.ndim == 2 else learner_side[idx[:, 1]]
        idx = idx[(idx[:, 2] == ls) | (ls == BOTH)]
    choices = buf["mask"].sum(-1)                              # (T, n, 2) legal actions per side
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "n": 0}
    if teacher is not None and kick > 0:
        stats["kick_kl"] = 0.0
    rng = np.random.default_rng()
    for _ in range(args.epochs):
        rng.shuffle(idx)
        for k in range(0, len(idx), args.minibatch):
            mb = idx[k:k + args.minibatch]
            loss_parts = []
            for side in (0, 1):
                sel = mb[mb[:, 2] == side]
                if len(sel) == 0:
                    continue
                t, e = sel[:, 0], sel[:, 1]
                o = observe_on(policy, buf["team"][t, e], buf["battle"][t, e], side, device)
                mask = torch.from_numpy(buf["mask"][t, e, side]).to(device)
                logits, value = policy(o, mask)
                dist = torch.distributions.Categorical(logits=logits)
                act = torch.from_numpy(buf["action"][t, e, side]).to(device)
                logp = dist.log_prob(act)
                old = torch.from_numpy(buf["logp"][t, e, side]).to(device)
                a = torch.from_numpy(adv[t, e, side]).to(device)
                a = (a - a.mean()) / (a.std() + 1e-8)
                live = torch.from_numpy(choices[t, e, side] > 1).to(device).float()
                ratio = (logp - old).exp()
                pg = -(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a) * live).sum() / live.sum().clamp(min=1)
                vf = ((value - torch.from_numpy(ret[t, e, side]).to(device)) ** 2).mean()
                ent = (dist.entropy() * live).sum() / live.sum().clamp(min=1)
                loss = pg + args.vf_coef * vf - args.ent_coef * ent
                if teacher is not None and kick > 0:
                    with torch.no_grad():
                        t_logits, _ = teacher(o, mask)
                    t_logp = torch.log_softmax(t_logits, -1)
                    s_logp = torch.log_softmax(logits, -1)
                    # Only legal actions count; masked entries are -inf in both.
                    kl = (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0).sum(-1)
                    kl = (kl * live).sum() / live.sum().clamp(min=1)
                    loss = loss + kick * kl
                    stats["kick_kl"] += kl.item()
                loss_parts.append(loss)
                with torch.no_grad():
                    stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item()
                    stats["kl"] += ((old - logp) * live).sum().item() / max(live.sum().item(), 1)
                    stats["n"] += 1
            opt.zero_grad(set_to_none=True)
            sum(loss_parts).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()
    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}


def evaluate(policy, device, games, seed):
    """Win rate (draws half) of the greedy policy against each benchmark bot."""
    out = {}
    bots = {"random": lambda mask, team, s, rng: random_actions(mask, s, rng), "greedy": greedy_actions, "smart": smart_actions}
    for name, bot_actions in bots.items():
        runner = Runner(games, seed, device)
        side_of_agent = np.arange(games) % 2                 # half the games on each side
        done = np.zeros(games, bool)
        score = np.zeros(games)
        for _ in range(1000):
            team, battle, mask = runner.state()
            a, _, _ = runner.act(policy, team, battle, mask, greedy=True)
            bot = np.stack([bot_actions(mask, team, s, runner.rng) for s in (0, 1)], 1)
            acts = np.where(side_of_agent[:, None] == np.arange(2)[None], a, bot)
            runner.env.step(acts.astype(np.int64))
            res = runner.env.results()
            new = (res >= 0) & ~done
            score[new] = np.where(res[new] == 2, 0.5, (res[new] == side_of_agent[new]).astype(float))
            done |= res >= 0
            if done.all():
                break
        out[name] = float(score[done].mean())
    return out


class League:
    """Frozen opponents for a fraction of the envs (see the module docstring). opp[e] is env e's
    opponent index in self.players, or -1 for self-play; side[e] is the learner's side, or BOTH."""

    def __init__(self, paths, frac, n, device, rng, pfsp=0.0, human=(), human_frac=0.0, recent=0):
        """pfsp > 0: draw league opponents with weight (1 - p) ** pfsp + 0.05, p the learner's running score
        against each (AlphaStar's prioritised fictitious self-play: mostly the ones it doesn't beat yet).
        human: more frozen players (e.g. human-imitation nets) that get their own share human_frac of the
        envs, drawn uniformly, so PFSP never starves them. They are players[len(paths):] after __init__;
        add() puts new league members after them, and PFSP covers every non-human player (only the newest
        `recent` of them if recent > 0)."""
        from agent.arena import Player
        self.Player, self.device, self.frac, self.rng = Player, device, frac, rng
        self.pfsp, self.human_frac, self.recent = pfsp, human_frac, recent
        self.players = [Player(p, device) for p in list(paths) + list(human)]
        self.is_human = np.array([False] * len(paths) + [True] * len(human))
        self.score = np.full(len(self.players), 0.5)       # running learner score against each player
        self.games = np.zeros(len(self.players))
        self.opp = np.full(n, -1)
        self.side = np.full(n, BOTH)
        self.redraw(np.ones(n, bool))

    def add(self, path):
        self.players.append(self.Player(str(path), self.device))
        self.is_human = np.append(self.is_human, False)
        self.score, self.games = np.append(self.score, 0.5), np.append(self.games, 0)

    def record(self, envs, results):
        """Update the running scores from battles that just ended (results: 0/1 side won, 2 draw)."""
        for e in np.flatnonzero(envs):
            k = self.opp[e]
            if k < 0:
                continue
            r = results[e]
            s = 0.5 if r == 2 else float(r == self.side[e])
            self.games[k] += 1
            self.score[k] += (s - self.score[k]) / min(self.games[k], 200)   # mean, then an EMA over ~200 games

    def redraw(self, envs):
        league = np.flatnonzero(~self.is_human)
        if self.recent > 0:
            league = league[-self.recent:]
        human = np.flatnonzero(self.is_human)
        w = (1 - self.score[league]) ** self.pfsp + 0.05 if self.pfsp > 0 else np.ones(len(league))
        for e in np.flatnonzero(envs):
            r = self.rng.random()
            if len(human) and r < self.human_frac:
                self.opp[e], self.side[e] = self.rng.choice(human), self.rng.integers(2)
            elif len(league) and r < self.human_frac + self.frac:
                self.opp[e], self.side[e] = self.rng.choice(league, p=w / w.sum()), self.rng.integers(2)
            else:
                self.opp[e], self.side[e] = -1, BOTH

    def actions(self, team, battle, mask, side, envs):
        """Opponents' actions on `side` for the flagged envs, each frozen player on its own envs."""
        out = np.zeros(len(team), np.int64)
        for k in np.unique(self.opp[envs]):
            sel = envs & (self.opp == k)
            out[sel] = self.players[k].actions(team[sel], battle[sel], mask[sel], side, self.rng)
        return out


def head_to_head(policy, paths, games, seed, device):
    """Score (draws half) of the greedy policy against each fixed checkpoint."""
    from agent.arena import Player, match
    me = Player("greedy", device)
    me.bot, me.policy, me.name = None, policy, "learner"
    policy.eval()
    out = {Path(p).parent.parent.name + "/" + Path(p).stem: match(me, Player(p, device), games, seed, device)[0] for p in paths}
    policy.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=512)
    ap.add_argument("--snapshot-every", type=int, default=100, help="keep a copy of the policy every N iterations")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--direct", action="store_true", help="action heads read the opposing active directly")
    ap.add_argument("--league", nargs="*", default=[], help="frozen opponent checkpoints")
    ap.add_argument("--league-frac", type=float, default=0.0)
    ap.add_argument("--kickstart", help="teacher checkpoint")
    ap.add_argument("--kick-iters", type=int, default=150)
    ap.add_argument("--kick-coef", type=float, default=1.0)
    ap.add_argument("--lr-final", type=float, help="anneal the learning rate linearly to this")
    ap.add_argument("--ent-final", type=float, help="anneal the entropy bonus linearly to this")
    ap.add_argument("--eval-vs", nargs="*", default=[], help="checkpoints to score head-to-head against")
    ap.add_argument("--eval-vs-every", type=int, default=50)
    ap.add_argument("--eval-vs-games", type=int, default=1024)
    ap.add_argument("--matchups", help="matchup table (tools/build_matchup_table.py) to feed the policy")
    ap.add_argument("--name", default="selfplay")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    device = default_device()
    out = ROOT / "agent/runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    policy = Policy(d=args.width, layers=args.layers, heads=args.heads, matchups=bool(args.matchups), direct=args.direct).to(device)
    if args.matchups:
        policy.mu = Matchups(device, args.matchups)
    save = lambda path, it: torch.save({"policy": policy.state_dict(), "width": args.width, "iter": it, "matchups": args.matchups,
                                        "arch": policy.config}, path)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    runner = Runner(args.envs, args.seed, device)
    league = League(args.league, args.league_frac, args.envs, device, runner.rng) if args.league_frac > 0 else None
    teacher = None
    if args.kickstart:
        from agent.arena import load_policy
        teacher = load_policy(args.kickstart, device)
    print(f"{sum(p.numel() for p in policy.parameters()):,} parameters; {args.envs} envs x {args.steps} steps; device {device}", flush=True)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    ent0 = args.ent_coef
    for it in range(1, args.iters + 1):
        frac = (it - 1) / max(args.iters - 1, 1)
        if args.lr_final is not None:
            for g in opt.param_groups:
                g["lr"] = args.lr + (args.lr_final - args.lr) * frac
        if args.ent_final is not None:
            args.ent_coef = ent0 + (args.ent_final - ent0) * frac
        kick = args.kick_coef * max(0.0, 1 - (it - 1) / args.kick_iters) if teacher is not None else 0.0
        t1 = time.time()
        if league is not None:
            buf, finished = rollout(runner, policy, args.steps, league.actions, league.side, on_finish=league.redraw)
        else:
            buf, finished = rollout(runner, policy, args.steps)
        adv, ret = gae(buf, args.gamma, args.lam)
        t2 = time.time()
        st = ppo_update(policy, opt, buf, adv, ret, args, device, buf.get("learner_side"), teacher, kick)
        if device == "mps":
            torch.mps.empty_cache()   # minibatch sizes vary by side; cached blocks otherwise pile up
        rec = {"iter": it, "time": round(time.time() - t0), "rollout_s": round(t2 - t1, 1), "update_s": round(time.time() - t2, 1),
               "games": len(finished), "turns": float(np.mean([f[1] for f in finished])) if finished else None,
               "draws": float(np.mean([f[0] == 2 for f in finished])) if finished else None, **{k: round(v, 4) for k, v in st.items()}}
        rec["lr"], rec["ent_coef"], rec["kick"] = opt.param_groups[0]["lr"], round(args.ent_coef, 5), round(kick, 3)
        if league is not None:
            rec["league_envs"] = int((league.opp >= 0).sum())
        if it % args.eval_every == 0 or it == 1:
            policy.eval()
            rec["eval"] = evaluate(policy, device, args.eval_games, 10_000 + it)
            policy.train()
            save(out / "latest.pt", it)
        if args.eval_vs and (it % args.eval_vs_every == 0 or it == 1):
            rec["vs"] = head_to_head(policy, args.eval_vs, args.eval_vs_games, 20_000 + it, device)
        if it % args.snapshot_every == 0 or it == 1:
            (out / "snapshots").mkdir(exist_ok=True)
            save(out / f"snapshots/{it:05d}.pt", it)
            if league is not None and it > 1:
                league.add(out / f"snapshots/{it:05d}.pt")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
