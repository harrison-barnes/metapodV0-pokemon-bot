"""v4 training: self-play that learns from exact answers (docs/V4_DESIGN.md, steps 2-3).

Two networks (both agent/policy.py, both warm-started from --init):
- the **policy** plays. It sees only what a player sees (plus --senses), and trains by PPO, the
  MMD magnet and (lightly) the tablebase's equilibrium mixes;
- the **critic** never plays. It reads a privileged observation in which the opposing team is fully
  revealed, and carries the value, "alive" and Q heads. It trains on returns, survivors,
  counterfactual turn targets and the tablebase's exact values. Keeping it separate means none
  of that pulls the policy's features around: in a first attempt with one shared network, value
  pretraining on endgames cost the policy 14 points against v3.

Options:
- --aux-coef: the critic's "alive" head learns which Pokemon are still standing when each battle
  ends, for steps whose battle finished inside the rollout.
- --tb-batch N --tb-coef C --tb-every K --tb-policy P: every K-th minibatch also trains on N exact
  positions from the 1v1 tablebase (agent/tablebase.py): the critic's value toward the solved
  value and its Q move-vs-move block toward the solved q, and (weight P) the policy toward the
  equilibrium mix. --pretrain S runs S tablebase steps for the critic before self-play.
- --qhead --cf-every S --cf-envs M --cf-seeds R: counterfactual turn targets. Every S rollout steps,
  M envs are copied (VecEnv.copy_from) and every pair of legal actions is played out in the
  copies (R times each), each result scored by the critic's value (+1/-1/0 if the battle ended).
  That gives a target for every entry of the payoff matrix at those positions.
- --magnet-coef A --magnet-every M: Magnetic Mirror Descent. The policy loss adds A * KL(policy ||
  magnet), the magnet a frozen copy of the policy refreshed every M iterations. Plain self-play
  cycles in games like this; the magnet makes the iterates converge.
- --senses: the multi-Pokemon inputs of agent/senses.py, for both networks.

Checkpoints: latest.pt / snapshots/ITER.pt hold the policy (agent/arena.py plays it);
latest_critic.pt / snapshots/ITER_critic.pt hold the critic, which agent/qsolve.py ("q:...") and
agent/search.py ("search:...") use.

Needs the engine's Python bindings (`gen1py`, built from engine/py by `uv sync`).

Usage: uv run python -m agent.train_v4 --name v4 [options]
"""
import argparse
import copy
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import gen1py
from agent.matchups import Matchups, default_device
from agent.obs import MOVES_REVEALED, PRESENT, REVEALED
from agent.policy import Policy
from agent.selfplay import (BOTH, ROOT, League, Runner, evaluate, gae, head_to_head, observe_on, observe_views)
from agent.tablebase import Tablebase
import torch.distributed as dist


def world():
    """(rank, size): several data-parallel copies under torchrun (one per GPU), else (0, 1)."""
    return (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)


def all_min(x, device):
    """The smallest x over the data-parallel copies (x itself when there is one)."""
    if world()[1] == 1:
        return x
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t.item()


def average_grads(params):
    """Average the gradients of every copy, in one flat all-reduce."""
    if world()[1] == 1:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= world()[1]
    off = 0
    for p, g in zip(params, grads):
        k = g.numel()
        p.grad = flat[off:off + k].view_as(p)
        off += k

ALIVE = 15


def portable(path):
    """A table path as stored in checkpoints: relative to the project when inside it."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def reveal(team):
    """The team arrays with every present member's species and moves visible to the opponent."""
    t = team.copy()
    present = t[..., PRESENT] > 0
    t[..., REVEALED] = np.where(present, 1, t[..., REVEALED])
    t[..., MOVES_REVEALED] = np.where(present, 15, t[..., MOVES_REVEALED])
    return t


def critic_obs(critic, team, battle, side, device):
    return observe_on(critic, reveal(team), battle, side, device)


def act(policy, critic, runner, team, battle, mask):
    """Sampled actions, log-probs and critic values for both sides of every env."""
    n, dev = runner.n, runner.device
    m = np.concatenate([mask[:, 0], mask[:, 1]])
    m[~m.any(-1), 11] = True
    m = torch.from_numpy(m).to(dev)
    cat = lambda xs: {k: torch.cat([x[k] for x in xs]) for k in xs[0]}
    with torch.no_grad():
        rt = reveal(team)
        o = observe_views([(policy, team, battle, 0), (policy, team, battle, 1), (critic, rt, battle, 0), (critic, rt, battle, 1)], dev)
        logits, _ = policy(cat(o[:2]), m)
        _, value = critic(cat(o[2:]), m)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
    to = lambda t: t.cpu().numpy().reshape(2, n).T
    return to(a), to(logp), to(value)


def critic_value(critic, team, battle, device, block=8192):
    """Side 0's critic value (-1..1) of each position."""
    out = np.zeros(len(team), np.float32)
    with torch.no_grad():
        for k in range(0, len(team), block):
            sl = slice(k, k + block)
            o = critic_obs(critic, team[sl], battle[sl], 0, device)
            _, v = critic(o, torch.ones(len(team[sl]), 12, dtype=torch.bool, device=device))
            out[sl] = v.float().cpu().numpy()
    return out


class Counterfactual:
    """Payoff-matrix targets: every pair of legal actions, played out in copies of real positions."""

    def __init__(self, seeds, device):
        self.seeds, self.device = seeds, device
        self.sims = None
        self.rng = np.random.default_rng(99)

    def targets(self, critic, runner, mask, choose):
        """Q targets (len(choose), 12, 12) from side 0's view, NaN for illegal pairs."""
        src, acts, owner = [], [], []
        for g, e in enumerate(choose):
            for a in np.flatnonzero(mask[e, 0]):
                for b in np.flatnonzero(mask[e, 1]):
                    for _ in range(self.seeds):
                        src.append(int(e)); acts.append((a, b)); owner.append((g, a, b))
        m = len(src)
        if self.sims is None or len(self.sims) < m:
            self.sims = gen1py.VecEnv(max(m, 2 * (len(self.sims) if self.sims else 0)), seed=7)
        self.sims.clear()
        self.sims.copy_from(runner.env, src, list(range(m)), int(self.rng.integers(1 << 62)))
        full = np.full((len(self.sims), 2), 11, np.int64)
        full[:m] = np.array(acts, np.int64)
        self.sims.step(full)
        res = self.sims.results()[:m]
        val = np.where(res == 0, 1.0, np.where(res == 1, -1.0, 0.0)).astype(np.float32)
        live = np.flatnonzero(res < 0)
        if len(live):
            t = self.sims.team_state().astype(np.int16)[:m][live]
            b = self.sims.battle_bytes()[:m][live]
            val[live] = critic_value(critic, t, b, self.device)
        q = np.zeros((len(choose), 12, 12), np.float32)
        cnt = np.zeros((len(choose), 12, 12), np.float32)
        o = np.array(owner)
        np.add.at(q, (o[:, 0], o[:, 1], o[:, 2]), val)
        np.add.at(cnt, (o[:, 0], o[:, 1], o[:, 2]), 1)
        return np.where(cnt > 0, q / np.maximum(cnt, 1), np.nan)


def rollout(runner, policy, critic, steps, args, opponent=None, learner_side=None, on_finish=None, cf=None):
    """As agent.selfplay.rollout, plus critic values from the privileged view, end-of-battle
    survivors and counterfactual Q targets at sampled steps."""
    buf = {k: [] for k in ("team", "battle", "mask", "action", "logp", "value", "reward", "done")}
    finished, sides, final_alive = [], [], []
    cf_t, cf_e, cf_q = [], [], []
    for t in range(steps):
        team, battle, mask = runner.state()
        a, logp, value = act(policy, critic, runner, team, battle, mask)
        if opponent is not None:
            for s in (0, 1):
                frozen = (learner_side != s) & (learner_side != BOTH)
                if frozen.any():
                    a[:, s] = np.where(frozen, opponent(team, battle, mask, s, frozen), a[:, s])
        if cf is not None and t % args.cf_every == 0:
            cand = np.flatnonzero((mask[:, 0].sum(-1) > 1) & (mask[:, 1].sum(-1) > 1))
            if len(cand):
                choose = runner.rng.choice(cand, min(args.cf_envs, len(cand)), replace=False)
                cf_t += [t] * len(choose); cf_e += list(choose); cf_q.append(cf.targets(critic, runner, mask, choose))
        runner.env.step(a.astype(np.int64))
        res = runner.env.results()
        done = res >= 0
        r0 = np.where(res == 0, 1.0, np.where(res == 1, -1.0, 0.0)) * done
        fa = np.zeros((runner.n, 2, 6), np.float32)
        if done.any():
            fa[done] = runner.env.team_state()[done][..., ALIVE]
        for k, v in zip(buf, (team, battle, mask, a, logp, value, np.stack([r0, -r0], 1), done)):
            buf[k].append(v)
        final_alive.append(fa)
        if learner_side is not None:
            sides.append(learner_side.copy())
        finished += [(int(res[i]), int(runner.env.turns()[i])) for i in np.flatnonzero(done)]
        runner.finish(done)
        if on_finish is not None and done.any():
            on_finish(done)
    team, battle, mask = runner.state()
    _, _, last_value = act(policy, critic, runner, team, battle, mask)
    out = {k: np.stack(v) for k, v in buf.items()}
    out["last_value"] = last_value
    if learner_side is not None:
        out["learner_side"] = np.stack(sides)
    # Survivors at the end of each step's battle, for steps whose battle ended in this rollout.
    fa, done = np.stack(final_alive), out["done"]
    alive = np.zeros_like(fa)
    known = np.zeros(done.shape, bool)
    cur, ok = np.zeros(fa.shape[1:], np.float32), np.zeros(done.shape[1], bool)
    for t in reversed(range(len(done))):
        cur[done[t]], ok[done[t]] = fa[t][done[t]], True
        alive[t], known[t] = cur, ok
    out["alive"], out["alive_known"] = alive, known
    if cf_q:
        out["cf_t"], out["cf_e"], out["cf_q"] = np.array(cf_t), np.array(cf_e), np.concatenate(cf_q)
    return out, finished


class TBStream:
    """Tablebase batches prefetched on a background thread."""

    def __init__(self, n, seed):
        self.tb, self.n = Tablebase(), n
        self.rng = np.random.default_rng(seed)
        self.next = None
        self._go()

    def _go(self):
        def work():
            self.next = self.tb.sample(self.n, self.rng)
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def get(self):
        self.thread.join()
        b = self.next
        self._go()
        return b


def tb_loss(policy, critic, b, device, policy_coef):
    """Critic value and Q losses, and the policy's equilibrium-mix loss, on a tablebase batch."""
    total, st = 0.0, {}
    for side in (0, 1):
        mask = torch.from_numpy(b["mask"][:, side]).to(device)
        _, value, ex = critic(critic_obs(critic, b["team"], b["battle"], side, device), mask, extras=True)
        sign = 1.0 if side == 0 else -1.0
        v_loss = ((value - torch.from_numpy(sign * b["value"]).to(device)) ** 2).mean()
        total = total + v_loss
        st[f"tb_v{side}"] = v_loss.item()
        if policy is not None and policy_coef > 0:
            logits, _ = policy(observe_on(policy, b["team"], b["battle"], side, device), mask)
            mix = torch.from_numpy(b["mix"][:, side]).to(device)
            logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
            p_loss = -(mix * logp).sum(-1).mean()
            total = total + policy_coef * p_loss
            st[f"tb_p{side}"] = p_loss.item()
        if critic.qhead:
            # Solved q (side 0's 0..1 value) for move rows x move columns.
            acts = b["actions"]                                                  # (n, 2, 4), -1 unused
            q = 2 * b["q"] - 1
            qq = q if side == 0 else -q.transpose(0, 2, 1)
            n = len(q)
            ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
            a, bb = acts[:, side][:, ii], acts[:, 1 - side][:, jj]               # (n, 4, 4)
            ok = (a >= 0) & (bb >= 0)
            r = np.repeat(np.arange(n), 16).reshape(n, 4, 4)
            to = lambda x: torch.from_numpy(x[ok]).to(device)
            q_loss = ((ex["q"][to(r), to(a), to(bb)] - to(qq)) ** 2).mean()
            total = total + q_loss
            st[f"tb_q{side}"] = q_loss.item()
    return total, st


def update(policy, critic, opt, buf, adv, ret, args, device, magnet, tbs, obs_cache=None, distill=None):
    """One PPO pass with the v4 losses (see the module docstring). obs_cache(side, t, e) -> (policy
    observation, critic observation), if given, replaces building them per minibatch (agent/actor_learner.py).
    distill (agent.search_targets.DistillBuffer), if given and not empty: every minibatch also pulls the
    policy toward the search's mix on args.distill_mb sampled search positions (policy only)."""
    T, n = buf["action"].shape[:2]
    idx = np.stack(np.meshgrid(np.arange(T), np.arange(n), np.arange(2), indexing="ij"), -1).reshape(-1, 3)
    ls = buf.get("learner_side")
    if ls is not None:
        l = ls[idx[:, 0], idx[:, 1]]
        idx = idx[(idx[:, 2] == l) | (l == BOTH)]
    choices = buf["mask"].sum(-1)
    thr = getattr(args, "adv_filter", 0.0)
    if thr > 0:                     # advantage filtering (Ataraxos): skip most decisions that carry little signal
        a_all = adv[idx[:, 0], idx[:, 1], idx[:, 2]]
        z = np.abs(a_all - a_all.mean()) / (a_all.std() + 1e-8)
        keep = (z >= thr) | (np.random.default_rng().random(len(idx)) < args.adv_keep)
        idx = idx[keep]
    cf_index = {}
    if "cf_q" in buf:
        cf_index = {(int(t), int(e)): i for i, (t, e) in enumerate(zip(buf["cf_t"], buf["cf_e"]))}
    stats, count = {}, 0
    add = lambda k, v: stats.__setitem__(k, stats.get(k, 0.0) + float(v))
    rng = np.random.default_rng()
    mb_no = 0
    # Every copy must take the same number of steps (each step all-reduces): the fewest any copy has.
    n_mb = int(all_min(-(-len(idx) // args.minibatch), device))
    params = list(policy.parameters()) + list(critic.parameters())
    # bf16 forward passes on CUDA (--bf16); the losses below are taken in float32
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(args, "bf16", False) and str(device).startswith("cuda"))
    for _ in range(args.epochs):
        rng.shuffle(idx)
        for k in range(0, n_mb * args.minibatch, args.minibatch):
            mb = idx[k:k + args.minibatch]
            loss = 0.0
            if obs_cache is None:
                views = []                               # both sides' observations in one batched call
                for side in (0, 1):
                    sel = mb[mb[:, 2] == side]
                    if len(sel):
                        tm, bt = buf["team"][sel[:, 0], sel[:, 1]], buf["battle"][sel[:, 0], sel[:, 1]]
                        views += [(policy, tm, bt, side), (critic, reveal(tm), bt, side)]
                obs = iter(observe_views(views, device))
            for side in (0, 1):
                sel = mb[mb[:, 2] == side]
                if len(sel) == 0:
                    continue
                t, e = sel[:, 0], sel[:, 1]
                team, battle = buf["team"][t, e], buf["battle"][t, e]
                mask = torch.from_numpy(buf["mask"][t, e, side]).to(device)
                o, oc = obs_cache(side, t, e) if obs_cache is not None else (next(obs), next(obs))
                with amp():
                    logits, _ = policy(o, mask)
                    _, value, ex = critic(oc, mask, extras=True)
                logits, value, ex = logits.float(), value.float(), {k: v.float() for k, v in ex.items()}
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(torch.from_numpy(buf["action"][t, e, side]).to(device))
                old = torch.from_numpy(buf["logp"][t, e, side]).to(device)
                a = torch.from_numpy(adv[t, e, side]).to(device)
                a = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)        # unbiased std of one sample is NaN
                live = torch.from_numpy(choices[t, e, side] > 1).to(device).float()
                nl = live.sum().clamp(min=1)
                ratio = (logp - old).exp()
                pg = -(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a) * live).sum() / nl
                vf = ((value - torch.from_numpy(ret[t, e, side]).to(device)) ** 2).mean()
                ent = (dist.entropy() * live).sum() / nl
                part = pg + args.vf_coef * vf - args.ent_coef * ent
                if magnet is not None:
                    with torch.no_grad(), amp():
                        m_logits, _ = magnet(o, mask)
                    m_logits = m_logits.float()
                    # Masked actions are -inf in both; drop them before differencing, or the
                    # gradient of 0 * (-inf - -inf) is NaN.
                    lp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
                    lm = torch.log_softmax(m_logits, -1).masked_fill(~mask, 0.0)
                    kl = ((torch.softmax(logits, -1) * (lp - lm)).sum(-1) * live).sum() / nl
                    part = part + args.magnet_coef * kl
                    add("magnet_kl", kl.item())
                if critic.aux and args.aux_coef > 0:
                    known = torch.from_numpy(buf["alive_known"][t, e]).to(device)
                    if known.any():
                        fa = buf["alive"][t, e]                                      # (b, 2, 6) absolute sides
                        target = np.concatenate([fa[:, side], fa[:, 1 - side]], 1)   # own six, then opponent's
                        bce = F.binary_cross_entropy_with_logits(ex["alive"], torch.from_numpy(target).to(device), reduction="none").mean(-1)
                        aux = (bce * known).sum() / known.sum()
                        part = part + args.aux_coef * aux
                        add("alive", aux.item())
                if critic.qhead and cf_index:
                    rows = [i for i, key in enumerate(zip(t.tolist(), e.tolist())) if key in cf_index]
                    if rows:
                        q_t = buf["cf_q"][[cf_index[(int(t[i]), int(e[i]))] for i in rows]]
                        if side == 1:
                            q_t = -q_t.transpose(0, 2, 1)
                        ok = torch.from_numpy(np.isfinite(q_t)).to(device)
                        pred = ex["q"][torch.as_tensor(rows, device=device)]
                        q_loss = (((pred - torch.from_numpy(np.nan_to_num(q_t)).to(device)) ** 2) * ok).sum() / ok.sum().clamp(min=1)
                        part = part + args.q_coef * q_loss
                        add("cf_q", q_loss.item())
                loss = loss + part
                add("pg", pg.item()); add("vf", vf.item()); add("ent", ent.item())
                add("kl", ((old - logp) * live).sum().item() / max(live.sum().item(), 1))
                count += 1
            if tbs is not None and args.tb_coef > 0 and mb_no % args.tb_every == 0:
                with amp():
                    tl, st = tb_loss(policy, critic, tbs.get(), device, args.tb_policy)
                loss = loss + args.tb_coef * tl
                for kk, vv in st.items():
                    add(kk, vv * 2 * args.tb_every)                                 # averaged like the per-side stats
            if distill is not None and len(distill) and args.distill_coef > 0:
                d = distill.sample(args.distill_mb)
                for side in (0, 1):
                    sel = np.flatnonzero(d["side"] == side)
                    if len(sel) == 0:
                        continue
                    (o,) = observe_views([(policy, d["team"][sel], d["battle"][sel], side)], device)
                    dm = torch.from_numpy(d["mask"][sel]).to(device)
                    with amp():
                        dl, _ = policy(o, dm)
                    lp = torch.log_softmax(dl.float(), -1).masked_fill(~dm, 0.0)
                    pi = torch.from_numpy(d["pi"][sel]).to(device)
                    ce = -(pi * lp).sum(-1).mean()
                    loss = loss + args.distill_coef * ce * len(sel) / len(d["pi"])
                    add("distill_ce", ce.item() * 2 * len(sel) / len(d["pi"]))     # averaged like the per-side stats
                    add("distill_top1", (dl.argmax(-1) == pi.argmax(-1)).float().mean().item() * 2 * len(sel) / len(d["pi"]))
            mb_no += 1
            opt.zero_grad(set_to_none=True)
            if not all_min(bool(torch.isfinite(loss)), device):  # never let one bad minibatch poison the weights
                add("skipped", 1)
                continue
            loss.backward()
            average_grads(params)
            norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if not torch.isfinite(norm):                 # identical on every copy after the average
                add("skipped", 1)
                continue
            opt.step()
    return {k: v / max(count, 1) for k, v in stats.items()}


def pretrain(critic, opt, tbs, steps, args, device, log):
    t0 = time.time()
    for s in range(1, steps + 1):
        loss, st = tb_loss(None, critic, tbs.get(), device, 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        average_grads(list(critic.parameters()))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
        opt.step()
        if s % 50 == 0 or s == 1:
            rec = {"pretrain": s, "time": round(time.time() - t0), **{k: round(v, 4) for k, v in st.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)


def add_args(ap):
    """The training options (shared with agent/actor_learner.py)."""
    for name, ty, default in [("envs", int, 512), ("steps", int, 128), ("iters", int, 1000), ("lr", float, 2e-4),
                              ("epochs", int, 1), ("minibatch", int, 4096), ("clip", float, 0.2), ("gamma", float, 0.995),
                              ("lam", float, 0.95), ("vf_coef", float, 0.5), ("ent_coef", float, 0.008), ("grad_clip", float, 0.5),
                              ("eval_every", int, 50), ("eval_games", int, 256), ("snapshot_every", int, 100),
                              ("league_frac", float, 0.0), ("aux_coef", float, 0.0), ("tb_batch", int, 0), ("tb_coef", float, 0.0),
                              ("tb_every", int, 4), ("tb_policy", float, 0.25), ("pretrain", int, 0),
                              ("cf_every", int, 16), ("cf_envs", int, 64), ("cf_seeds", int, 2), ("q_coef", float, 1.0),
                              ("magnet_coef", float, 0.0), ("magnet_every", int, 50), ("eval_vs_every", int, 50),
                              ("eval_vs_games", int, 1024), ("seed", int, 1)]:
        ap.add_argument("--" + name.replace("_", "-"), type=ty, default=default)
    ap.add_argument("--lr-final", type=float)
    ap.add_argument("--ent-final", type=float)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for the update's forward passes (CUDA only)")
    ap.add_argument("--opp-pp", action="store_true", help="the networks also observe the opponent's PP left on revealed moves (agent/obs.py); inputs unchanged in size, so any checkpoint can start with it")
    ap.add_argument("--distill-coef", type=float, default=0.0, help="weight of cross-entropy(search mix, policy) per minibatch (agent/search_targets.py; actor_learner only)")
    ap.add_argument("--distill-mb", type=int, default=256, help="search positions per minibatch for --distill-coef")
    ap.add_argument("--league", nargs="*", default=[])
    ap.add_argument("--eval-vs", nargs="*", default=[])
    ap.add_argument("--matchups", default=str(ROOT / "data/matchup_table_v2.npz"))
    ap.add_argument("--qhead", action="store_true")
    ap.add_argument("--senses", action="store_true", help="agent/senses.py inputs (needs a table with hp_left)")
    ap.add_argument("--init", required=True, help="checkpoint both networks start from")
    ap.add_argument("--teams", default="pool", type=lambda v: v if v in ("pool", "smogon") or v.startswith(("mix:", "mix1:")) else ap.error(f"bad --teams {v}"),
                    help="training teams: pool (independent usage draws) or smogon (agent/teams.py: teammate and lead statistics)")
    ap.add_argument("--init-critic", help="trained critic to continue from (e.g. snapshots/00500_critic.pt), instead of --init")
    ap.add_argument("--name", default="v4")
    ap.add_argument("--device", default=default_device())
    return ap


def main():
    ap = add_args(argparse.ArgumentParser())
    args = ap.parse_args()
    device = args.device
    import os
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:      # launched by torchrun
        local = int(os.environ.get("LOCAL_RANK", 0))
        if device.startswith("cuda"):
            device = f"cuda:{local}"
            torch.cuda.set_device(local)
        import datetime
        # copy 0's evaluations run while the others wait at their next all-reduce: allow for them
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo", timeout=datetime.timedelta(minutes=60))
    rank, size = world()
    lead = rank == 0                              # only copy 0 evaluates, saves and logs
    out = ROOT / "agent/runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)                  # same initial weights everywhere; rollouts differ by rank below
    init = torch.load(args.init, map_location=device, weights_only=True)
    base = dict(init["arch"])
    mu = Matchups(device, args.matchups)
    sense = None
    if args.senses:
        from agent.senses import Senses
        sense = Senses(mu, args.matchups)

    def build(ck=init, **extra):
        net = Policy(**{**dict(ck["arch"]), "senses": args.senses, **extra}).to(device)
        missing, unexpected = net.load_state_dict(ck["policy"], strict=False)
        net.mu, net.sense = mu, sense
        return net, len(missing)
    policy, n_new = build(aux=False, qhead=False)
    c_init = torch.load(args.init_critic, map_location=device, weights_only=True) if args.init_critic else init
    critic, c_new = build(c_init, aux=args.aux_coef > 0, qhead=args.qhead)
    print(f"init from {args.init}: policy {n_new} new tensors, critic {c_new}", flush=True)

    def save(stem, it):
        common = {"width": base["d"], "iter": it, "matchups": portable(args.matchups)}
        torch.save({"policy": policy.state_dict(), "arch": policy.config, **common}, out / f"{stem}.pt")
        torch.save({"policy": critic.state_dict(), "arch": critic.config, "privileged": True, **common}, out / f"{stem}_critic.pt")
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=args.lr)
    log = open(out / "log.jsonl", "a") if lead else open(os.devnull, "w")
    log.write(json.dumps({"args": vars(args), "world_size": size}) + "\n")
    tbs = TBStream(args.tb_batch, args.seed + 1000 * rank) if args.tb_batch > 0 else None
    print(f"{sum(p.numel() for p in policy.parameters()):,} + {sum(p.numel() for p in critic.parameters()):,} parameters; device {device}", flush=True)
    if args.pretrain and tbs is not None:
        c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
        pretrain(critic, c_opt, tbs, args.pretrain, args, device, log)
        if lead:
            save("pretrained", 0)
    teams = None
    if args.teams == "smogon":
        from agent.teams import TeamGen
        teams = TeamGen()
    runner = Runner(args.envs, args.seed + 1000 * rank, device, teams)
    league = League(args.league, args.league_frac, args.envs, device, runner.rng) if args.league_frac > 0 else None
    cf = Counterfactual(args.cf_seeds, device) if args.qhead else None
    magnet = None
    t0, ent0 = time.time(), args.ent_coef
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
        if league is not None:
            buf, finished = rollout(runner, policy, critic, args.steps, args, league.actions, league.side, league.redraw, cf)
        else:
            buf, finished = rollout(runner, policy, critic, args.steps, args, cf=cf)
        adv, ret = gae(buf, args.gamma, args.lam)
        t2 = time.time()
        st = update(policy, critic, opt, buf, adv, ret, args, device, magnet, tbs)
        if device == "mps":
            torch.mps.empty_cache()
        rec = {"iter": it, "time": round(time.time() - t0), "rollout_s": round(t2 - t1, 1), "update_s": round(time.time() - t2, 1),
               "games": len(finished) * size, "turns": float(np.mean([f[1] for f in finished])) if finished else None,
               **{k: round(v, 4) for k, v in st.items()}, "lr": opt.param_groups[0]["lr"], "ent_coef": round(args.ent_coef, 5)}
        if (it % args.eval_every == 0 or it == 1) and lead:
            policy.eval()
            rec["eval"] = evaluate(policy, device, args.eval_games, 10_000 + it)
            policy.train()
            save("latest", it)
        if args.eval_vs and (it % args.eval_vs_every == 0 or it == 1) and lead:
            rec["vs"] = head_to_head(policy, args.eval_vs, args.eval_vs_games, 20_000 + it, device)
        if it % args.snapshot_every == 0 or it == 1:
            (out / "snapshots").mkdir(exist_ok=True)
            if lead:
                save(f"snapshots/{it:05d}", it)
            if size > 1:
                # the copies' weights must be identical; and the snapshot must exist before every league loads it
                chk = sum(float(p.detach().double().sum()) for p in list(policy.parameters()) + list(critic.parameters()))
                if all_min(chk, device) != -all_min(-chk, device):
                    raise RuntimeError(f"data-parallel copies diverged at iteration {it}")
                dist.barrier()
            if league is not None and it > 1:
                league.add(out / f"snapshots/{it:05d}.pt")
        log.write(json.dumps(rec) + "\n"); log.flush()
        if lead:
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
