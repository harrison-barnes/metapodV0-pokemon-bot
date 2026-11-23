"""Supervised training on human Gen 1 OU battles (agent/metamon_convert.py shards): behavioural cloning
of the human's action plus the game result as the value target.

The network is agent/policy.py's Policy with the matchups / senses features of a reference checkpoint
(the exact 1v1 tables travel with it), either initialised from that checkpoint (--init) or fresh with
its architecture (default), optionally resized (--d, --layers). Checkpoints use the v4 format, so
agent.arena, the searches and the Showdown player load them unchanged.

Decisions train the policy only where the human's action is known and there was a real choice; every
decision trains the value head. The last shard (or --val-shard) is held out: top-1 agreement with the
human, cross-entropy and value MSE are reported there, with the reference checkpoint as a baseline.

    PYTHONPATH=. uv run python -m agent.train_bc --data data/metamon_gen1ou --train-shards 0-3 \
        --steps 4000 --out agent/runs/bc_trial
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agent.arena import load_policy
from agent.matchups import default_device
from agent.policy import Policy, skill_bucket
from agent.selfplay import observe_on


def load(paths):
    zs = [np.load(p) for p in paths]
    out = {k: np.concatenate([z[k] for z in zs]) for k in ("team", "bb", "mask", "action")}
    # per-decision game result from the per-game arrays (game indices restart in each shard)
    out["result"] = np.concatenate([z["result"][z["game"]] for z in zs]).astype(np.float32)
    out["skill"] = np.concatenate([skill_bucket(z["rating"], z["tour"])[z["game"]] for z in zs]).astype(np.int64)
    return out


def batches(data, idx, bs, rng):
    while True:
        perm = rng.permutation(idx)
        for c in range(0, len(perm) - bs + 1, bs):
            yield np.sort(perm[c:c + bs])


def forward(policy, data, sel, dev, skill=None):
    o = observe_on(policy, data["team"][sel].astype(np.int16), data["bb"][sel], 0, dev)
    if getattr(policy, "skill", False):
        sk = data["skill"][sel] if skill is None else np.full(len(sel), skill)
        o["skill"] = torch.from_numpy(sk).to(dev)
    mask = torch.from_numpy(data["mask"][sel]).to(dev)
    return policy(o, mask)


@torch.no_grad()
def evaluate(policy, data, idx, dev, bs=2048, skill=None):
    policy.eval()
    hits = ce = n_pol = mse = 0.0
    for c in range(0, len(idx), bs):
        sel = idx[c:c + bs]
        logits, value = forward(policy, data, sel, dev, skill)
        a = torch.from_numpy(data["action"][sel].astype(np.int64)).to(dev)
        pol = (a >= 0) & (torch.from_numpy(data["mask"][sel].sum(1) > 1).to(dev))
        lg = logits.float()[pol]
        hits += (lg.argmax(-1) == a[pol]).sum().item()
        ce += F.cross_entropy(lg, a[pol], reduction="sum").item()
        n_pol += pol.sum().item()
        mse += ((value.float() - torch.from_numpy(data["result"][sel]).to(dev)) ** 2).sum().item()
    policy.train()
    return {"acc": hits / n_pol, "ce": ce / n_pol, "value_mse": mse / len(idx)}


def shard_list(spec, root):
    files = sorted(Path(root).glob("shard*.npz"))
    a, b = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    return files[a:b + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metamon_gen1ou")
    ap.add_argument("--train-shards", default="0-18")
    ap.add_argument("--val-shard", type=int, default=-1)
    ap.add_argument("--val-size", type=int, default=50000)
    ap.add_argument("--ref", default="agent/runs/v4/snapshots/00500.pt", help="features (and --init weights) from here")
    ap.add_argument("--init", action="store_true", help="start from the reference weights")
    ap.add_argument("--d", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--skill", action="store_true", help="give the network the player's level (agent/policy.py skill)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = default_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref_ck = torch.load(args.ref, map_location="cpu", weights_only=True)
    ref = load_policy(args.ref, dev)
    arch = dict(ref_ck["arch"])
    arch.update({k: v for k, v in (("d", args.d), ("layers", args.layers)) if v})
    if args.skill:
        arch["skill"] = True
    policy = Policy(**arch).to(dev)
    if args.init:
        missing, unexpected = policy.load_state_dict(ref_ck["policy"], strict=False)
        assert not unexpected and all(k.startswith("skill_emb") for k in missing), (missing, unexpected)
    policy.mu = ref.mu
    if getattr(ref, "sense", None) is not None:
        policy.sense = ref.sense
    print(f"{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params, arch {arch}, init={args.init}", flush=True)

    files = sorted(Path(args.data).glob("shard*.npz"))
    train = load(shard_list(args.train_shards, args.data))
    val = load([files[args.val_shard]])
    vidx = rng.choice(len(val["action"]), min(args.val_size, len(val["action"])), replace=False)
    vidx.sort()
    print(f"train {len(train['action'])} decisions, val {len(vidx)}; reference: {evaluate(ref, val, vidx, dev)}", flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=max(0.05, 3 / args.steps))
    it = batches(train, np.arange(len(train["action"])), args.bs, rng)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = next(it)
        logits, value = forward(policy, train, sel, dev)
        a = torch.from_numpy(train["action"][sel].astype(np.int64)).to(dev)
        pol = (a >= 0) & torch.from_numpy(train["mask"][sel].sum(1) > 1).to(dev)
        loss_pi = F.cross_entropy(logits.float()[pol], a[pol])
        loss_v = F.mse_loss(value.float(), torch.from_numpy(train["result"][sel]).to(dev))
        loss = loss_pi + args.value_weight * loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(policy, val, vidx, dev)
            tour = vidx[val["skill"][vidx] == 4]
            if getattr(policy, "skill", False) and len(tour):
                ev["tour_acc_as_tour"] = evaluate(policy, val, tour, dev)["acc"]
                ev["tour_acc_as_low"] = evaluate(policy, val, tour, dev, skill=1)["acc"]
            row = {"step": step, "loss_pi": loss_pi.item(), "loss_v": loss_v.item(), **{f"val_{k}": v for k, v in ev.items()},
                   "sec": round(time.time() - t0)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            torch.save({"policy": policy.state_dict(), "arch": arch, "width": arch["d"], "iter": step,
                        "matchups": ref_ck["matchups"]}, out / "last.pt")


if __name__ == "__main__":
    main()
