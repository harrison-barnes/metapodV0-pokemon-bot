"""Exact 1v1 positions from the solver's dataset, dressed as 6v6 endgames for the 6v6 network.

When each side has one Pokemon left, a 6v6 battle is a 1v1 game, and the dataset (docs/DATASET.md)
holds its exact solution: the equilibrium value, each side's optimal move mix, and the value of
every pair of move choices (q). The 6v6 env's battle bytes use the dataset's battle layout, so a
row becomes a 6v6 position by building the team arrays around it:
- each side's active Pokemon comes from the row (species, moves, HP, status), with full PP;
- five fainted team members per side, drawn from the ladder pool by usage (distinct species),
  all revealed, as they would be by the time a battle gets down to 1v1;
- the opponent has seen each of the active's moves with probability 0.7, so the network learns
  the endgame under the hidden information it really has.

Sources: the main dataset (data/dataset_aws/shard-*) and the healing-PP re-solve
(data/dataset_pp/*), whose rows replace the main dataset's for the pairs it solved. Rows whose
clause flags can't occur in a 1v1 endgame (sleep or freeze blocked by a Pokemon that isn't there)
are skipped, and so are rows the solver flagged as likely endless stalls.

    tb = Tablebase()
    batch = tb.sample(4096, rng)   # team, battle, mask, value (side 0's view, -1..1), mix (2, 12), q (4, 4)
"""
import glob
import json
from pathlib import Path

import numpy as np

from agent.obs import ACTIVE, ALIVE, HP, MAX_HP, MOVES, MOVES_REVEALED, PP, PRESENT, REVEALED, SLEPT_BY_FOE, SPECIES, STATUS, MAX_PP
from analysis.dataset import decode_sides, row_dtype

ROOT = Path(__file__).resolve().parent.parent
TEAM_FIELDS = 18
SLP, FRZ = 0b111, 1 << 5
SLEEP_BLOCKED, FREEZE_BLOCKED, SLEPT_BY_FOE_BIT = 1, 2, 4
DEX = json.load(open(ROOT / "data/gen1.json"))
SPECIES_ID = {s["name"]: i for i, s in enumerate(DEX["species"])}
MOVE_ID = {m["name"]: m["id"] for m in DEX["moves"] if m}


def _shard(path):
    """(memmap rows, done records) for one dataset directory."""
    path = Path(path)
    schema = json.load(open(path / "schema.json"))
    done = [json.loads(l) for l in open(path / "done.jsonl")]
    end = max((d["offset"] + d["rows"] * schema["row_bytes"] for d in done), default=0)
    rows = np.memmap(path / "rows.bin", dtype=row_dtype(schema), mode="r", shape=(end // schema["row_bytes"],))
    return rows, done, schema


# Showdown's gen1ou bans these (data/mods/gen1/rulesets.ts); pairs and dressing teams that use them are skipped.
GEN1OU_BANNED_MOVES = {"DIG", "FLY", "DOUBLE_TEAM", "MINIMIZE", "FISSURE", "HORN_DRILL", "GUILLOTINE"}


class Tablebase:
    def __init__(self, main=ROOT / "data/dataset_aws", pp=ROOT / "data/dataset_pp", pool=ROOT / "data/sets_ladder.json",
                 max_endless=0.01, reveal_p=0.7):
        self.reveal_p = reveal_p
        pool = json.load(open(pool))
        banned = {i for i, s in enumerate(pool) if GEN1OU_BANNED_MOVES & set(s["moves"])}
        self.shards, spans = [], []
        pp_pairs = set()
        pp_dirs = sorted(glob.glob(str(Path(pp) / "*_shard-*")))
        for d in pp_dirs:
            _, done, _ = _shard(d)
            pp_pairs |= {(r["a"], r["b"]) for r in done if r["rows"]}
        for d in sorted(glob.glob(str(Path(main) / "shard-*"))) + pp_dirs:
            rows, done, schema = _shard(d)
            is_pp = d in pp_dirs
            k = len(self.shards)
            self.shards.append(rows)
            self.schema = schema
            for r in done:
                if r["rows"] and (is_pp or (r["a"], r["b"]) not in pp_pairs) and not banned & {r["a"], r["b"]}:
                    start = r["offset"] // schema["row_bytes"]
                    spans.append((k, start, r["rows"]))
        self.spans = np.array(spans, np.int64)                      # (shard, first row, count) per pair
        self.cum = np.cumsum(self.spans[:, 2])
        self.n = int(self.cum[-1])
        self.max_endless = max_endless
        self.pool_species = np.array([SPECIES_ID[s["species"]] for s in pool])
        self.pool_moves = np.array([[MOVE_ID[m] for m in s["moves"]] + [0] * (4 - len(s["moves"])) for s in pool])
        w = np.array([0.0 if i in banned else s["weight"] for i, s in enumerate(pool)], np.float64)
        self.pool_w = w / w.sum()

    def __len__(self):
        return self.n

    def _rows(self, idx):
        """Rows for global indices (sorted per shard for memmap locality)."""
        pair = np.searchsorted(self.cum, idx, side="right")
        before = self.cum[pair] - self.spans[pair, 2]
        shard, local = self.spans[pair, 0], self.spans[pair, 1] + (idx - before)
        out = np.empty(len(idx), self.shards[0].dtype)
        for k in np.unique(shard):
            sel = np.flatnonzero(shard == k)
            order = np.argsort(local[sel])
            out[sel[order]] = self.shards[k][local[sel][order]]
        return out

    def _consistent(self, rows, sides):
        ok = np.isfinite(rows["value"]) & (rows["p_endless"] <= self.max_endless)
        cl, st = sides["clauses"], sides["status"]
        for s in (0, 1):
            o = 1 - s
            sleeper = ((st[:, o] & SLP) > 0) & ((cl[:, o] & SLEPT_BY_FOE_BIT) > 0)
            ok &= ((cl[:, s] & SLEEP_BLOCKED) > 0) == sleeper
            ok &= ((cl[:, s] & FREEZE_BLOCKED) == 0) | ((st[:, o] & FRZ) > 0)
        return ok

    def _fillers(self, n, active_species, rng):
        """Five fainted, revealed team members per (env, side): species and moves. Each comes from a
        random team in a bank of usage-weighted six-species teams, minus the survivor's species."""
        if not hasattr(self, "bank_sp"):
            brng = np.random.default_rng(12345)
            B = 20000
            self.bank_sp = np.zeros((B, 6), np.int64)
            self.bank_mv = np.zeros((B, 6, 4), np.int64)
            for i in range(B):
                used, k = set(), 0
                while k < 6:
                    d = brng.choice(len(self.pool_w), p=self.pool_w)
                    if self.pool_species[d] not in used:
                        used.add(self.pool_species[d])
                        self.bank_sp[i, k], self.bank_mv[i, k] = self.pool_species[d], self.pool_moves[d]
                        k += 1
        pick = rng.integers(0, len(self.bank_sp), (n, 2))
        sp, mv = self.bank_sp[pick], self.bank_mv[pick]                       # (n, 2, 6), (n, 2, 6, 4)
        keep = np.argsort(sp == active_species[..., None], axis=-1, kind="stable")[..., :5]
        return np.take_along_axis(sp, keep, -1), np.take_along_axis(mv, keep[..., None], -2)

    def sample(self, n, rng):
        """n random consistent positions as 6v6 arrays (see the module docstring)."""
        rows = np.empty(0, self.shards[0].dtype)
        while len(rows) < n:
            idx = np.sort(rng.integers(0, self.n, int(n * 1.3) + 16))
            r = self._rows(idx)
            r = r[self._consistent(r, decode_sides(r["battle"], self.schema))]
            rows = np.concatenate([rows, r])
        rows = rows[rng.permutation(len(rows))[:n]]
        sides = decode_sides(rows["battle"], self.schema)
        team = np.zeros((n, 2, 6, TEAM_FIELDS), np.int16)
        slot = rng.integers(0, 6, (n, 2))                                     # the survivor's team slot
        fsp, fmv = self._fillers(n, sides["species"], rng)
        for s in (0, 1):
            act = slot[:, s, None] == np.arange(6)[None]                       # (n, 6)
            # Filler index for each non-survivor slot: how many non-survivor slots come before it.
            k = np.minimum(np.cumsum(~act, 1) - 1, 4).clip(0)
            t = team[:, s]
            t[..., PRESENT] = 1
            t[..., REVEALED] = 1
            t[..., SPECIES] = np.where(act, sides["species"][:, s, None], np.take_along_axis(fsp[:, s], k, 1))
            moves = np.where(act[..., None], sides["moves"][:, s, None], np.take_along_axis(fmv[:, s], k[..., None], 1))
            t[..., MOVES] = moves
            t[..., PP] = MAX_PP[moves]
            t[..., MOVES_REVEALED] = (rng.random((n, 6, 4)) < self.reveal_p) @ (1 << np.arange(4))
            t[..., ACTIVE] = act
            t[..., ALIVE] = act
            t[..., HP] = np.where(act, sides["hp"][:, s, None], 0)
            t[..., STATUS] = np.where(act, sides["status"][:, s, None], 0)
            t[..., SLEPT_BY_FOE] = act & ((sides["clauses"][:, s, None] & SLEPT_BY_FOE_BIT) > 0)
        from agent.features import MAX_HP as SPECIES_MAX_HP
        team[..., MAX_HP] = SPECIES_MAX_HP[team[..., SPECIES]]
        battle = np.zeros((n, 102), np.uint8)
        battle[:] = rows["battle"]
        # Legal actions and targets in the 6v6 action space (0-3 move slots, 4 Struggle, 11 pass).
        # Row choice codes (engine/src/dataset.rs): 0-3 a move slot, 4 Struggle, 254 no choice, 255 unused.
        code_to_action = np.full(256, -1)
        code_to_action[[0, 1, 2, 3, 4, 254]] = [0, 1, 2, 3, 4, 11]
        ch = code_to_action[rows["choices"]].reshape(n, 2, 4)
        mask = np.zeros((n, 2, 12), bool)
        mix = np.zeros((n, 2, 12), np.float32)
        r = np.arange(n)
        for s in (0, 1):
            for k in range(4):
                a = ch[:, s, k]
                ok = a >= 0
                mask[r[ok], s, a[ok]] = True
                mix[r[ok], s, a[ok]] += rows["mix"][ok, 4 * s + k]
        mix /= np.maximum(mix.sum(-1, keepdims=True), 1e-9)
        q = rows["q"].reshape(n, 4, 4).astype(np.float32)                    # side 0's value, 0..1
        return {"team": team, "battle": battle, "mask": mask, "value": (2 * rows["value"] - 1).astype(np.float32),
                "mix": mix, "q": q, "actions": ch, "kind": rows["kind"].copy()}
