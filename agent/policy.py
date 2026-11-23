"""The 6v6 policy/value network.

Tokens: the twelve team members (own six, opponent's six; see agent/obs.py), the two active
Pokemon's volatile state, and a context token, each with a learned token-type embedding. A
transformer encoder mixes them.
- Move actions are scored from the own active Pokemon's token, the move itself and its PP left.
- Switch actions are scored from the target team member's token.
- Struggle and pass are scored from the context token.
- The value head predicts the game result from the side's perspective (-1 loss .. +1 win).

With matchups=True the network also gets the solver's exact 1v1 values (agent/matchups.py): each
own token its row of the 6x6 matchup matrix, each opposing token its column, and each move the
equilibrium probability of playing it in the active 1v1. The Matchups object that computes them
is attached as policy.mu (it is data, not a parameter).

With direct=True (v3 on) the action heads also read the opposing active Pokemon directly: each move
is scored from the own active's token and volatile state, the opposing active's token and
volatile state, the move and the per-move features; each switch from the context, the target's
token, the opposing active's token and (with matchups) the target's exact 1v1 value, stall flag
and known flag against the opposing active. Without it the heads only see the opponent through
attention. `arch()` returns the constructor arguments, which checkpoints store.

v4 heads (forward(..., extras=True) returns them in a dict):
- aux=True: "alive", a logit per team token (own six, then the opponent's six) for that Pokemon
  still standing when the battle ends: Go's ownership target, for Pokemon. A binary result is a
  noisy target; which Pokemon survive says much more about why.
- qhead=True: "q", a 12x12 payoff matrix Q[a, b] (-1..1, this side's view) for this side playing
  action a and the opponent action b this turn, with the battle continuing from there. Each side's
  actions are embedded from the same tokens the policy heads use, so the opponent's actions need
  its moves: Q is meant for observations where the opposing team is known (privileged training
  passes, or a guess of the hidden sets at play time). agent/qsolve.py turns Q into an
  equilibrium mix.
- skill=True: the player's level (unrated / <1300 / 1300-1499 / 1500+ / Smogon tournament) as an
  embedding added to the context token, learned from human games; `play_skill` (default tournament)
  is the level asked for when the observation has none.
- senses=True: the multi-Pokemon quantities of agent/senses.py (sweep and threat chains,
  predictability, concealment, sack gain), per token and for the context token. The Senses object
  is attached as policy.sense.
"""
import numpy as np
import torch
from torch import nn

from agent.features import N_MOVES, N_SPECIES, SCALARS
from agent.obs import PRIOR, TOKEN_SCALARS

N_ACTIONS = 12
STRUGGLE, PASS = 4, 11
# skill=True: the player's level as an input (human data, agent/train_bc.py), so play can ask for a level.
N_SKILL = 5
SKILL_UNRATED, SKILL_LOW, SKILL_MID, SKILL_HIGH, SKILL_TOUR = range(N_SKILL)   # <1300, 1300-1499, 1500+, Smogon tournament


def skill_bucket(rating, tour):
    rating, tour = np.asarray(rating), np.asarray(tour)
    out = np.where(rating >= 1500, SKILL_HIGH, np.where(rating >= 1300, SKILL_MID, np.where(rating > 0, SKILL_LOW, SKILL_UNRATED)))
    return np.where(tour, SKILL_TOUR, out)


def mlp(*dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class Policy(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, matchups=False, direct=False, aux=False, qhead=False, senses=False, skill=False,
                 opp_pp=False):
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads, "matchups": matchups, "direct": direct, "aux": aux, "qhead": qhead,
                       "senses": senses, "skill": skill, "opp_pp": opp_pp}
        self.aux, self.qhead, self.senses = aux, qhead, senses
        self.opp_pp = opp_pp            # observe the opponent's PP left on its revealed moves (agent/obs.py)
        self.sense = None
        self.matchups, self.direct = matchups, direct
        self.mu = None
        self.species = nn.Embedding(N_SPECIES + 2, d)       # 0 empty, 1..151, 152 unknown
        self.moves = nn.Linear(N_MOVES, d, bias=False)       # multi-hot / prior over moves
        self.team_scalars = mlp(TOKEN_SCALARS + (18 if matchups else 0), d, d)
        if senses:
            self.sense_tokens = mlp(4, d, d)
            self.sense_global = mlp(4, d, d)
            for m in (self.sense_tokens[-1], self.sense_global[-1]):   # start as a no-op on a warm-started network
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.volatile = mlp(SCALARS, d, d)
        self.kind = nn.Embedding(5, d)                       # own mon, opp mon, own active, opp active, context
        self.context = nn.Parameter(torch.zeros(d))
        self.skill = skill
        if skill:
            self.skill_emb = nn.Embedding(N_SKILL, d)            # added to the context token; starts as a no-op
            nn.init.zeros_(self.skill_emb.weight)
            self.play_skill = SKILL_TOUR                         # used when the observation has no "skill"
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)   # unused with norm_first anyway
        self.norm = nn.LayerNorm(d)
        self.move_emb = nn.Embedding(N_MOVES, d)
        self.move_head = mlp((5 if direct else 3) * d + (3 if matchups else 1), d, 1)
        self.switch_head = mlp((3 if direct else 2) * d + (3 if direct and matchups else 0), d, 1)
        self.other_head = mlp(d, d, 2)                       # Struggle, pass
        self.value_head = mlp(d, d, 1)
        if aux:
            self.alive_head = mlp(d, d, 1)
        if qhead:
            k = d // 2
            self.q_move = mlp(4 * d, d, k)                   # own active, own volatile, foe active, move
            self.q_switch = mlp(3 * d, d, k)                 # context, target, foe active
            self.q_other = nn.Parameter(torch.zeros(2, k))   # Struggle, pass
            self.q_other_ctx = nn.Linear(d, k)
            self.q_bil = nn.Parameter(torch.randn(k, k) / k)
            self.q_row, self.q_col = nn.Linear(k, 1), nn.Linear(k, 1)
        self.register_buffer("prior", torch.from_numpy(PRIOR), persistent=False)

    def move_vectors(self, o):
        """Per token, known moves one-hot plus (for a seen opponent) the usage prior over the moves
        not yet revealed, scaled to the number of unknown slots: (B, 12, N_MOVES)."""
        hot = torch.zeros(*o["move_ids"].shape[:2], N_MOVES, device=o["move_ids"].device)
        hot.scatter_add_(-1, o["move_ids"], o["move_known"])
        hot[..., 0] = 0
        pri = self.prior[o["prior_species"]] * (1 - hot)
        left = 4 - o["move_known"].sum(-1, keepdim=True)
        pri = pri / pri.sum(-1, keepdim=True).clamp(min=1e-6) * left
        return hot + pri * o["use_prior"][..., None]

    def action_embeddings(self, h, o, me):
        """(B, 12, k) embeddings of side `me`'s actions (0 own, 1 opponent) for the Q head."""
        B = h.shape[0]
        r = torch.arange(B, device=h.device)
        base, foe = 6 * me, 6 * (1 - me)
        act_idx = o["active"] if me == 0 else o["opp_active"]
        foe_idx = o["opp_active"] if me == 0 else o["active"]
        act, foe_act = h[r, base + act_idx], h[r, foe + foe_idx]
        vol, c = h[:, 12 + me], h[:, 14]
        moves = o["move_ids"][r, base + act_idx]                              # (B, 4); 0 = unknown/empty
        mv = self.move_emb(moves)
        e_move = self.q_move(torch.cat([act[:, None].expand(-1, 4, -1), vol[:, None].expand(-1, 4, -1),
                                        foe_act[:, None].expand(-1, 4, -1), mv], -1))
        team = h[:, base:base + 6]
        e_sw = self.q_switch(torch.cat([c[:, None].expand(-1, 6, -1), team, foe_act[:, None].expand(-1, 6, -1)], -1))
        oth = self.q_other[None] + self.q_other_ctx(c)[:, None]
        return torch.cat([e_move, oth[:, :1], e_sw, oth[:, 1:]], 1)

    def q_matrix(self, h, o):
        ea, eb = self.action_embeddings(h, o, 0), self.action_embeddings(h, o, 1)
        q = torch.einsum("bik,kl,bjl->bij", ea, self.q_bil, eb) + self.q_row(ea) + self.q_col(eb).transpose(1, 2)
        return torch.tanh(q)

    def forward(self, o, mask, extras=False):
        B = o["species"].shape[0]
        scalars = o["scalars"]
        if self.matchups:
            m = [o["mu_value"], o["mu_stall"], o["mu_known"]]
            own = torch.cat(m, -1)                                          # row i: own i vs each opponent
            opp = torch.cat([x.transpose(1, 2) for x in m], -1)            # row j: each own vs opponent j
            scalars = torch.cat([scalars, torch.cat([own, opp], 1)], -1)
        team = self.species(o["species"]) + self.moves(self.move_vectors(o)) + self.team_scalars(scalars)
        if self.senses:
            team = team + self.sense_tokens(torch.cat([o["own_senses"], o["opp_senses"]], 1))
        kinds = torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long), torch.tensor([2, 3, 4])]).to(team.device)
        vol = self.volatile(o["volatile"])                                   # (B, 2, d)
        ctx = self.context.expand(B, 1, -1)
        if self.senses:
            ctx = ctx + self.sense_global(o["global_senses"])[:, None]
        if self.skill:
            sk = o["skill"] if "skill" in o else torch.full((B,), self.play_skill, dtype=torch.long, device=ctx.device)
            ctx = ctx + self.skill_emb(sk)[:, None]
        x = torch.cat([team, vol, ctx], 1) + self.kind(kinds)
        empty = torch.cat([o["species"] == 0, torch.zeros(B, 3, dtype=torch.bool, device=team.device)], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=empty))
        own, own_vol, opp_vol, c = h[:, :6], h[:, 12], h[:, 13], h[:, 14]
        r = torch.arange(B, device=h.device)
        act = own[r, o["active"]]                                            # own active Pokemon's token
        mv = self.move_emb(o["act_moves"])                                   # (B, 4, d)
        per_move = [o["act_pp"][..., None]]
        if self.matchups:
            per_move += [o["mu_mix"][..., None], o["mu_mix_known"][:, None, None].expand(-1, 4, 1)]
        ctx_move = [act, own_vol]
        ctx_switch = [c]
        if self.direct:
            opp_act = h[r, 6 + o["opp_active"]]                              # opposing active Pokemon's token
            ctx_move += [opp_act, opp_vol]
            ctx_switch += [opp_act]
        move_logits = self.move_head(torch.cat([x[:, None].expand(-1, 4, -1) for x in ctx_move] + [mv, *per_move], -1)).squeeze(-1)
        per_switch = [x[:, None].expand(-1, 6, -1) for x in ctx_switch] + [own]
        if self.direct and self.matchups:
            vs_active = lambda k: o[k][r, :, o["opp_active"]][..., None]     # (B, 6, 1): each own Pokemon vs the opposing active
            per_switch += [vs_active("mu_value"), vs_active("mu_stall"), vs_active("mu_known")]
        switch_logits = self.switch_head(torch.cat(per_switch, -1)).squeeze(-1)
        other = self.other_head(c)
        logits = torch.cat([move_logits, other[:, :1], switch_logits, other[:, 1:]], -1)  # 4 + 1 + 6 + 1
        logits = logits.masked_fill(~mask, float("-inf"))
        value = torch.tanh(self.value_head(c).squeeze(-1))
        if not extras:
            return logits, value
        ex = {}
        if self.aux:
            ex["alive"] = self.alive_head(h[:, :12]).squeeze(-1)
        if self.qhead:
            ex["q"] = self.q_matrix(h, o)
        return logits, value, ex
