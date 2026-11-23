# 1v1 dataset for the 6v6 agent (design)

## Purpose

The 6v6 agent (see the plan in `README.md`) learns by self-play, but it starts from, and is checked
against, exact 1v1 results. The dataset serves four uses:

1. **Matchup features.** At every turn the agent is told how its active Pokémon fares against the
   opposing active Pokémon and each revealed one, in their *current* condition, if neither switches.
2. **Endgame labels.** When each side has one Pokémon left, a 6v6 battle *is* a 1v1 game. Those
   positions get exact values instead of noisy self-play outcomes (the chess tablebase idea).
3. **Move priors.** For each position, the value of every pair of move choices and the optimal mix.
   This is the "value of moves in a matchup", and the starting policy before self-play.
4. **Set priors.** Which movesets each Pokémon realistically carries, so the agent can reason about
   moves it hasn't seen yet.

It isn't used directly. A small network (the **1v1 evaluator**) is trained on it, so the agent
can query any position, including movesets and conditions the dataset never contained, in
microseconds.

## Rules

Smogon RBY OU as it applies inside a 6v6 battle (`Rules::smogon_6v6`, to be added):

- **Battle setup:** level 100, all DVs 15, maximum stat experience.
- **Banned:** Mew and Mewtwo, plus the Evasion Clause (no Double Team or Minimize) and the OHKO
  Clause (no Fissure, Horn Drill or Guillotine).
- **Explosion and Self-Destruct are allowed.** They are central to real play. A double KO is a
  separate outcome, so the label records it on its own instead of folding it into "win" or "lose".
- **Sleep and Freeze Clauses depend on the rest of the team**, so they are inputs to each
  position rather than fixed rules:
  - `sleep_blocked[side]`: the opponent of `side` already has a Pokémon asleep from a move, so
    `side`'s sleep moves fail. This matches Showdown's Sleep Clause Mod.
  - `freeze_blocked[side]`: the same for freezing.

## Movesets

The pool comes from the September 2026 Smogon usage statistics for Gen 1 OU, at the 1760+ rating
cutoff (`data/smogon/gen1ou-1760-2026-09.json`, 11,702 battles).

- **Roster:** every species with at least 0.5% usage, which is the bulk of real play. Other legal
  species get one set each, at low weight, so the evaluator still sees them.
- **Sets:** the usage file gives each move's frequency but not full movesets. For each species, take
  the K most likely four-move sets (K = 3–5 by usage) from the move frequencies, and keep only sets
  the learnset allows. Nothing comes from the round-1 search, which used the video's rules at
  level 50 and allowed Double Team.
  Assuming moves are chosen independently is an approximation, so sets are checked against
  Smogon's published analyses for the top 15 species.
- **Size:** about 40 species × ~3.5 sets = 140 sets, giving about 10,000 ordered set pairs.

## Positions

A 6v6 battle reaches 1v1 positions in two ways, and both need coverage.

1. **Fresh entry.** A Pokémon has just switched in, or the turn opens after a KO. Volatile state is
   clean: no stat stages, confusion, Substitute, Leech Seed or Reflect. HP and major status carry
   over. In Gen 1, switching out turns Toxic into ordinary poison, and the paralysis and burn stat
   drops are applied again on entry (`LoadBattleMonFromParty` calls
   `ApplyBurnAndParalysisPenaltiesToPlayer`; Showdown's `setSpecies` does the same).
   - Each side's HP takes 5 levels (100, 75, 50, 25, 10%), and its status one of 6 values (none,
     PAR, BRN, PSN, sleep with 1–2 turns left, sleep with 3+ turns left). FRZ is added when the
     attacker can freeze.
   - Combined with the clause flags, that is up to 5 × 5 × 6 × 6 × 4 = 3,600 roots per set pair.
   - All roots go into **one solver instance**: lower-HP roots mostly reach states the
     full-HP root already explores, so the extra cost is small.
2. **Mid-battle.** Positions inside an exchange (+2 Special, a Substitute up, the second turn of
   Thrash). These come from sampling states in proportion to how often equilibrium play visits
   them from the fresh-entry roots: about 500 per set pair and flag combination.

## Labels per position

| Field | Meaning |
|---|---|
| `p_win`, `p_lose`, `p_double_ko` | Outcome probabilities under equilibrium play (side 0's view) |
| `value` | `p_win + 0.5 · p_double_ko`: the solved game value |
| `q[i][j]` | Value if side 0 picks move i and side 1 picks move j this turn, with equilibrium play afterwards (up to 4×4) |
| `mix0`, `mix1` | Each side's optimal mixed strategy this turn |
| `hp_left` | Expected HP fraction of the survivor (chip damage matters in 6v6) |
| `turns` | Expected battle length |
| `precision` | The solver setting used (exact, or roll buckets plus HP grid) |

`p_win`, `p_double_ko`, `hp_left` and `turns` need one extra pass after solving: a linear
evaluation of the equilibrium strategies (value iteration with the strategies fixed). That costs
about as much as the solve's own value iteration, which profiling puts at a few percent of the
total.

**The double-KO value.** A 6v6 player values a double KO differently depending on the teams left. The
equilibrium is computed at 0.5. For a sample of set pairs it is also computed at 0.3 and 0.7, to
measure how much the equilibrium moves.

## Inputs per position

Per side: species, the four moves, HP fraction, status (with sleep counter), stat stages, the
volatile flags the engine tracks (confusion, Substitute HP, Reflect and Light Screen, Leech Seed,
trapping, recharge, charging, Disable, Rage, Bide) and the clause flags. These are exactly the
engine's `Side` fields after canonicalisation, so the input is lossless.

## Precision and cost

- **Setting:** 3 damage-roll buckets and a 48-level HP grid, falling back like `meta::FINAL`.
  Every row records the setting it was solved at.
- **Cost estimate:**
  - Today's solver takes 0.2–0.7 s for typical games of this size at level 50, and about a minute
    for the largest stall games.
  - Roots and flag variants add about 4× (shared states make it less than 4 separate solves).
  - About 10,000 set pairs × ~10 s ≈ 30 core-hours, roughly 2–3 hours on 14 cores.
  - Level 100 costs about the same as level 50 once HP is on a grid. Measured on Tauros vs Starmie
    with 3 roll buckets and a 48-level grid: 26,671 states vs 26,213.
- **Size:**
  - Fresh-entry rows: 10,000 pairs × up to 3,600 roots ≈ 36M.
  - Mid-battle rows: 10,000 × 4 × 500 ≈ 20M.
  - About 100 bytes each, so ≈ 6 GB in total.

## Storage

Fixed-width little-endian binary records, sharded by set pair. A `schema.json` gives field
offsets, so Python reads the shards with `numpy.memmap`, without a parser. A manifest lists the sets
and their indices, and the solver version.

## Splits and validation

- **Held-out data:** 10% of sets and 3 entire species (one of them a top-10 species) are held out,
  to measure whether the evaluator generalises to movesets and species it has never seen.
- **Approximation error:** a calibration subset is solved exactly (as in `examples/calibrate.rs`),
  to measure the error of the approximate settings.
- **Real-play check:** compare against Smogon's "checks and counters" statistics (from real
  1760+ games), which say how often each Pokémon KOs or forces out each other one. The
  comparison is qualitative, because real play includes switching and our 1v1 values don't.

## Engine work needed first

1. A position builder for arbitrary fresh-entry roots (HP, status, sleep counter, Toxic → poison,
   entry stat penalties).
2. Clause flags that make sleep or freeze moves fail as Showdown's clause mods do, plus a
   Showdown spot-check.
3. Explosion allowed, with double KO as its own terminal outcome, and a configurable value for it.
4. Multi-root solving in one solver instance.
5. The equilibrium-evaluation pass (outcome split, survivor HP, battle length) and visitation
   sampling.
6. The shard writer, `schema.json`, and a numpy loader with a round-trip test.
