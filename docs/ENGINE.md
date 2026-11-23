# Engine and solver

## Battle engine (`engine/src/engine.rs`)

A port of the turn logic in the pret/pokered disassembly (`vendor/pokered/engine/battle/core.asm`
and `effects.asm`). It follows the cartridge routine by routine: `MainInBattleLoop` →
`ExecutePlayerMove` → `CheckPlayerStatusConditions` → `CalculateDamage` / `AdjustDamageForMoveType`
/ `RandomizeDamage` / `MoveHitTest` → move effects → `HandlePoisonBurnLeechSeed`.

Every random event goes through a `Chance` (`engine/src/chance.rs`) with the cartridge's exact odds
on its 0–255 scale. The same code can therefore sample one battle or enumerate every outcome with
its exact probability.

### Cartridge behaviour reproduced (selection)

- Accuracy is `rand < acc` on a 0–255 scale, so "100%" moves hit 255/256 of the time. Accuracy and
  evasion stages scale it in two floored steps.
- Critical hit rate is base Speed / 512, ×8 for Slash, Karate Chop, Razor Leaf and Crabhammer.
  Focus Energy *quarters* it. Crits double the level and use unmodified stats, so they ignore
  stat stages, burn, Reflect and Light Screen.
- Damage: `((2L/5+2)·P·A/D)/50`, capped at 997, plus 2. When either stat exceeds 255, both are
  divided by 4 and truncated to a byte, which is why Reflect can backfire on very high Defense.
  STAB, then the type chart applied in cartridge table order, then a random 217–255 / 255 factor.
- Raising any stat re-applies the paralysis Speed drop and burn Attack drop to the *opponent*.
  Lowering a stat does the same to the target.
- Paralysis: 63/256 chance of full paralysis.
- Sleep: 1–7 turns, and the turn the Pokémon wakes up is lost.
- Freeze: permanent unless hit by a Fire move that can burn.
- Confusion self-hits use the opponent's Reflect and can hit a Substitute.
- Trapping moves (Wrap, Bind, ...) stop the target from acting at all, and keep hitting for the
  first hit's damage. Wrap on a Ghost still traps when its accuracy roll succeeds.
- Hyper Beam needs no recharge if it KOs, misses or breaks a Substitute.
- Thunder Wave, sleep moves and Leech Seed go through Substitute.
- A Substitute at exactly 1/4 HP faints the user.
- Recover and Softboiled fail when missing HP is 255 or 511 (they compare only the low byte).
- Counter doubles the last damage dealt by *either* side, using the opponent's last selected move
  (stale while it sleeps or recharges). It has −1 priority; Quick Attack has +1.
- Rest: 2 turns of sleep. It does not remove Toxic's escalation or the stat drops from
  paralysis and burn.
- Leech Seed's drain uses and advances the Toxic counter.
- Thrash and Rage reuse move data left in RAM, including accuracy already scaled by
  accuracy/evasion stages, which gets scaled again each turn.
- Haze resets both sides' stats and cures only the target's status. A target cured of sleep or
  freeze loses its move that turn.
- One-hit KO moves fail against faster targets.

### Departures from the cartridge

- **PP is not tracked in 1v1 solving.** Battles use maximum PP. That is harmless for about 95% of
  set pairs (every fresh-entry root ends in under 64 expected turns, before any healing move could
  run out), but not for recovery stalls: in about 5% of pairs some roots run long, and about 0.1%
  of roots are labelled endless (scored as a draw) where in the real game healing PP runs out and
  someone wins (Chansey vs Starmie, the Starmie mirror). The 168 pairs the dataset run could not
  finish in time are such stalls. The `smogon-pp` ruleset (`Rules::heal_pp`) fixes this
  approximately; see below. Transform's 5-PP copies are also not modelled.
  The 6v6 layer does track PP; see below.
- **Healing PP (`smogon-pp`).** Tracking every move's PP exactly would multiply a stall
  matchup's state count by 300-1,000x. Instead only the healing move (Recover, Softboiled or
  Rest; no ladder set has two) is tracked, as a level in bits 3-7 of `Side::clauses`, so the
  dataset format is unchanged. Level L stands for 0, 1, 2, 4, 8, 16 (, 32) uses. Each use drops a
  level with probability 1/(gap to the next level down), which keeps the expected number of uses
  left exact, as the HP grid does for damage; at level 0 the move can't be selected. The cost of
  the approximation is variance: a healer can run dry after as few as 5-6 uses, though rarely. So
  the dataset uses this ruleset only for pairs that stall (over 64 expected turns or endless
  games) under plain `smogon`, where the alternative is an "endless" label. Measured on Chansey
  (Softboiled/Thunder Wave/Sing/Seismic Toss) vs Starmie (Recover/Thunder Wave/Thunderbolt/Surf):
  endless entry roots 8% -> 0%, mean expected length 680 -> 17 turns, states 158K -> 1.55M (the
  48-level HP grid overflowed, so it solved at 24 levels), 33 s -> 169 s, 2.5 GB.
- **Metronome, Mirror Move and Mimic are not implemented**, and are excluded from movesets.
- The cartridge has small player/enemy asymmetries (the source of link-battle desyncs). Both
  sides use the player-side routine, as each Game Boy does for its own Pokémon.
- A defense stat of 0 after scaling freezes the real game. It is treated as 1 (unreachable at
  level 50).
- The start-of-turn flinch reset follows the player-side rule symmetrically.

### Validation

- `engine/tests/mechanics.rs`: exact outcome probabilities for specific quirks, checked against
  values derived by hand from the disassembly.
- Exact-vs-sampled agreement: the solver's exact value under "both sides pick uniformly random
  moves" matches 1,000,000 sampled battles within sampling error (0.748575 vs 0.748384 for
  Tauros/Chansey; 0.757783 vs 0.758059 for Gengar/Arbok).
- Differential test against Pokémon Showdown's Gen 1 simulator (`tools/diffcheck.py`,
  `pokemon-showdown` 0.11.9). Both simulators played 160 random matchups (fully evolved Pokémon,
  random legal movesets, uniformly random move choices), 3,000 battles each in Showdown and
  100,000 in this engine. The mean absolute difference in win rate was 0.87 points, and 150 of
  160 agree within 3σ. The 10 outliers all involve Rage, Bide or Transform:
  - **Rage + confusion** (largest gap): on the cartridge, a confusion self-hit writes power 40 and
    Normal type into the move-data RAM and restores only the effect byte. Rage never reloads move
    data, so a raging Pokémon's later attacks hit at 40 power. This engine reproduces it and
    Showdown does not. Switching the behaviour off makes the isolated case agree exactly (0.450 vs
    Showdown 0.452), so the gap is entirely this quirk.
  - **Transform**: Showdown gives Transform's copied moves 5 PP; this engine doesn't track PP.
  - **Bide** against Toxic or Rest: gaps of 2–3 points remain in isolated tests (e.g. Snorlax
    with only Bide vs Golem with Tackle and Toxic: 0.110 vs 0.081). Plain Bide, Bide vs Bide,
    Bide vs Substitute and Bide vs sleep agree. Not yet attributed. Both simulators skip the
    shared damage reset for the same Residual-1 moves.

## 6v6 battles (`engine/src/battle6.rs`)

The 6v6 layer wraps the turn engine with teams, switching, fainting and replacements, following
core.asm:
- **Switching:** a switch uses up the side's turn, but the side still takes its place in the
  Speed order and its residual damage. Sending a Pokémon out clears its volatile status and stat
  stages, applies the paralysis and burn stat drops, and ends the opponent's Wrap.
- **Fainting:** a faint ends the turn at once.
- **PP:** spent where the cartridge calls `DecrementPP`.
- **Clauses:** Sleep and Freeze Clauses follow Showdown's clause mods.

Validation (`tools/diffcheck6.py`): Both simulators played 60 pairs of random six-Pokémon teams
from the ladder set pool under the same random policy, in Showdown's `gen1ou`. Each pair was
played 2,000 times in Showdown and 20,000 times here:
- **Win rates:** mean absolute difference 0.87 points; none beyond 3σ, 2 beyond 2σ (about 2.8
  expected by chance).
- **Battle length:** 60.0 vs 60.1 turns.

Before PP was tracked, the win rates already agreed, but battles ran 13% longer (23% longer for
the longest third of battles), because recovery moves never ran out.

## Solver (`engine/src/solver.rs`)

A 1v1 battle is a two-player zero-sum stochastic game with simultaneous moves. Each turn both
sides pick a move without seeing the other's choice. The value of a state is P(side 0 wins) +
½·P(draw) under optimal play by both.

1. **Enumerate** every state reachable from the start. Each state's turn is a matrix game: rows
   are side 0's moves, columns side 1's, and each cell is the exact distribution over next states
   (from `Enumerator`).
2. **Decompose** the state graph into strongly connected components (Tarjan) and solve them in
   reverse topological order. Acyclic parts of the game take one pass. Components with cycles
   (healing, misses, sleep, stat stages) use Gauss–Seidel value iteration to 1e-12.
3. **Solve each matrix game** exactly with the simplex method (`engine/src/nash.rs`). A pure
   saddle point is detected first, as a fast path.

A cycle that can never be left counts as a draw (0.5).

**Strategies in games that can cycle.** Value iteration finds the correct values, but the mixed
strategies read off the converged matrix games don't always achieve them. In a winning position the
matrix game is often indifferent between making progress and stalling (healing keeps the value
where it is), and a strategy that always stalls never cashes in. An example is Chansey (Softboiled,
Thunder Wave, Ice Beam, Sing) vs Starmie (Recover, Thunder Wave, Psychic, Blizzard):
- **The value:** 0.431.
- **Playing the per-turn mixes:** Chansey wins 9% of the time and the battle never ends 34% of
  the time, which is worth 0.26.

This is a known property of recursive games. `Solver::discount` fixes it: a per-turn pull of
10⁻⁷ towards the endless-battle value makes the side that is ahead strictly prefer winning sooner.
With it, playing the per-turn mixes achieves the undiscounted value to within 10⁻⁴ in every state
(`tests/eqstats.rs`), and the values move by about 10⁻⁷ times the battle length. The 1v1 tables
use no discount (their values are unaffected). The dataset and anything that plays out strategies
use it.

**Assumption:** both players see the full state, including sleep counters and the exact HP
numbers, and know each other's movesets. The only hidden information is the simultaneous move
choice.

### State canonicalisation

Fields that can no longer affect the battle are zeroed after each turn (`Engine::canonicalize`),
so equivalent states merge. For example, the last damage dealt is kept only while a Counter could
still read it, and a stale move selection only while it affects priority or a charging move. This
is lossless: values are unchanged, only the state count drops.

### Approximation ladder

Exact solving costs from milliseconds to tens of minutes per matchup. Stall matchups (Rest, Double
Team, sleep, Counter) can reach tens of millions of states. Two controlled approximations are
available:

- **Damage-roll buckets**: the 39 random rolls are grouped into k equally likely buckets, each
  represented by its middle roll.
- **HP grid**: after each exactly-computed turn, each side's HP is rounded stochastically to the
  two nearest of N evenly spaced levels, preserving expected HP. Gen 1's HP bar is 48 pixels.

`meta::FINAL` tries exact first and falls back step by step. Every result records the precision
it was solved at.

The largest games come from both sides stacking evasion while healing. For example, Ninetales (Fire
Blast, Rest, Double Team, Confuse Ray) vs Alakazam (Psychic, Thunder Wave, Rest, Double Team)
reaches every combination of both evasion stages and Ninetales' Special drops (7³ = 343). That
gives about 1M distinct non-HP states and 8.9M states even on a 3-level HP grid. These states are
genuinely different, so canonicalisation can't merge them. The cycles stay small (the largest
strongly connected component has 969 states), so such a game solves in about 2 minutes on one
core. It needs about 500 bytes per state, though. `run_final.sh` therefore retries the matchups
`FINAL` gives up on with `meta::BIG` (16–24M-state caps), two at a time.

## 2v2 endgame solver (`engine/src/endgame.rs`)

The same exact method as the 1v1 solver, for battles where each side has two Pokémon (one active,
one benched or already fainted) and switching is an action. `meta endgame "A1|A2" "B1|B2"
--rules smogon [--precision B,L[,BL];...] [--hp a1,a2/b1,b2]` prints the value (side 0's win
probability plus half the double-KO probability), the state count, the time and the opening mixed
strategies.

**State.** The active `Battle` (both active Pokémon's full volatile state), which member is
active on each side, and each benched member's lasting state: HP, major status with the sleep
counter, whether its sleep came from the opponent (Sleep Clause), and its healing-PP level under
`smogon-pp`. Like the 1v1 solver it ignores PP otherwise.

**Semantics** are `battle6`'s, built from the same functions (`battle6::send_out_side`,
`clause_flags`, `survivor_after_faint`, `can_switch`):
- A switch is allowed unless the side is recharging, raging, thrashing or charging (also while
  trapped). It uses the side's turn; the incoming Pokémon keeps the outgoing one's stale move
  selection, which still sets Quick Attack/Counter priority.
- Sleep Clause and Freeze Clause are recomputed from both teams before every turn.
- A faint ends the turn. With one Pokémon left the replacement is forced, so it is applied inside
  the transition; no state waits for a replacement. A double KO with both benches alive sends out
  both; a side with nothing left loses, and both at once is a double KO (`draw_value`, 0.5).
  Explosion and Selfdestruct are ordinary moves here, as they are in the 1v1 tables.
- Canonicalisation uses `Engine::set_teams`, which judges Counter/Bide/two-turn-move staleness
  over both whole teams and keeps what switching can expose (a stale Quick Attack or Counter
  selection, `SLEPT_BY_FOE` while asleep).

**Checks** (`engine/tests/endgame.rs`): with both benches fainted the state graph and value equal
the 1v1 solver's exactly (5,044 states, 0.460830 for Tauros vs Starmie at 3 buckets/24 levels),
also after a forced replacement at the root or mid-battle (survivor keeping its volatile state). A
mirror is worth 0.5. A useless bench (Magikarp with Splash) never lowers its side's value. Random
6v6 battles with two Pokémon a side (sleep, Wrap, Counter, Explosion, Quick Attack, Leech Seed,
Toxic, freeze, Bide, Dig, Substitute, Reflect): 2,455 turns with 801 switches and 267 forced
replacements give the same legal actions as `Battle6`, the same next state from the same random
numbers, and a next state among the solver's exact successors. The turn cache, and a bench grid
equal to the HP grid, change nothing.

Making this work also fixed a `battle6` bug: Rest after waking from an opponent's sleep move still
counted as "slept by the foe" for Sleep Clause. Rest now clears the flag.

**Precision.** Damage-roll buckets and the HP grid work as in the 1v1 solver, with a grid per
team member. A third setting rounds benched Pokémon's HP to a coarser grid (in practice, when they
switch out). The state count is roughly 4 matchups × the 1v1 state count × (bench configurations
per side)², where a bench configuration is an HP level times a status. So the 2v2 game is
hundreds to thousands of times larger than each 1v1 game at the same precision. For example,
Tauros/Starmie at 1 bucket and 6 levels has 553 states in 1v1, but Tauros+Starmie vs
Starmie+Tauros has 961,164.

**Cost.** Two things make it affordable:
- **Turn cache.** States that differ only in their benches share their turn outcomes, cached by
  pre-turn position. On Tauros+Starmie vs Starmie+Tauros at (1, 3) this cuts the paths
  enumerated 43× (10.5M to 243k).
- **Faster value iteration.** Switching back and forth creates cycles whose matrix games stay
  mixed. There, Gauss–Seidel stalls at the simplex round-off (residual stuck at ~2×10⁻¹², just
  above the 10⁻¹² tolerance, for 100,000 sweeps). So iteration also stops once the residual is
  below 10⁻¹⁰ and has stopped shrinking, and only states whose successors moved are re-evaluated.
  The same game went from 8.9 s to 1.2 s with unchanged values.

Most of the remaining time is value iteration on cycles: stalling with Recover, Softboiled and
sleep, and switch loops (the largest component reached 3,028 states). Memory is about 0.5–1 KB per
state. Each solve runs on one core, like the 1v1 solver.

Measured on an M-series Mac (one core per solve, two solves at a time), main ladder sets
(`data/sets_ladder.json`), `--rules smogon`, 4M-state cap. Precision is (roll buckets, HP levels,
bench HP levels; 0 = same as active).

| Teams (side 0 vs side 1) | Precision | Value | States | Time | Peak RSS |
|---|---|---|---|---|---|
| Tauros+Starmie vs Starmie+Tauros | (1, 3) | 0.4807 | 129k | 1.2 s | 0.14 GB |
| | (1, 6, 3) | 0.4642 | 355k | 4.2 s | 0.3 GB |
| | (3, 6, 3) | 0.4601 | 375k | 5.1 s | 0.3 GB |
| | (1, 6) | 0.4633 | 961k | 13 s | 0.6 GB |
| | (1, 12, 3) | 0.4652 | 1.07M | 20 s | 0.9 GB |
| | (1, 12, 4) | 0.4619 | 1.51M | 19 s | 1.1 GB |
| | (1, 12, 6) | 0.4643 | 2.94M | 56 s | 1.3 GB |
| Tauros+Starmie vs Snorlax+Chansey | (1, 3) | 0.4301 | 1.17M | 77 s | 0.6 GB |
| | (1, 6, 3) | 0.4186 | 3.06M | 366 s | 1.3 GB |
| | (1, 6), (1, 12, 4) | over 4M states (after 19 and 14 min) | | | |
| Tauros+Chansey vs Snorlax+Starmie | (1, 3) | 0.5035 | 1.01M | 80 s | 0.6 GB |
| | (1, 6, 3) | 0.4988 | 2.70M | 313 s | 1.4 GB |
| Snorlax+Rhydon vs Tauros+Zapdos | (1, 3) | 0.6612 | 735k | 28 s | 0.7 GB |
| | (1, 6, 3) | 0.6652 | 1.92M | 50 s | 1.7 GB |
| | (1, 6), (1, 12, 4) | over 4M states | | | |
| Alakazam+Tauros vs Starmie+Snorlax | (1, 3) | 0.5468 | 894k | 57 s | 0.6 GB |
| | (1, 6, 3) | 0.4863 | 2.30M | 57 s | 2.2 GB |
| | (1, 6), (1, 12, 4) | over 4M states | | | |
| Zapdos+Rhydon vs Tauros+Exeggutor | (1, 2) | 0.2166 | 2.70M | 41 s | 1.2 GB |
| | (1, 3) and finer | over 4M states | | | |
| Exeggutor+Snorlax vs Chansey+Tauros | (1, 2) and finer | over 4M states (in under a minute) | | | |

**Reading the table:**
- The precision effect is larger than in 1v1. Between (1, 3) and (1, 6, 3) values move by 0.004
  to 0.06 (Alakazam+Tauros: 0.547 to 0.486). Only the Tauros/Starmie game can be followed to
  (1, 12, 6), where the finer settings agree to about ±0.003.
- The bench grid is the cheapest lever. At 12 active levels, rounding benches to 3–4 levels moves
  the value by ≤0.003 while using a third to half the states.
- The state count is set by the variety of statuses, not just HP. Sleep counters (1–7) on both
  active and benched Pokémon on both sides (Sleep Powder/Sing teams), Agility stages and
  Substitute multiply it. The two sleep-and-Agility teams don't fit 4M states even at 2–3 HP
  levels.
- Roll buckets matter little at these grids: 3 buckets moves (1, 6, 3) by 0.004 for 5% more
  states.

The root strategies are informative. Snorlax facing Tauros mostly Body Slams but switches to
Rhydon 3–5% of the time, and Tauros answers by switching to Zapdos 91%. In the Tauros+Starmie
mirror (1, 3) and against Snorlax+Chansey at (1, 6, 3), Tauros opens with Blizzard, going for
the one freeze Freeze Clause allows.

`endgame::LADDER` (the CLI default) tries (1, 12, 6), (1, 12, 4), (1, 6), (1, 6, 3), (1, 3),
(1, 2), each capped at 4M states. A failed rung can take many minutes before it hits the cap.

**Limitations:**
- PP is ignored (beyond healing PP under `smogon-pp`).
- The clauses are evaluated at the start of the turn, as `battle6` does. So a sleep move used
  after a sleeping target woke up earlier in the same turn is still blocked (Showdown checks at
  the moment of use).
- Exact roll and HP solves are out of reach for realistic teams.
- Only two Pokémon per side. Three would multiply the state count by another (bench
  configurations)² and add a replacement choice.
