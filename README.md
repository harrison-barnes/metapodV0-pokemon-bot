# MetapodV0: a Gen 1 OU Pokémon Showdown bot

MetapodV0 plays Generation 1 OU singles (Red/Blue/Yellow rules, Smogon RBY OU as played on Pokémon Showdown). It has
three parts:

- A small transformer policy, trained by self-play PPO, in a Gen 1 battle engine written in Rust from the
  [pret/pokered](https://github.com/pret/pokered) disassembly.
- A larger *privileged* critic that scores positions.
- At play time, an exact one-turn search over guessed opposing sets.

It played the public Gen 1 OU ladder as the account `MetapodV0` from 2026-11-09 to 2026-11-12.

> Video: _link coming soon_ <!-- VIDEO_LINK -->

This repository holds everything needed to run the bot as it played: the checkpoint, the engine, the search, the
Showdown adapter and the two teams. It also holds the training and evaluation code that produced it. See
[What is not included](#what-is-not-included) for the inputs you have to fetch or rebuild yourself.

## Contents

- [Architecture](#architecture)
- [Results, with caveats](#results-with-caveats)
- [Install](#install)
- [Play](#play)
- [Train](#train)
- [Evaluate against Metamon](#evaluate-against-metamon)
- [Tests](#tests)
- [Hardware notes](#hardware-notes)
- [Repository layout](#repository-layout)
- [What is not included](#what-is-not-included)
- [Which code version this is](#which-code-version-this-is)
- [Credits](#credits)

## Architecture

**What MetapodV0 is, exactly:**
- Checkpoint `spec2@200`: `checkpoints/spec2_00200.pt` (policy) plus `checkpoints/spec2_00200_critic.pt` (critic).
- Exact one-turn search with tau 0.02, `--infer`, and K = 8 guessed opposing sets.
- Teams: `data/search_teams_v1`, one of the two drawn per battle.

### Network (policy and critic share the class)

**Input: 15 tokens.**
- Own team ×6 and opponent team ×6.
- The two active Pokémon's volatile state.
- One context token.
- A learned type embedding marks each token as own mon, opposing mon, own active, opposing active or context.

**Each team token is built from:**
- A species embedding: 151 species, plus "empty" and "unknown until sent out".
- A linear map of the move set. An opponent's unrevealed moves are filled with the ladder usage prior
  P(move | species) from `data/sets_ladder.json`.
- An MLP over 17 scalars: HP fraction, status, active/alive/seen, PP, and Sleep Clause.
- 18 *matchup* values: the exact 1v1 equilibrium value, a stall flag and a known flag against each of the six opposing
  Pokémon. These are looked up in a precomputed solver table, `data/matchup_table_v2.npz`.
- 4 "senses" features chained from the same table.

**Volatile tokens:** an MLP over 66 per-side features (stats, stages, battle-status bits, Substitute, traps, Bide,
clauses, last moves). Counters a player cannot see are zeroed: sleep, confusion, Disable and trap turns, and, for the
opponent, exact HP, Substitute HP and Bide damage.

**Encoder:** 4 transformer layers, d = 256, 8 heads (head dim 32), FFN 1024, pre-norm, no dropout; empty slots are
masked.

**Per-action heads over 12 actions** (4 moves, Struggle, 6 switches, pass):
- A move-logit MLP over the two active tokens, their volatile states, the move embedding, its PP and the 1v1
  equilibrium probability of that move.
- A switch-logit MLP over the context, the opposing active and the target with its exact 1v1 value against the
  opposing active.
- A tanh value head.

**Policy:** 4,236,549 parameters.

**Privileged critic:** 4,877,448 parameters. It has the same architecture plus an "alive" auxiliary head and a 12×12
payoff-matrix head Q[a, b] (bilinear, tanh). It is trained on the *full* state: every opposing species and move is
revealed, but hidden counters stay masked. It runs at play time as the search's leaf evaluator, so the bot uses
9,113,997 parameters per decision.

### Search (`agent/search_exact.py`)

One-turn determinized simultaneous-move search with exact chance:

1. **Guess the opposing team K = 8 times** (the set guesser in `agent/search.py`):
   - Revealed Pokémon keep their species and seen moves; the rest of each set is drawn from a usage-weighted posterior
     over the 156-set ladder pool (79 species).
   - Unrevealed slots are drawn from the pool without repeating species.
   - Species with no pool set get a fallback set: seen moves plus the strongest learnset moves.
   - Hidden counters are drawn per guess: sleep 1-7 (Rest 2), confusion 2-5, trap turns, and Substitute HP = 1/4 of
     max HP.
2. **`--infer`:** draw 4 × K guesses, weight each by how likely the opponent's last 12 choices are under our policy,
   and resample K.
3. **Expand every pair of candidate actions in every guess with the engine.** Each side gets 6 candidates (the top 6
   by policy prior). Damage rolls fall into 3 buckets, identical outcomes are merged, and at most 16 outcomes are kept
   per pair. Each outcome is scored by the critic, or by +1/-1/0 at game end.
4. **Solve the averaged 6×6 payoff matrix M as a KL-regularized zero-sum game:**
   max_x min_y x'My − tau·KL(x‖p) + tau·KL(y‖q), with tau = 0.02, where p and q are the policy priors.
   Our action is sampled from x.

About 59% of its decisions come out pure (one action with all the weight).

### Training (lineage summary; numbers from the project's records)

**spec2 is PPO from an earlier generalist, v7@300.** It ran 200 iterations of 2048 envs × 128 steps with these
settings:
- clip 0.2, gamma 0.995, GAE lambda 0.95;
- lr 3e-5 → 1.5e-5, entropy 0.003 → 0.002;
- 1 epoch, minibatch 4096, bf16.

**Our side brought the two searched teams in half of its battles.** In the rest, and on the opposing side always, the
teams were real players' teams.

**Opponents, beyond plain self-play:**
- 30% of envs play a league (v5d@400, v6, v7@300) with PFSP (exponent 1.5).
- 10% play two frozen human-imitation networks.

**Other losses and filters:**
- Advantage filtering at 0.75 (20% of low-advantage decisions kept).
- An exact-endgame tablebase: 31M solved 1v1 rows dressed as 6v6 endgames, every 4th update.
- Counterfactual Q targets for the critic's payoff head.
- A KL "magnet" to a slowly refreshed copy.

`scripts/train.sh` is that recipe. The lineage before v7 (v1-v6, about 3.9M self-play games through v4) is not
reproduced here step by step.

**The two teams** come from an evolutionary team search in the engine (`agent/team_search.py`). Their opponents were
real players' teams and a hall of fame of earlier candidates; none of the bots below or their teams were used.

## Results, with caveats

**Read this first.**
- Every number below except the ladder comes from bots on a local Showdown server. None of those games were against
  people.
- MetapodV0 always brings one of its two tuned teams; the Metamon bots draw a random team every game. **Part of every
  win rate below is the team, not the play.** The team search was the one improvement the project could show transfers
  to outside opponents.
- Win rate counts a tie as half. Intervals are Wilson 95%. Games inside one box share processes, so per-game
  intervals are somewhat too narrow; the box-to-box spread is the honest error bar (TaurosV0: .60-.71 by run).

### Against Metamon's Gen 1 OU bots: 62.3% over 26,700 games

These are the four Gen 1 OU bots of [Metamon](https://github.com/UT-Austin-RPL/metamon) (UT Austin), pooled over
24 evaluation boxes (2026-11-09 to 11-12):

| Opponent | Games | MetapodV0 score | 95% CI | Luck-adjusted |
|---|---:|---:|---|---:|
| TaurosV0 | 9,914 | .649 | .640-.659 | .661 |
| Kakuna | 9,737 | .623 | .613-.632 | .635 |
| TaurosEnsemble | 1,440 | .609 | .584-.634 | .607 |
| KakunaEnsemble | 5,609 | .579 | .566-.592 | .594 |
| **Pooled** | **26,700** | **.623** | **.617-.628** | |

How to read the table:
- **Luck adjustment.** It regresses each game's result on its counts of crits, misses, full paralysis, freezes and
  secondary paralysis against their expected rates, within opponent. It is useful for comparisons, not as an absolute
  level: the parser's expected rates are slightly off, which shifts every adjusted figure by about 1 point.
- **Long games.** MetapodV0 wins short games and is roughly even in long ones. Against TaurosV0 it scores .82 under
  40 turns and .48 at 80+.
- **Earlier gauntlet.** The configuration was chosen on an earlier, smaller block of games (`g_final2`, 600 games
  weighted 2:2:1:1): .6425. Pooling every later box gives 61.8% of 8,399 games on that same weighting. The two are
  1.2 sigma apart.
- **Metamon's own versions.** These are the bots as published by Metamon's repository at commit 0a00a759 with their
  Hugging Face checkpoints. TaurosEnsemble needed a checkpoint key-name fix to load (`scripts/patch_metamon_ensemble.py`).

**What the search is worth.** One paired box (next_ab1, 1,200 games per arm against the same opponents, same box and
schedule) compared the live configuration with variants:

| Arm | Score | Difference from live |
|---|---:|---|
| live (tau .02, `--infer`) | .611 | |
| search off (policy only) | .573 | −3.8 [−7.6, −0.2] |
| `--infer` off | .623 | +1.2 [−2.4, +4.8] |
| tau .01 | .647 | +3.6 [−0.9, +8.0] |
| tau .05 | .592 | −1.9 [−4.7, +0.7] |
| v7@300 (pre-specialization) + search | .602 | −0.9 [−5.1, +3.2] |

The search helps. The choices of tau .02 and `--infer` were tuned in-engine against our own policy. On outside
opponents, neither beats its alternatives by a margin the data can separate.

### Round robin: Metamon's bots against each other

The Metamon bots also played each other: 6,918 games on one A40 box on 2026-11-22, each bot drawing its own random
team every game, with the same server setup. That shows where MetapodV0's row sits.

| Row vs column | MetapodV0 | TaurosV0 | Kakuna | TaurosEnsemble | KakunaEnsemble |
|---|---:|---:|---:|---:|---:|
| MetapodV0 | | .649 | .623 | .609 | .579 |
| TaurosV0 | .351 | | .548 | .457 | .535 |
| Kakuna | .377 | .452 | | .404 | .461 |
| TaurosEnsemble | .391 | .543 | .596 | | .563 |
| KakunaEnsemble | .421 | .465 | .539 | .437 | |

Caveats for this table:
- The MetapodV0 row is the historical 26,700 games above, not games from the round-robin box. Its column entries are
  1 − row.
- Round-robin cells are 1,114-1,200 games each, with Wilson 95% half-widths of about ±.03.
- MetapodV0 had its fixed teams, and the Metamon bots random ones.

### On the public ladder

**Record: 305 wins, 136 losses, 3 ties** in 444 logged battles as `MetapodV0`, from 2026-11-09 07:42 to 2026-11-12
09:57 (that is spec2@200 alone; the account later ran other versions). It played one or two battles at a time from a
laptop.

- **Peak Elo: 1607** (1606.9) on 2026-11-10 00:32, after 80 games on the profile; GXE at that point was 85.7.
- **Elo on 2026-11-12, when this version was retired:** 1399 (GXE 82.7). Ladder Elo swings by a couple of hundred
  points over a few dozen games, so quote the record and the peak together, not the peak alone.
- **Battle counts:** our battle log counts more battles than the profile's own counter, so quote one source per
  number.
- **Glicko:** snapshots of about 1820-1826 were taken after later versions had joined the account, so they are not
  MetapodV0's alone.
- **Long games:** in a review of its first ~100 ladder battles, it went 0-9 in games of 80+ turns.
- **Set pool misses:** the 156-set pool misses revealed moves in 14% of opposing Pokémon on the ladder.

### Other opponents (for context)

Against Foul Play (GPL, run unmodified except for a small run-time compatibility patch for Gen 1's "fight" request,
and not included here), the live arm scored:
- .820 of 300 at 200 ms per move, human teams;
- .800 of 300 at 1,000 ms;
- .785 of 200 in a mirror of our two teams.

## Install

**Needs:**
- [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12+; the project was developed on 3.14);
- a Rust toolchain ([rustup](https://rustup.rs), stable; the engine uses edition 2024, so Rust ≥ 1.85);
- for the Showdown tests and tools only: Node.js ≥ 18.

```sh
git clone https://github.com/harrison-barnes/metapodV0-pokemon-bot.git
cd metapodV0-pokemon-bot
uv sync --all-groups          # builds the engine's Python bindings (gen1py) from engine/py with maturin
uv run pytest -q              # about 4 minutes on a laptop CPU
```

The checkpoints (2 × 17-20 MB) and the matchup table (37 MB + 17 MB, `data/matchup_table_v2*.npz`) are plain files in
the repository. No download step is needed.

## Play

### Locally, in the engine (no server)

```sh
uv run python scripts/play_local.py                       # one game: MetapodV0 (search) vs its own policy without search
uv run python scripts/play_local.py --games 10 --opponent smart
uv run python scripts/play_local.py --games 2 --out examples/demo_game.log
```

The log shows each turn's active Pokémon, both choices, and the search's root value and mixed strategy.
`examples/demo_game.log` is a saved run of two games.

### On Pokémon Showdown

The account comes from environment variables. Nothing is read from or written to the repository:

```sh
export SHOWDOWN_USERNAME=your_bot_account
export SHOWDOWN_PASSWORD='...'
N=5 bash scripts/play_showdown.sh                     # 5 battles on the public Gen 1 OU ladder
CHALLENGE=friend N=1 bash scripts/play_showdown.sh    # challenge one user instead
SERVER=local ACCEPT=you bash scripts/play_showdown.sh # a local server at 127.0.0.1:8000, accept your challenges
```

**What the adapter does:** `agent/showdown_player.py` (on top of [poke-env](https://github.com/hsahovic/poke-env))
rebuilds the engine state from the Showdown protocol and runs the search each turn. With `--trace`, it writes one row
per decision: candidates, payoff matrix, mixes and timing.

**If you run it on the public ladder:**
- Use a dedicated account that is clearly marked as a bot.
- Keep `CONCURRENT` at 1-2.
- Respect Pokémon Showdown's rules for bots.

## Train

| Command | What it does |
|---|---|
| `bash scripts/train_smoke.sh` | 2 PPO iterations on the CPU (about 2 min), checking the whole training loop with no extra inputs |
| `RUN=spec2_repro bash scripts/train.sh` | the spec2 recipe: 200 iterations from v7@300 (see the header of the script) |
| `bash scripts/build_tables.sh` | rebuilds the exact 1v1 dataset, the tablebase and the matchup table from the engine (about 1,350 core-hours) |

**`scripts/train.sh` needs inputs this repository does not contain:**
- **Starting and league checkpoints:** v7@300 policy and critic, v5d@400, v6, and the two human-imitation nets
  (`bc_full`, `bc_skill`), in `checkpoints/lineage/`. They are not published here (yet); if they are, they will be
  attached to a release on this repository's Releases page rather than committed.
- **Real players' teams:** `data/human_teams_v1.json`. Build it from Metamon's parsed replays: download them, convert
  them with `agent/metamon_convert.py` (needs `uv sync --group data`), then dump the teams with `agent/team_search.py`.
- **The exact 1v1 dataset** (`data/dataset_aws`, `data/dataset_pp`, about 7.5 GB), behind the tablebase:
  `scripts/build_tables.sh`.

**Other training code:**
- `agent/train_bc.py`: the human-imitation networks.
- `agent/team_search.py`: the team search.
- `agent/train_v4.py`: the single-process trainer whose options `agent/actor_learner.py` shares.

## Evaluate against Metamon

Metamon is an external dependency: install it from its own repository, not from here.

```sh
git clone --recursive https://github.com/UT-Austin-RPL/metamon.git ../metamon     # we used commit 0a00a759
(cd ../metamon && uv venv --python 3.12 && uv pip install -e .)
(cd ../metamon/server/pokemon-showdown && npm install && node pokemon-showdown start 8000 --no-security &)
uv run python scripts/patch_metamon_ensemble.py ../metamon      # only needed for TaurosEnsemble

METAMON_DIR=../metamon bash scripts/eval_metamon.sh             # live vs search-off, TaurosV0 and Kakuna
uv run python -m agent.next_eval_analyze agent/runs/eval_metamon/games.jsonl --ref live
```

**How the harness works:**
- `agent/next_eval_box.py` runs every arm against every opponent at the same time on one server, with the same team
  schedule per (opponent, replica), so arms can be compared pair by pair.
- `agent/next_eval_play.py` is one side of a pair.
- `agent/next_eval_analyze.py` reports scores with Wilson intervals, matched differences with a cluster bootstrap,
  and the luck adjustment.

**Sizing:**
- Metamon's bots are 50-140M-parameter models; give them a GPU (each pair needs a few GB) or use `TaurosV0@cpu`.
- For a 4-point effect, plan on more than 1,200 games per arm.

## Tests

```sh
uv run pytest -q                                # Python: engine bindings, checkpoint, search, adapter (offline)
(cd engine && cargo test --release)             # Rust: battle mechanics, solver, endgames
(cd tools/showdown && npm install) && (cd engine && cargo build --release)
uv run python tools/diffcheck.py                # engine vs Showdown's simulator, 1v1 random play (docs/ENGINE.md)
uv run python tools/diffcheck6.py               # the same for random 6v6 teams
```

**The adapter tests run offline,** on fixtures recorded with the Showdown simulator (`tests/fixtures`, regenerate with
`tools/showdown/gen1_*.js` after `npm install` there). They check:
- 17 hand-written cases (traps, recharge, sleep counters, Counter's damage, HP from percentages, and others);
- Gen 1 paralysis and burn stat drops against Showdown's own numbers;
- the whole state rebuild against Showdown's internal state over 40 battles of deliberately exotic teams.

The rebuild test allows a 0.5% mismatch rate per field. The known residual gaps are Mimic, Transform and Rage corner
cases at 0.1-0.3% of decisions.

## Hardware notes

**Playing:**
- CPU only is fine. On a laptop CPU (4 threads) a decision takes about 0.3-1 s on average and a few seconds at worst.
- On the ladder the bot used under 1% of its turn timer.
- Memory: about 1-1.5 GB per process. The matchup table decompresses to about 700 MB, and loading takes a few
  seconds.
- `--device cuda` or `mps` works too.

**Evaluation boxes:** one 24-48 GB GPU with 32-64 cores, running 15 battle pairs and 8 Metamon bot processes at once.
Search time there was about 0.26 s per decision.

**Training** (`scripts/train.sh`):
- Hardware: one CUDA GPU (bf16) for the learner, plus 16-64 CPU cores for the actor processes that run the Rust
  engine.
- The original runs used rented single-GPU cloud machines.
- On a CPU-only machine use `DEVICE=cpu` and much smaller `ENVS`/`MINIBATCH` (see `scripts/train_smoke.sh`).

**Exact 1v1 dataset** (`scripts/build_tables.sh`): the recorded per-pair solve times add up to about 1,350
core-hours. It was run sharded over a fleet of cloud machines. (`docs/DATASET.md` is the design document; its
cost estimate predates the real run.)

## Repository layout

```
agent/          policy.py, obs.py, features.py, senses.py, matchups.py   the network and its inputs
                search_exact.py, search.py, qsolve.py                    the play-time search and set guesser
                showdown_player.py                                       Showdown adapter (poke-env)
                actor_learner.py, train_v4.py, selfplay.py, tablebase.py PPO training, league, tablebase
                teams.py, team_search.py, team_tournament.py             team generation and search
                train_bc.py, metamon_convert.py                          human-imitation nets, replay conversion
                next_eval_box.py, next_eval_play.py, next_eval_analyze.py evaluation harness
                test_*.py, next_search_test.py                           test programs (wrapped by tests/)
analysis/       dataset.py                                               1v1 dataset row format
engine/         Rust battle engine, solver and dataset pipeline (src/, tests/); engine/py: Python bindings (gen1py)
data/           gen1.json                (species, moves, learnsets, type chart: extracted from pret/pokered)
                sets_ladder.json         the 156-set ladder pool (from Smogon usage statistics)
                smogon/                  Gen 1 OU usage statistics (1760+, 2026-09) and fitted team weights
                matchup_table_v2*.npz    exact 1v1 matchup values (two files, merged on load)
                search_teams_v1/         MetapodV0's two teams (Showdown export format)
                showdown_names.json, dataset_smogon_v1/schema.json, stall_pairs.json
checkpoints/    spec2_00200.pt, spec2_00200_critic.pt                    MetapodV0
scripts/        play_local.py, play_showdown.sh, train.sh, train_smoke.sh, eval_metamon.sh, build_tables.sh
tools/          extract_pokered.py, ladder_sets.py, build_matchup_table.py, luck.py (luck counts from battle logs);
                showdown/ (Showdown-simulator scripts that record the test fixtures)
tests/          pytest suite and fixtures
docs/           ENGINE.md (what the engine reproduces and how it is validated), DATASET.md
examples/       demo_game.log
```

## What is not included

| Left out | Why |
|---|---|
| Metamon's parsed human replays, and anything copied from them (the human team pool, converted arrays) | Metamon's dataset, licensed CC BY-NC 4.0. Get it from [jakegrigsby/metamon-parsed-replays](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays) and convert it with `agent/metamon_convert.py` |
| The exact 1v1 dataset (7.5 GB), training runs, evaluation logs, ladder replays | Bulk outputs; all rebuildable with the scripts above |
| Metamon, Foul Play, pokered and Pokémon Showdown sources | Third-party projects: install them from their own repositories. Foul Play is GPL-3.0 and was only ever run as a separate program |
| Earlier checkpoints of the lineage (v1-v7, bc nets) | Kept out of the clone to keep it small; the bc nets are direct imitations of the CC BY-NC replay data. See [Train](#train) |
| Research notes, cloud launch scripts, the later ladder versions (mpr1/mpr2, opponent memory) | Project-internal, or not MetapodV0 |

**About the weights:**
- The MIT license covers this repository's code and data files.
- The checkpoints were trained partly against human-imitation opponents, and with real players' teams, derived from
  Metamon's CC BY-NC 4.0 replay dataset.
- If you build on the weights, respect that dataset's terms as well.

## Which code version this is

The ladder bot ran from a working tree that kept changing after MetapodV0 was retired. This repository follows these
rules:

- **The live code, as MetapodV0 played,** for the set guesser and matchup lookups: `agent/search.py` and
  `agent/matchups.py`. The live `search.py` already includes the fix for opposing species the set pool doesn't cover
  (`fallback_set`).
- **The later cleaned-up copy, where its changes are bug fixes and everything new defaults to off,** for:
  - `agent/search_exact.py`. With default options it makes the same choices and payoff matrices as before. Its bug
    fixes: hidden sleep and confusion counters advanced once per decision instead of once per search call; an action
    illegal in one guess is played as pass instead of raising; a trap that may continue is drawn per guess.
  - `agent/showdown_player.py`, rebuilt against the ground-truth test (`agent/test_adapter_truth.py`). Fixes include:
    trap state after the first continuation turn, a recharge flag that never cleared, sleep and confusion turns
    counted from the protocol, opposing HP from percentages, the damage Counter reads, and a battle opening that
    arrives in one message. The Gen 1 paralysis and burn stat-drop fix was already in the live code.
  - The engine (`engine/`). Default rules are unchanged; an opt-in `smogon-sd` ruleset was added and a Transform bug
    in the bindings was fixed.
- **How much these fixes matter:** the cleaned-up code was measured against the original in paired games (4,800
  pairs against the four Metamon bots) at +0.1 points [−1.9, +2.0]. That is the same strength.
- **Additions for this release:**
  - credentials from environment variables;
  - the matchup table split into two compressed files (loaded by `agent.matchups.load_hp_left`);
  - an evaluation harness that points at an external Metamon install;
  - `scripts/` and `tests/`.
- **Removed:** experimental options that were never part of MetapodV0, namely the learned set model, opponent memory,
  and evaluation opponents built from specific ladder players' replays.

## Credits

- **[pret/pokered](https://github.com/pret/pokered):** the Pokémon Red/Blue disassembly. `data/gen1.json` (species,
  moves, learnsets, type chart) is extracted from it by `tools/extract_pokered.py`, and the engine's mechanics follow
  its battle code.
- **[Metamon](https://github.com/UT-Austin-RPL/metamon)** (Jake Grigsby et al., UT Austin Robot Perception and
  Learning Lab):
  - the opponents every result above is measured against;
  - the parsed human replay dataset behind our human-imitation opponents and team pool;
  - the local-server evaluation setup we followed.
- **[Pokémon Showdown](https://github.com/smogon/pokemon-showdown)** (Smogon): the battle simulator and server. The
  ladder, the tests' ground truth, and Gen 1 OU's rules as played.
- **[poke-env](https://github.com/hsahovic/poke-env)** (Haris Sahovic): the Python Showdown client the adapter is
  built on.
- **[Smogon](https://www.smogon.com/stats/):** usage statistics, the source of the 156-set ladder pool.
- **Foul Play** (pmariglia): an outside opponent in our evaluation, run as a separate program.

Pokémon is a trademark of Nintendo, Game Freak and Creatures. This is an unofficial fan research project, not
affiliated with or endorsed by them.

## License

MIT, © 2026 Harrison Barnes. See [LICENSE](LICENSE).
