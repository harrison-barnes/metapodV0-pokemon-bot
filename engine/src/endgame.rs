//! Exact solver for 2v2 endgames: each side has two Pokemon (one active, one on the bench, possibly
//! fainted), and switching is an action. Same algorithm as `solver` (enumerate every reachable
//! state, solve each turn's matrix game with `nash`, strongly connected components in reverse
//! topological order with Gauss-Seidel value iteration on cycles); the turn semantics are
//! `battle6`'s, built from the same pieces (`battle6::send_out_side`, `clause_flags`, ...):
//!
//! - A switch happens before moves and uses up the side's turn (`Engine::turn_switching`). It is
//!   allowed unless the side is recharging, raging, thrashing or charging, including while trapped.
//!   The Pokemon leaving keeps HP, major status (Toxic becomes ordinary poison), the sleep counter,
//!   whether its sleep came from the opponent, and its healing-PP level; everything volatile is lost.
//! - A faint ends the turn; the survivor keeps its volatile state (`battle6::survivor_after_faint`).
//!   With one Pokemon left there is only one possible replacement, so the forced send-out is
//!   applied right away inside the transition: no state ever waits for a replacement, and a double
//!   KO with both benches alive sends out both. A side with nothing left loses; both at once is a
//!   double KO, worth `draw_value`.
//! - Sleep Clause and Freeze Clause are recomputed from both teams before every turn, as
//!   `Battle6::set_clauses` does (a side's sleep move fails while an opposing Pokemon, active or
//!   benched, is asleep from a move; its moves can't freeze while one is frozen).
//! - PP is ignored, as in the 1v1 solver, except the healing move's coarse PP level under
//!   `Rules::heal_pp` (it is carried with a benched Pokemon). `battle6` tracks every move's PP; the
//!   two agree until a move would run out.
//!
//! A state is the active `Battle` (both active Pokemon, canonicalised by an `Engine` in team mode,
//! see `Engine::set_teams`), which member is active on each side, and each benched member's lasting
//! state. The HP grid and roll buckets work as in `solver`; the grid is per team member, and
//! benched Pokemon can have a coarser one (`set_bench_levels`).
//!
//! Most states differ from others only in their benches, which never affect the moves, so each
//! turn's outcomes are cached by pre-turn position (`TurnOutcomes`). Value iteration on a cycle
//! re-evaluates only states whose successors moved (`solve_component`).

use crate::battle6::{can_switch, clause_flags, send_out_side, slept_by_foe, survivor_after_faint, Battle6};
use crate::chance::{Chance, Enumerator};
use crate::data::Dex;
use crate::engine::{Engine, MidTurn, STRUGGLE_SLOT};
use crate::nash;
use crate::solver::{Expansion, SolveStats};
use crate::state::*;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// Action codes beside move slots 0-3 and `engine::STRUGGLE_SLOT`: no choice to make (asleep,
/// recharging, trapped...), and switching to the benched Pokemon.
pub const PASS: u8 = 5;
pub const SWITCH: u8 = 6;

/// Precision of a 2v2 solve: damage-roll buckets and HP levels as in `meta::Precision`, a coarser
/// HP grid for benched Pokemon (0 = the same), and a state cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndgamePrecision {
    pub buckets: u8,
    pub hp_levels: u16,
    pub bench_levels: u16,
    pub max_states: usize,
}

impl EndgamePrecision {
    pub const fn new(buckets: u8, hp_levels: u16, bench_levels: u16, max_states: usize) -> EndgamePrecision {
        EndgamePrecision { buckets, hp_levels, bench_levels, max_states }
    }
}

/// Precision ladder for `solve_ladder`: the finest setting that fits, then coarser ones. Caps keep
/// a solve under about 4 GB (roughly 1 KB per state); hitting a cap can take many minutes.
pub const LADDER: &[EndgamePrecision] = &[
    EndgamePrecision::new(1, 12, 6, 4_000_000),
    EndgamePrecision::new(1, 12, 4, 4_000_000),
    EndgamePrecision::new(1, 6, 0, 4_000_000),
    EndgamePrecision::new(1, 6, 3, 4_000_000),
    EndgamePrecision::new(1, 3, 0, 4_000_000),
    EndgamePrecision::new(1, 2, 0, 4_000_000),
];

/// `levels` evenly spaced HP values from 1 to `max` (empty: exact).
fn hp_grid(max: u16, levels: u16) -> Vec<u16> {
    if levels == 0 || levels >= max {
        return Vec::new();
    }
    let mut g: Vec<u16> = (0..=levels).map(|k| ((k as u32 * max as u32 + levels as u32 - 1) / levels as u32).max(1) as u16).collect();
    g.dedup();
    g
}

/// The grid points around `hp` and the probability of rounding up, preserving expected HP.
fn split_hp(g: &[u16], hp: u16) -> (u16, u16, f64) {
    if g.is_empty() {
        return (hp, hp, 0.0);
    }
    let k = g.partition_point(|&x| x <= hp); // g[k-1] <= hp < g[k]
    if k == 0 || g[k - 1] == hp || k == g.len() {
        return (hp, hp, 0.0);
    }
    let (lo, hi) = (g[k - 1], g[k]);
    (lo, hi, (hp - lo) as f64 / (hi - lo) as f64)
}

/// A benched team member's lasting state (zero when fainted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bench {
    pub hp: u16,
    /// Major status, with the sleep counter in the low bits.
    pub status: u8,
    /// Asleep because of the opponent's move (counts for Sleep Clause), not Rest.
    pub slept_by_foe: bool,
    /// Healing-PP level (`Side::clauses >> HEAL_PP_SHIFT`), under `Rules::heal_pp`.
    pub heal: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct State2 {
    /// The two active Pokemon, as the 1v1 engine sees them.
    pub b: Battle,
    /// Team index (0 or 1) of each side's active Pokemon.
    pub active: [u8; 2],
    /// Each side's other team member.
    pub bench: [Bench; 2],
}

/// `State2` packed for hashing: the battle's 102 bytes, the active indices in its two spare bytes,
/// and the benches in the last word.
pub type Packed2 = [u64; 14];

impl State2 {
    pub fn pack(&self) -> Packed2 {
        let mut out = [0u64; 14];
        out[..13].copy_from_slice(&self.b.pack());
        out[12] |= (self.active[0] as u64) << 48 | (self.active[1] as u64) << 56;
        let w = |x: &Bench| x.hp as u64 | (x.status as u64) << 16 | (x.slept_by_foe as u64 | (x.heal as u64) << 1) << 24;
        out[13] = w(&self.bench[0]) | w(&self.bench[1]) << 32;
        out
    }
}

/// One turn's outcomes from a pre-turn position (after switches and clauses), independent of the
/// benches: battles still going on (canonical, HP already on the grid) and battles where someone
/// fainted (raw: the replacement and the result depend on the benches).
struct TurnOutcomes {
    ongoing: Vec<(Battle, f64)>,
    ended: Vec<(Battle, Outcome, f64)>,
}

/// Where one path through a turn leads.
#[derive(Debug, Clone, PartialEq)]
pub enum Next {
    State(State2),
    End(Outcome),
}

/// The four Pokemon and the rules of a 2v2 endgame.
pub struct Teams {
    pub mons: [[Mon; 2]; 2],
    pub rules: Rules,
    /// matchups[i][j]: side 0's member i against side 1's member j.
    matchups: [[Matchup; 2]; 2],
}

impl Teams {
    pub fn new(mons: [[Mon; 2]; 2], rules: Rules) -> Teams {
        let m = |i: usize, j: usize| Matchup { mons: [mons[0][i].clone(), mons[1][j].clone()], rules: rules.clone() };
        let matchups = [[m(0, 0), m(0, 1)], [m(1, 0), m(1, 1)]];
        Teams { mons, rules, matchups }
    }

    pub fn matchup(&self, active: [u8; 2]) -> &Matchup {
        &self.matchups[active[0] as usize][active[1] as usize]
    }
}

pub struct EndgameSolver<'a> {
    pub dex: &'a Dex,
    pub teams: &'a Teams,
    /// One engine per active pair, indexed `2 * active[0] + active[1]`.
    engines: Vec<Engine<'a>>,
    ids: FxHashMap<Packed2, u32>,
    pub states: Vec<State2>,
    pub value: Vec<f64>,
    solved: Vec<bool>,
    /// Convergence tolerance for cyclic components.
    pub tol: f64,
    /// Residual below which a component that has stopped converging counts as converged.
    pub noise: f64,
    pub max_sweeps: usize,
    /// Give up (`try_solve` returns None) once this many states have been discovered.
    pub max_states: usize,
    /// hp_grid[s][i]: HP grid of side s's member i (see `Solver::set_hp_levels`); empty means exact.
    hp_grid: [[Vec<u16>; 2]; 2],
    /// The same while benched (see `set_bench_levels`); empty means no further rounding.
    bench_grid: [[Vec<u16>; 2]; 2],
    aborted: bool,
    /// Enumerate each turn in two halves (see `Solver::split_turns`).
    pub split_turns: bool,
    /// Per-turn discount towards 0.5 (see `Solver::discount`).
    pub discount: f64,
    /// Value of both sides running out of Pokemon on the same turn.
    pub draw_value: f64,
    /// Report every millionth state on stderr.
    pub progress: bool,
    /// Turn outcomes by pre-turn position: states that differ only in their benches share them.
    /// Holds up to `cache_size` outcomes, then starts over (0 disables it).
    cache: FxHashMap<[u64; 14], Rc<TurnOutcomes>>,
    cached: usize,
    pub cache_size: usize,
    pub cache_hits: u64,
    started: std::time::Instant,
    pub stats: SolveStats,
}

const UNVISITED: u32 = u32::MAX;
const MAX_CELLS: usize = nash::MAX_CHOICES * nash::MAX_CHOICES;

impl<'a> EndgameSolver<'a> {
    pub fn new(dex: &'a Dex, teams: &'a Teams) -> EndgameSolver<'a> {
        let team = |s: usize| &teams.mons[s][..];
        let engines = (0..4u8)
            .map(|k| {
                let mut e = Engine::new(dex, teams.matchup([k >> 1, k & 1]));
                e.set_teams([team(0), team(1)]);
                e
            })
            .collect();
        EndgameSolver {
            dex,
            teams,
            engines,
            ids: FxHashMap::default(),
            states: Vec::new(),
            value: Vec::new(),
            solved: Vec::new(),
            tol: 1e-12,
            noise: 1e-10,
            max_sweeps: 100_000,
            max_states: usize::MAX,
            hp_grid: Default::default(),
            bench_grid: Default::default(),
            aborted: false,
            split_turns: true,
            discount: 1.0,
            draw_value: 0.5,
            progress: false,
            cache: FxHashMap::default(),
            cached: 0,
            cache_size: 4_000_000,
            cache_hits: 0,
            started: std::time::Instant::now(),
            stats: SolveStats { states: 0, sccs: 0, largest_scc: 0, sweeps: 0, max_residual: 0.0, paths: 0, transitions: 0 },
        }
    }

    /// Damage-roll buckets for every matchup (0 = exact; see `Engine::roll_buckets`).
    pub fn set_roll_buckets(&mut self, k: u8) {
        for e in &mut self.engines {
            e.roll_buckets = k;
        }
        self.split_turns = k != 1;
    }

    /// Approximate HP with `levels` evenly spaced values per team member (0 restores exact HP).
    pub fn set_hp_levels(&mut self, levels: u16) {
        self.hp_grid = [0, 1].map(|s| [0, 1].map(|i| hp_grid(self.teams.mons[s][i].max_hp, levels)));
    }

    /// A coarser grid for benched Pokemon (0: none beyond `set_hp_levels`). After every turn a
    /// benched Pokemon's HP is rounded onto it the same way, which in practice happens when it
    /// switches out. States that differ only in their benches are most of a 2v2 game, so this cuts
    /// the state count roughly by the square of the ratio. With `hp_levels` a multiple of
    /// `levels`, the bench grid is part of the active grid.
    pub fn set_bench_levels(&mut self, levels: u16) {
        self.bench_grid = [0, 1].map(|s| [0, 1].map(|i| hp_grid(self.teams.mons[s][i].max_hp, levels)));
    }

    /// All precision settings at once.
    pub fn set_precision(&mut self, p: EndgamePrecision) {
        self.set_roll_buckets(p.buckets);
        self.set_hp_levels(p.hp_levels);
        self.set_bench_levels(p.bench_levels);
        self.max_states = p.max_states;
    }

    pub fn eng(&self, active: [u8; 2]) -> &Engine<'a> {
        &self.engines[2 * active[0] as usize + active[1] as usize]
    }

    // ------------------------------------------------------------------ positions

    /// A benched member's state at full health.
    fn fresh_bench(&self, s: usize, i: usize) -> Bench {
        let m = self.teams.matchup([i as u8, i as u8]);
        let heal = m.heal_max_pp(self.dex, s).map_or(0, heal_top_level);
        Bench { hp: self.teams.mons[s][i].max_hp, status: 0, slept_by_foe: false, heal }
    }

    /// A position as it stands right after both active Pokemon have been sent out: the given HP and
    /// major status for every team member (HP 0 is fainted), full healing PP, everything volatile
    /// fresh. An active Pokemon with 0 HP is replaced by its bench at once.
    pub fn position(&self, active: [u8; 2], hp: [[u16; 2]; 2], status: [[u8; 2]; 2]) -> State2 {
        let a = active.map(|x| x as usize);
        let m = self.teams.matchup(active);
        let mut b = m.initial(self.dex);
        let mut need = [false; 2];
        let mut bench = [Bench::default(); 2];
        for s in 0..2 {
            for i in 0..2 {
                assert!(hp[s][i] <= self.teams.mons[s][i].max_hp, "HP {} out of range for side {s} member {i}", hp[s][i]);
            }
            let x = &mut b.sides[s];
            x.hp = hp[s][a[s]];
            x.status = if x.hp > 0 { status[s][a[s]] } else { 0 };
            if x.status & PAR != 0 {
                x.stats[SPE] = (x.stats[SPE] >> 2).max(1);
            }
            if x.status & BRN != 0 {
                x.stats[ATK] = (x.stats[ATK] >> 1).max(1);
            }
            need[s] = x.hp == 0;
            let o = 1 - a[s];
            if hp[s][o] > 0 {
                bench[s] = Bench { hp: hp[s][o], status: status[s][o], ..self.fresh_bench(s, o) };
            }
        }
        let mut st = State2 { b, active, bench };
        for s in 0..2 {
            if need[s] {
                assert!(st.bench[s].hp > 0, "side {s} has no Pokemon left");
                self.send_out_bench(&mut st, s);
                st.bench[s] = Bench::default();
            }
        }
        self.canonical(st)
    }

    /// The start of the endgame: both sides lead with member 0, everyone at full health.
    pub fn initial(&self) -> State2 {
        let hp = [0, 1].map(|s| [0, 1].map(|i| self.teams.mons[s][i].max_hp));
        self.position([0, 0], hp, [[0; 2]; 2])
    }

    /// A 6v6 battle with two Pokemon a side as a 2v2 state (pending replacements applied), not
    /// canonicalised. PP and what has been revealed are dropped.
    pub fn from_battle6_raw(&self, g: &Battle6) -> State2 {
        assert!(g.teams.iter().all(|t| t.len() == 2) && g.result.is_none(), "need an ongoing battle with two Pokemon a side");
        let mut st = State2 { b: g.b.clone(), active: g.active.map(|i| i as u8), bench: [Bench::default(); 2] };
        for s in 0..2 {
            let t = &g.teams[s][1 - g.active[s]];
            if t.hp > 0 {
                st.bench[s] = Bench { hp: t.hp, status: t.status, slept_by_foe: t.slept_by_foe, heal: self.fresh_bench(s, 1 - g.active[s]).heal };
            }
        }
        for s in 0..2 {
            if g.need_replace[s] {
                self.send_out_bench(&mut st, s);
                st.bench[s] = Bench::default();
            }
        }
        st
    }

    pub fn from_battle6(&self, g: &Battle6) -> State2 {
        self.canonical(self.from_battle6_raw(g))
    }

    /// Zeroes state that can no longer matter (see `Engine::canonicalize` in team mode).
    pub fn canonical(&self, mut st: State2) -> State2 {
        self.eng(st.active).canonicalize(&mut st.b);
        for s in 0..2 {
            let x = &mut st.bench[s];
            if x.hp == 0 {
                *x = Bench::default();
            }
            x.slept_by_foe &= x.status & SLP_MASK != 0;
        }
        st
    }

    // ------------------------------------------------------------------ transitions

    /// Legal action codes for side s: move slots (or Struggle), `PASS` when the side has no
    /// choice, and `SWITCH` when its bench is alive and the menu isn't skipped.
    pub fn actions(&self, st: &State2, s: usize) -> Vec<u8> {
        let eng = self.eng(st.active);
        let mut v = eng.legal_choices(&st.b, s);
        if !eng.needs_choice(&st.b, s) {
            v.push(PASS);
        }
        if st.bench[s].hp > 0 && can_switch(&st.b.sides[s]) {
            v.push(SWITCH);
        }
        v
    }

    pub fn action_name(&self, st: &State2, s: usize, code: u8) -> String {
        let a = st.active[s] as usize;
        match code {
            PASS => "(no choice)".to_string(),
            SWITCH => format!("switch to {}", self.dex.species[self.teams.mons[s][1 - a].species as usize].name),
            STRUGGLE_SLOT => "STRUGGLE".to_string(),
            c => self.dex.mv(self.teams.mons[s][a].moves[c as usize]).name.clone(),
        }
    }

    /// Side s's benched member becomes active (`SendOutMon`); the caller decides what the bench
    /// slot holds afterwards.
    fn send_out_bench(&self, st: &mut State2, s: usize) {
        let i = 1 - st.active[s] as usize;
        let t = st.bench[s];
        send_out_side(self.dex, &mut st.b, s, &self.teams.mons[s][i], t.hp, t.status, t.slept_by_foe);
        st.b.sides[s].clauses |= t.heal << HEAL_PP_SHIFT;
        st.active[s] = i as u8;
    }

    /// Before the moves: switches, then Sleep and Freeze Clause from both teams. Returns the
    /// position the turn starts from, the move choices and who switched.
    fn pre_turn(&self, st: &State2, a: [u8; 2]) -> (State2, [u8; 2], [bool; 2]) {
        let mut st = st.clone();
        let switched = a.map(|x| x == SWITCH);
        for s in 0..2 {
            if switched[s] {
                let x = &st.b.sides[s];
                let leaving = Bench { hp: x.hp, status: x.status, slept_by_foe: slept_by_foe(x), heal: (x.clauses & HEAL_PP_MASK) >> HEAL_PP_SHIFT };
                self.send_out_bench(&mut st, s);
                st.bench[s] = leaving;
            }
        }
        for s in 0..2 {
            let o = 1 - s;
            let y = &st.b.sides[o];
            let bench = &st.bench[o];
            let foes = [(y.status, y.clauses & SLEPT_BY_FOE != 0), (if bench.hp > 0 { bench.status } else { 0 }, bench.slept_by_foe)];
            let x = &mut st.b.sides[s];
            x.clauses = (x.clauses & !(SLEEP_BLOCKED | FREEZE_BLOCKED)) | clause_flags(foes.into_iter());
        }
        let choice = a.map(|x| if x <= STRUGGLE_SLOT { x } else { 0 });
        (st, choice, switched)
    }

    /// After the moves: faints, the forced replacements, and the end of the battle.
    fn post_turn(&self, pre: &State2, b: Battle, o: Outcome) -> Next {
        let mut st = State2 { b, active: pre.active, bench: pre.bench };
        let fainted = match o {
            Outcome::Ongoing => return Next::State(self.canonical(st)),
            Outcome::Win(w) => [w == 1, w == 0],
            Outcome::Draw => [true, true],
        };
        let out = [0, 1].map(|s| fainted[s] && st.bench[s].hp == 0);
        match out {
            [true, true] => return Next::End(Outcome::Draw),
            [true, false] => return Next::End(Outcome::Win(1)),
            [false, true] => return Next::End(Outcome::Win(0)),
            [false, false] => {}
        }
        for s in 0..2 {
            if fainted[s] {
                survivor_after_faint(&mut st.b.sides[1 - s]);
            }
        }
        for s in 0..2 {
            if fainted[s] {
                self.send_out_bench(&mut st, s);
                st.bench[s] = Bench::default();
            }
        }
        Next::State(self.canonical(st))
    }

    /// Plays one turn with the given action codes, drawing from `ch` (no HP grid rounding).
    pub fn step<C: Chance>(&self, st: &State2, a: [u8; 2], ch: &mut C) -> Next {
        let (pre, choice, switched) = self.pre_turn(st, a);
        let r = self.eng(pre.active).turn_switching(&pre.b, choice, switched, ch, false);
        self.post_turn(&pre, r.b, r.outcome)
    }

    /// Every outcome of one turn with exact probabilities (no HP grid rounding): the successor
    /// states and the probability of each battle result.
    pub fn transitions(&mut self, st: &State2, a: [u8; 2]) -> (Vec<(State2, f64)>, [f64; 3]) {
        let mut acc: FxHashMap<Packed2, (State2, f64)> = FxHashMap::default();
        let mut ends = [0.0; 3];
        self.enumerate(st, a, |next, p| match next {
            Next::State(ns) => acc.entry(ns.pack()).or_insert_with(|| (ns, 0.0)).1 += p,
            Next::End(Outcome::Win(w)) => ends[w] += p,
            Next::End(_) => ends[2] += p,
        });
        (acc.into_values().collect(), ends)
    }

    fn enumerate(&mut self, st: &State2, a: [u8; 2], mut f: impl FnMut(Next, f64)) {
        let (pre, choice, switched) = self.pre_turn(st, a);
        let eng = self.eng(pre.active);
        let paths = Self::enumerate_turn(eng, self.split_turns, &pre.b, choice, switched, |nb, o, p| f(self.post_turn(&pre, nb, o), p));
        self.stats.paths += paths;
    }

    /// Every path through one turn's moves: (battle, outcome, probability) to `f`. Returns the
    /// number of paths.
    fn enumerate_turn(eng: &Engine, split: bool, b: &Battle, choice: [u8; 2], switched: [bool; 2], mut f: impl FnMut(Battle, Outcome, f64)) -> u64 {
        let mut paths = 0;
        let mut e = Enumerator::new();
        if !split {
            loop {
                let r = eng.turn_switching(b, choice, switched, &mut e, false);
                f(r.b, r.outcome, e.prob());
                paths += 1;
                if !e.advance() {
                    break;
                }
            }
            return paths;
        }
        let mut mids: FxHashMap<MidTurn, f64> = FxHashMap::default();
        loop {
            match eng.start_turn_switching(b, choice, switched, &mut e) {
                Ok(mid) => *mids.entry(mid).or_insert(0.0) += e.prob(),
                Err((nb, o)) => f(nb, o, e.prob()),
            }
            paths += 1;
            if !e.advance() {
                break;
            }
        }
        for (mid, p1) in mids.iter() {
            let mut e = Enumerator::new();
            loop {
                let (nb, o) = eng.finish_turn(mid, &mut e);
                f(nb, o, p1 * e.prob());
                paths += 1;
                if !e.advance() {
                    break;
                }
            }
        }
        paths
    }

    /// Stochastic rounding of both active Pokemon's HP onto their grids: up to four weighted
    /// battles, passed to `out`.
    fn snap_battle(&self, active: [u8; 2], b: Battle, p: f64, mut out: impl FnMut(Battle, f64)) {
        let split = |s: usize| split_hp(&self.hp_grid[s][active[s] as usize], b.sides[s].hp);
        let (s0, s1) = (split(0), split(1));
        for (h0, p0) in [(s0.0, 1.0 - s0.2), (s0.1, s0.2)] {
            if p0 == 0.0 {
                continue;
            }
            for (h1, p1) in [(s1.0, 1.0 - s1.2), (s1.1, s1.2)] {
                if p1 == 0.0 {
                    continue;
                }
                let mut nb = b.clone();
                nb.sides[0].hp = h0;
                nb.sides[1].hp = h1;
                out(nb, p * p0 * p1);
            }
        }
    }

    /// Stochastic rounding of the benched Pokemon's HP onto the bench grid: up to four weighted
    /// pairs of benches.
    fn snap_bench(&self, active: [u8; 2], bench: [Bench; 2]) -> Vec<([Bench; 2], f64)> {
        let split = |s: usize| {
            let x = bench[s];
            if x.hp == 0 { (0, 0, 0.0) } else { split_hp(&self.bench_grid[s][1 - active[s] as usize], x.hp) }
        };
        let (s0, s1) = (split(0), split(1));
        let mut v = Vec::with_capacity(4);
        for (h0, p0) in [(s0.0, 1.0 - s0.2), (s0.1, s0.2)] {
            for (h1, p1) in [(s1.0, 1.0 - s1.2), (s1.1, s1.2)] {
                if p0 * p1 > 0.0 {
                    v.push(([Bench { hp: h0, ..bench[0] }, Bench { hp: h1, ..bench[1] }], p0 * p1));
                }
            }
        }
        v
    }

    /// The turn's outcomes from a pre-turn position (see `TurnOutcomes`), computed or cached.
    fn turn_outcomes(&mut self, pre: &State2, choice: [u8; 2], switched: [bool; 2]) -> Rc<TurnOutcomes> {
        let mut key = [0u64; 14];
        key[..13].copy_from_slice(&pre.b.pack());
        key[13] = choice[0] as u64 | (choice[1] as u64) << 8 | (switched[0] as u64) << 16 | (switched[1] as u64) << 24
            | (pre.active[0] as u64) << 32 | (pre.active[1] as u64) << 40;
        if let Some(t) = self.cache.get(&key) {
            self.cache_hits += 1;
            return t.clone();
        }
        let mut ongoing: FxHashMap<Packed, (Battle, f64)> = FxHashMap::default();
        let mut ended: FxHashMap<(Packed, u8), (Battle, Outcome, f64)> = FxHashMap::default();
        let eng = self.eng(pre.active);
        let mut add = |mut b: Battle, o: Outcome, p: f64| match o {
            Outcome::Ongoing => {
                eng.canonicalize(&mut b);
                self.snap_battle(pre.active, b, p, |nb, q| ongoing.entry(nb.pack()).or_insert_with(|| (nb, 0.0)).1 += q);
            }
            _ => {
                let tag = match o {
                    Outcome::Win(w) => w as u8,
                    _ => 2,
                };
                ended.entry((b.pack(), tag)).or_insert_with(|| (b, o, 0.0)).2 += p;
            }
        };
        let paths = Self::enumerate_turn(eng, self.split_turns, &pre.b, choice, switched, &mut add);
        self.stats.paths += paths;
        let t = Rc::new(TurnOutcomes { ongoing: ongoing.into_values().collect(), ended: ended.into_values().collect() });
        let n = t.ongoing.len() + t.ended.len();
        if self.cached + n > self.cache_size {
            self.cache.clear();
            self.cached = 0;
        }
        if n <= self.cache_size {
            self.cached += n;
            self.cache.insert(key, t.clone());
        }
        t
    }

    // ------------------------------------------------------------------ solving

    fn intern(&mut self, st: State2) -> u32 {
        let key = st.pack();
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.states.len() as u32;
        if self.progress && id > 0 && id % 1_000_000 == 0 {
            eprintln!("  {id} states, {} solved components, {:.0}s", self.stats.sccs, self.started.elapsed().as_secs_f64());
        }
        self.ids.insert(key, id);
        self.states.push(st);
        self.value.push(0.5);
        self.solved.push(false);
        id
    }

    /// Solved value of a (canonical) state, if it was reached during solving.
    pub fn value_of(&self, st: &State2) -> Option<f64> {
        self.ids.get(&st.pack()).map(|&id| self.value[id as usize])
    }

    /// All transitions out of a state, with successors interned. Rows and columns are action codes.
    pub fn expand(&mut self, id: u32) -> Expansion {
        let st = self.states[id as usize].clone();
        let rows = self.actions(&st, 0);
        let cols = self.actions(&st, 1);
        let n = rows.len() * cols.len();
        let mut terminal = Vec::with_capacity(n);
        let mut start = Vec::with_capacity(n + 1);
        let mut succ = Vec::new();
        start.push(0);
        let mut acc: FxHashMap<Packed2, (State2, f64)> = FxHashMap::default();
        // Cells share most successors: a small local map spares lookups in the big one.
        let mut local: FxHashMap<Packed2, u32> = FxHashMap::default();
        for &r in &rows {
            for &c in &cols {
                let (pre, choice, switched) = self.pre_turn(&st, [r, c]);
                let t = self.turn_outcomes(&pre, choice, switched);
                let benches = self.snap_bench(pre.active, pre.bench);
                // Distinct battles and benches: distinct successors, unlike the faints.
                for (b, p) in &t.ongoing {
                    for &(bench, q) in &benches {
                        let ns = State2 { b: b.clone(), active: pre.active, bench };
                        let key = ns.pack();
                        let id = match local.get(&key) {
                            Some(&id) => id,
                            None => {
                                let id = self.intern(ns);
                                local.insert(key, id);
                                id
                            }
                        };
                        succ.push((id, p * q));
                    }
                }
                let mut cell_terminal = 0.0;
                for (b, o, p) in &t.ended {
                    match self.post_turn(&pre, b.clone(), *o) {
                        Next::State(ns) => {
                            let active = ns.active;
                            let benches = self.snap_bench(active, ns.bench);
                            self.snap_battle(active, ns.b, *p, |nb, q| {
                                for &(bench, r) in &benches {
                                    let ns = State2 { b: nb.clone(), active, bench };
                                    acc.entry(ns.pack()).or_insert_with(|| (ns, 0.0)).1 += q * r;
                                }
                            });
                        }
                        Next::End(Outcome::Win(0)) => cell_terminal += p,
                        Next::End(Outcome::Draw) => cell_terminal += self.draw_value * p,
                        Next::End(_) => {}
                    }
                }
                self.stats.transitions += (t.ongoing.len() * benches.len() + acc.len()) as u64;
                for (key, (ns, p)) in acc.drain() {
                    let id = match local.get(&key) {
                        Some(&id) => id,
                        None => {
                            let id = self.intern(ns);
                            local.insert(key, id);
                            id
                        }
                    };
                    succ.push((id, p));
                }
                terminal.push(cell_terminal);
                start.push(succ.len() as u32);
            }
        }
        Expansion { rows, cols, terminal, start, succ, ends: Vec::new() }
    }

    fn fill_matrix(&self, x: &Expansion, m: &mut [f64; MAX_CELLS]) {
        let g = self.discount;
        for (k, out) in m.iter_mut().enumerate().take(x.terminal.len()) {
            let next: f64 = x.cell(k).iter().map(|&(s, p)| p * self.value[s as usize]).sum();
            *out = if g == 1.0 {
                x.terminal[k] + next
            } else {
                let mass: f64 = x.cell(k).iter().map(|&(_, p)| p).sum();
                x.terminal[k] + g * next + (1.0 - g) * 0.5 * mass
            };
        }
    }

    /// The matrix game at an expanded state under the current values (row-major).
    pub fn matrix(&self, x: &Expansion) -> Vec<f64> {
        let mut m = [0.0; MAX_CELLS];
        self.fill_matrix(x, &mut m);
        m[..x.terminal.len()].to_vec()
    }

    fn state_value(&self, x: &Expansion) -> f64 {
        let mut m = [0.0; MAX_CELLS];
        self.fill_matrix(x, &mut m);
        nash::value(&m[..x.terminal.len()], x.rows.len(), x.cols.len())
    }

    /// Solves the endgame from `root`: side 0's win probability plus `draw_value` times the
    /// probability of a double KO, or None if the state cap was hit.
    pub fn try_solve(&mut self, root: &State2) -> Option<f64> {
        let root = self.canonical(root.clone());
        let rid = self.intern(root);
        if self.solved[rid as usize] {
            return Some(self.value[rid as usize]);
        }
        self.tarjan(rid);
        self.stats.states = self.states.len();
        if self.aborted { None } else { Some(self.value[rid as usize]) }
    }

    pub fn solve(&mut self, root: &State2) -> f64 {
        self.try_solve(root).expect("state limit exceeded")
    }

    /// Tarjan's algorithm as in `Solver::tarjan`: each component is solved as soon as it is
    /// complete, and expansions are kept only for states still on the stack.
    fn tarjan(&mut self, root: u32) {
        let mut index: Vec<u32> = Vec::new();
        let mut low: Vec<u32> = Vec::new();
        let mut on_stack: Vec<bool> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut expansions: FxHashMap<u32, Expansion> = FxHashMap::default();
        let mut counter = 0u32;
        let mut dfs: Vec<(u32, Vec<u32>, usize)> = Vec::new();

        macro_rules! visit {
            ($s:expr) => {{
                let s = $s;
                let x = self.expand(s);
                let n = self.states.len();
                if index.len() < n {
                    index.resize(n, UNVISITED);
                    low.resize(n, UNVISITED);
                    on_stack.resize(n, false);
                }
                index[s as usize] = counter;
                low[s as usize] = counter;
                counter += 1;
                stack.push(s);
                on_stack[s as usize] = true;
                let mut succ: Vec<u32> = x.succ.iter().map(|p| p.0).collect();
                succ.sort_unstable();
                succ.dedup();
                expansions.insert(s, x);
                dfs.push((s, succ, 0));
            }};
        }

        visit!(root);
        while let Some(frame) = dfs.last_mut() {
            if self.states.len() > self.max_states {
                self.aborted = true;
                return;
            }
            let s = frame.0;
            if frame.2 < frame.1.len() {
                let t = frame.1[frame.2];
                frame.2 += 1;
                if self.solved[t as usize] {
                    continue;
                }
                if (t as usize) >= index.len() || index[t as usize] == UNVISITED {
                    visit!(t);
                } else if on_stack[t as usize] {
                    low[s as usize] = low[s as usize].min(index[t as usize]);
                }
                continue;
            }
            dfs.pop();
            if let Some(parent) = dfs.last() {
                let p = parent.0 as usize;
                low[p] = low[p].min(low[s as usize]);
            }
            if low[s as usize] == index[s as usize] {
                let mut comp = Vec::new();
                loop {
                    let t = stack.pop().unwrap();
                    on_stack[t as usize] = false;
                    comp.push(t);
                    if t == s {
                        break;
                    }
                }
                let xs: Vec<Expansion> = comp.iter().map(|t| expansions.remove(t).unwrap()).collect();
                self.solve_component(&comp, &xs);
            }
        }
    }

    /// One pass for a single acyclic state, Gauss-Seidel value iteration for a cycle. Switching
    /// back and forth makes cycles whose matrix games stay mixed, and there the simplex round-off
    /// (around 1e-12) can keep the residual just above `tol` forever: iteration also stops once the
    /// residual is below `noise` and has stopped shrinking.
    fn solve_component(&mut self, comp: &[u32], xs: &[Expansion]) {
        self.stats.sccs += 1;
        self.stats.largest_scc = self.stats.largest_scc.max(comp.len());
        let self_loop = comp.len() == 1 && xs[0].succ.iter().any(|&(t, _)| t == comp[0]);
        if comp.len() == 1 && !self_loop {
            self.value[comp[0] as usize] = self.state_value(&xs[0]);
            self.solved[comp[0] as usize] = true;
            return;
        }
        // Gauss-Seidel restricted to a worklist: a state is re-evaluated only after one of its
        // successors moved by more than `tol`. Each round ends with a full sweep that checks the
        // residual (and catches changes too small to propagate).
        let n = comp.len();
        let pos: FxHashMap<u32, usize> = comp.iter().enumerate().map(|(k, &s)| (s, k)).collect();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (k, x) in xs.iter().enumerate() {
            for &(t, _) in &x.succ {
                if let Some(&j) = pos.get(&t) {
                    if preds[j].last() != Some(&k) {
                        preds[j].push(k);
                    }
                }
            }
        }
        let mut queue: std::collections::VecDeque<usize> = (0..n).collect();
        let mut queued = vec![true; n];
        let (mut sweeps, mut evals) = (0usize, 0usize);
        let mut residual;
        let (mut best, mut improved) = (f64::INFINITY, 0);
        loop {
            // Worklist phase, bounded so round-off churn can't run forever.
            let budget = evals + 20 * n;
            while let Some(k) = queue.pop_front() {
                queued[k] = false;
                let s = comp[k] as usize;
                let v = self.state_value(&xs[k]);
                let d = (v - self.value[s]).abs();
                self.value[s] = v;
                evals += 1;
                if d > self.tol {
                    for &p in &preds[k] {
                        if !queued[p] {
                            queued[p] = true;
                            queue.push_back(p);
                        }
                    }
                }
                if evals >= budget {
                    break;
                }
            }
            // Verification sweep.
            residual = 0.0f64;
            for (k, &s) in comp.iter().enumerate() {
                let v = self.state_value(&xs[k]);
                let d = (v - self.value[s as usize]).abs();
                residual = residual.max(d);
                self.value[s as usize] = v;
                if d > self.tol && !queued[k] {
                    queued[k] = true;
                    queue.push_back(k);
                }
            }
            evals += n;
            sweeps += 1;
            if residual < 0.5 * best {
                (best, improved) = (residual, sweeps);
            }
            if residual < self.tol || evals >= self.max_sweeps * n || (residual < self.noise && sweeps - improved > 5) {
                break;
            }
        }
        let sweeps = evals / n;
        self.stats.sweeps += sweeps;
        self.stats.max_residual = self.stats.max_residual.max(residual);
        for &s in comp {
            self.solved[s as usize] = true;
        }
    }

    /// Optimal mixed strategies at a solved state (rows and columns are action codes).
    pub fn policy(&mut self, st: &State2) -> (Expansion, nash::GameSolution) {
        let st = self.canonical(st.clone());
        let id = self.intern(st);
        let x = self.expand(id);
        let m = self.matrix(&x);
        let sol = nash::solve(&m, x.rows.len(), x.cols.len());
        (x, sol)
    }
}

/// Result of `solve_ladder`.
pub struct EndgameEval {
    pub value: f64,
    pub states: usize,
    pub secs: f64,
    pub precision: EndgamePrecision,
    pub largest_scc: usize,
    pub sweeps: usize,
    pub paths: u64,
    pub transitions: u64,
    /// Root mixed strategies: (action name, probability) for each side.
    pub opening: [Vec<(String, f64)>; 2],
}

/// Solves from `root` (None: `EndgameSolver::initial`) at each precision in turn until one fits
/// within its state cap.
pub fn solve_ladder(dex: &Dex, teams: &Teams, root: Option<&State2>, ladder: &[EndgamePrecision], progress: bool) -> Option<EndgameEval> {
    let t = std::time::Instant::now();
    for &p in ladder {
        let mut s = EndgameSolver::new(dex, teams);
        s.set_precision(p);
        s.progress = progress;
        let root = root.cloned().unwrap_or_else(|| s.initial());
        if let Some(v) = s.try_solve(&root) {
            let states = s.states.len();
            let (x, sol) = s.policy(&root);
            let root = s.canonical(root);
            let opening = [
                x.rows.iter().zip(&sol.row).map(|(&c, &p)| (s.action_name(&root, 0, c), p)).collect(),
                x.cols.iter().zip(&sol.col).map(|(&c, &p)| (s.action_name(&root, 1, c), p)).collect(),
            ];
            let st = &s.stats;
            return Some(EndgameEval {
                value: v,
                states,
                secs: t.elapsed().as_secs_f64(),
                precision: p,
                largest_scc: st.largest_scc,
                sweeps: st.sweeps,
                paths: st.paths,
                transitions: st.transitions,
                opening,
            });
        }
    }
    None
}
