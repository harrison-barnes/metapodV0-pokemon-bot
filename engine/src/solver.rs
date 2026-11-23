//! Exact solver for 1v1 battles as a two-player zero-sum stochastic game with simultaneous moves.
//!
//! Every state reachable from the start is enumerated. Each turn is a matrix game (rows: side 0's
//! move, columns: side 1's move) whose entries are expected values of the successor states. The value
//! of a state is side 0's probability of winning plus half the probability of a draw, under optimal
//! (Nash) play by both sides.
//!
//! States are processed as strongly connected components (Tarjan), in reverse topological order, so
//! acyclic parts of the game are solved in a single pass. Components with cycles (healing, misses,
//! sleep, stat stages bouncing) are solved by Gauss-Seidel value iteration. A cycle that can never be
//! left counts as a draw.

use crate::chance::Enumerator;
use crate::engine::{Engine, MidTurn};
use crate::nash;
use crate::state::{Battle, Outcome, Packed};
use rustc_hash::FxHashMap;

/// A state's full turn: legal choices for each side and the outcome of every combination.
///
/// Cell `k` (row `k / cols.len()`, column `k % cols.len()`) ends the battle with weighted
/// probability `terminal[k]` (win 1, draw 0.5) and otherwise moves to `succ[start[k]..start[k + 1]]`.
/// Flat storage keeps a state to a handful of allocations.
#[derive(Clone)]
pub struct Expansion {
    pub rows: Vec<u8>,
    pub cols: Vec<u8>,
    pub terminal: Vec<f64>,
    pub start: Vec<u32>,
    pub succ: Vec<(u32, f64)>,
    /// Per cell, when `Solver::track_stats` is set: probability mass of the battle ending this turn
    /// with side 0 winning, ending in a double KO, and the expected HP fraction of each side at that
    /// end (mass-weighted, so a fainted side contributes 0).
    pub ends: Vec<End>,
}

/// How a cell's turn can end the battle (see `Expansion::ends`).
#[derive(Clone, Copy, Default)]
pub struct End {
    pub win: f64,
    pub lose: f64,
    pub double_ko: f64,
    pub hp: [f64; 2],
}

/// Outcome statistics of a state under the solved equilibrium strategies (see `Solver::track_stats`).
#[derive(Clone, Copy, Default, Debug)]
pub struct EqStats {
    pub p_win: f64,
    pub p_double_ko: f64,
    /// Probability the battle never ends (counted as a draw in the value).
    pub p_endless: f64,
    /// Expected final HP fraction of each side (0 if it fainted).
    pub hp: [f64; 2],
    /// Expected number of remaining turns, given that the battle ends.
    pub turns: f64,
}

impl Expansion {
    pub fn cell(&self, k: usize) -> &[(u32, f64)] {
        &self.succ[self.start[k] as usize..self.start[k + 1] as usize]
    }
}

pub struct SolveStats {
    pub states: usize,
    pub sccs: usize,
    pub largest_scc: usize,
    pub sweeps: usize,
    pub max_residual: f64,
    pub paths: u64,
    pub transitions: u64,
}

pub struct Solver<'a> {
    pub eng: Engine<'a>,
    ids: FxHashMap<Packed, u32>,
    pub states: Vec<Battle>,
    pub value: Vec<f64>,
    solved: Vec<bool>,
    /// Convergence tolerance for cyclic components.
    pub tol: f64,
    pub max_sweeps: usize,
    /// Evaluate "both sides pick uniformly random legal moves" instead of optimal play (for testing).
    pub uniform: bool,
    /// Give up (solve returns None) once this many states have been discovered.
    pub max_states: usize,
    /// HP grid per side (sorted, starting at 1 and ending at max HP). When set, HP after each turn is
    /// rounded to the two neighbouring grid points with probabilities that preserve its expected value.
    /// This bounds the number of distinct HP values (an approximation; empty means exact).
    hp_grid: [Vec<u16>; 2],
    aborted: bool,
    /// Enumerate each turn in two halves, merging identical halfway positions (see
    /// `Engine::start_turn`). Pays off when turns branch a lot (exact or 3-bucket damage rolls);
    /// with single-bucket rolls the extra hashing costs more than it saves.
    pub split_turns: bool,
    /// Per-turn discount towards the endless-battle value 0.5 (1.0 = none). In a game that can
    /// cycle forever, the undiscounted matrix games can be indifferent between making progress and
    /// stalling, so strategies read from them may never realise the value. A discount just below 1
    /// makes the side ahead strictly prefer winning sooner; its stationary strategies then achieve
    /// the undiscounted value up to a tiny error (see tests/eqstats.rs), and the values themselves
    /// move by about (1 - discount) times the battle length.
    pub discount: f64,
    /// Value of a battle ending with both sides fainted (Explosion, recoil). 0.5 treats it as a draw;
    /// inside a 6v6 battle a double KO can be worth more or less than that to side 0.
    pub draw_value: f64,
    /// Also compute `EqStats` for every state (outcome split, final HP, length) under the solved
    /// equilibrium strategies.
    pub track_stats: bool,
    pub eq: Vec<EqStats>,
    pub stats: SolveStats,
}

const UNVISITED: u32 = u32::MAX;
const MAX_CELLS: usize = nash::MAX_CHOICES * nash::MAX_CHOICES;

impl<'a> Solver<'a> {
    pub fn new(eng: Engine<'a>) -> Solver<'a> {
        let split_turns = eng.roll_buckets != 1;
        Solver {
            eng,
            ids: FxHashMap::default(),
            states: Vec::new(),
            value: Vec::new(),
            solved: Vec::new(),
            tol: 1e-12,
            max_sweeps: 100_000,
            uniform: false,
            max_states: usize::MAX,
            hp_grid: [Vec::new(), Vec::new()],
            aborted: false,
            split_turns,
            draw_value: 0.5,
            discount: 1.0,
            track_stats: false,
            eq: Vec::new(),
            stats: SolveStats { states: 0, sccs: 0, largest_scc: 0, sweeps: 0, max_residual: 0.0, paths: 0, transitions: 0 },
        }
    }

    /// Approximate HP with `levels` evenly spaced values per side (0 restores exact HP).
    pub fn set_hp_levels(&mut self, levels: u16) {
        for s in 0..2 {
            let max = self.eng.m.mons[s].max_hp;
            self.hp_grid[s] = if levels == 0 || levels >= max {
                Vec::new()
            } else {
                let mut g: Vec<u16> = (0..=levels).map(|k| ((k as u32 * max as u32 + levels as u32 - 1) / levels as u32).max(1) as u16).collect();
                g.dedup();
                g
            };
        }
    }

    /// Stochastic rounding of both sides' HP onto the grid: up to four weighted states.
    fn snap(&self, b: Battle, p: f64, out: &mut FxHashMap<Packed, (Battle, f64)>) {
        // For each side: (low HP, high HP, probability of rounding up).
        let split = |s: usize| -> (u16, u16, f64) {
            let g = &self.hp_grid[s];
            let hp = b.sides[s].hp;
            if g.is_empty() {
                return (hp, hp, 0.0);
            }
            let k = g.partition_point(|&x| x <= hp); // g[k-1] <= hp < g[k]
            if k == 0 || g[k - 1] == hp || k == g.len() {
                return (hp, hp, 0.0);
            }
            let (lo, hi) = (g[k - 1], g[k]);
            (lo, hi, (hp - lo) as f64 / (hi - lo) as f64)
        };
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
                out.entry(nb.pack()).or_insert_with(|| (nb, 0.0)).1 += p * p0 * p1;
            }
        }
    }

    /// Rounds HP onto the grid the way the solver does, sampling the rounding direction.
    pub fn sample_snap(&self, b: Battle, rng: &mut crate::chance::Rng) -> Battle {
        let mut out = FxHashMap::default();
        self.snap(b, 1.0, &mut out);
        let mut r = rng.f64();
        let mut last = None;
        for (_, (b, p)) in out {
            if r < p {
                return b;
            }
            r -= p;
            last = Some(b);
        }
        last.unwrap()
    }

    /// Solved value of a state, if it was reached during solving.
    pub fn value_of(&self, b: &Battle) -> Option<f64> {
        self.ids.get(&b.pack()).map(|&id| self.value[id as usize])
    }

    fn intern(&mut self, b: Battle) -> u32 {
        let key = b.pack();
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.states.len() as u32;
        self.ids.insert(key, id);
        self.states.push(b);
        self.value.push(0.5);
        self.solved.push(false);
        if self.track_stats {
            self.eq.push(EqStats::default());
        }
        id
    }

    pub fn id_of(&self, b: &Battle) -> Option<u32> {
        self.ids.get(&b.pack()).copied()
    }

    /// All transitions out of a state, with successors interned.
    pub fn expand(&mut self, id: u32) -> Expansion {
        let b = self.states[id as usize].clone();
        let mut rows = self.eng.legal_choices(&b, 0);
        let mut cols = self.eng.legal_choices(&b, 1);
        if rows.is_empty() {
            rows.push(0);
        }
        if cols.is_empty() {
            cols.push(0);
        }
        let n = rows.len() * cols.len();
        let mut terminal = Vec::with_capacity(n);
        let mut ends = Vec::with_capacity(if self.track_stats { n } else { 0 });
        let mut start = Vec::with_capacity(n + 1);
        let mut succ = Vec::new();
        start.push(0);
        let mut acc: FxHashMap<Packed, (Battle, f64)> = FxHashMap::default();
        let mut snapped: FxHashMap<Packed, (Battle, f64)> = FxHashMap::default();
        let mut mids: FxHashMap<MidTurn, f64> = FxHashMap::default();
        let mut ended: Vec<(Battle, Outcome, f64)> = Vec::new();
        let grid = !self.hp_grid[0].is_empty() || !self.hp_grid[1].is_empty();
        for &r in &rows {
            for &c in &cols {
                acc.clear();
                let mut cell_terminal = 0.0;
                let mut end = End::default();
                // First half of the turn, merging identical halfway positions; then the second half
                // from each distinct one.
                mids.clear();
                ended.clear();
                let mut e = Enumerator::new();
                while !self.split_turns {
                    let (nb, o) = self.eng.turn(&b, [r, c], &mut e);
                    ended.push((nb, o, e.prob()));
                    self.stats.paths += 1;
                    if !e.advance() {
                        break;
                    }
                }
                while self.split_turns {
                    match self.eng.start_turn(&b, [r, c], &mut e) {
                        Ok(mid) => *mids.entry(mid).or_insert(0.0) += e.prob(),
                        Err((nb, o)) => ended.push((nb, o, e.prob())),
                    }
                    self.stats.paths += 1;
                    if !e.advance() {
                        break;
                    }
                }
                for (mid, p1) in mids.iter() {
                    let mut e = Enumerator::new();
                    loop {
                        let (nb, o) = self.eng.finish_turn(mid, &mut e);
                        ended.push((nb, o, p1 * e.prob()));
                        self.stats.paths += 1;
                        if !e.advance() {
                            break;
                        }
                    }
                }
                for (mut nb, o, p) in ended.drain(..) {
                    if o != Outcome::Ongoing && self.track_stats {
                        for s in 0..2 {
                            end.hp[s] += p * nb.sides[s].hp as f64 / self.eng.m.mons[s].max_hp as f64;
                        }
                    }
                    match o {
                        Outcome::Ongoing => {
                            self.eng.canonicalize(&mut nb);
                            acc.entry(nb.pack()).or_insert_with(|| (nb, 0.0)).1 += p;
                        }
                        Outcome::Win(0) => {
                            cell_terminal += p;
                            end.win += p;
                        }
                        Outcome::Draw => {
                            cell_terminal += self.draw_value * p;
                            end.double_ko += p;
                        }
                        Outcome::Win(_) => end.lose += p,
                    }
                }
                if self.track_stats {
                    ends.push(end);
                }
                if grid {
                    snapped.clear();
                    for (_, (nb, p)) in acc.drain() {
                        self.snap(nb, p, &mut snapped);
                    }
                    std::mem::swap(&mut acc, &mut snapped);
                }
                self.stats.transitions += acc.len() as u64;
                for (_, (nb, p)) in acc.drain() {
                    let id = self.intern(nb);
                    succ.push((id, p));
                }
                terminal.push(cell_terminal);
                start.push(succ.len() as u32);
            }
        }
        Expansion { rows, cols, terminal, start, succ, ends }
    }

    /// The matrix game at an expanded state under the current values: entry `i * cols + j` is the
    /// value of side 0 choosing row i and side 1 column j.
    pub fn matrix_of(&self, x: &Expansion) -> Vec<f64> {
        self.matrix(x)
    }

    fn matrix(&self, x: &Expansion) -> Vec<f64> {
        let mut m = [0.0; MAX_CELLS];
        self.fill_matrix(x, &mut m);
        m[..x.terminal.len()].to_vec()
    }

    fn fill_matrix(&self, x: &Expansion, m: &mut [f64; MAX_CELLS]) {
        let g = self.discount;
        for (k, out) in m.iter_mut().enumerate().take(x.terminal.len()) {
            let next: f64 = x.cell(k).iter().map(|&(s, p)| p * self.value[s as usize]).sum();
            *out = if g == 1.0 {
                x.terminal[k] + next
            } else {
                // Continuing pulls the value towards an endless battle's 0.5 by a factor g per turn.
                let mass: f64 = x.cell(k).iter().map(|&(_, p)| p).sum();
                x.terminal[k] + g * next + (1.0 - g) * 0.5 * mass
            };
        }
    }

    fn state_value(&self, x: &Expansion) -> f64 {
        let mut m = [0.0; MAX_CELLS];
        self.fill_matrix(x, &mut m);
        let m = &m[..x.terminal.len()];
        if self.uniform {
            return m.iter().sum::<f64>() / m.len() as f64;
        }
        nash::value(m, x.rows.len(), x.cols.len())
    }

    /// Solves the game from `root`, returning side 0's value (win probability + half draw probability).
    /// Panics if the state limit is hit; use `try_solve` when a limit is set.
    pub fn solve(&mut self, root: &Battle) -> f64 {
        self.try_solve(root).expect("state limit exceeded")
    }

    pub fn try_solve(&mut self, root: &Battle) -> Option<f64> {
        let mut root = root.clone();
        self.eng.canonicalize(&mut root);
        let rid = self.intern(root);
        // Several roots can share one solver: everything solved for earlier roots is reused.
        if self.solved[rid as usize] {
            return Some(self.value[rid as usize]);
        }
        self.tarjan(rid);
        if self.aborted {
            self.stats.states = self.states.len();
            return None;
        }
        self.stats.states = self.states.len();
        if self.aborted { None } else { Some(self.value[rid as usize]) }
    }

    fn tarjan(&mut self, root: u32) {
        let mut index: Vec<u32> = Vec::new();
        let mut low: Vec<u32> = Vec::new();
        let mut on_stack: Vec<bool> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut expansions: FxHashMap<u32, Expansion> = FxHashMap::default();
        let mut counter = 0u32;
        // DFS frames: (state, distinct successors, next successor position)
        let mut dfs: Vec<(u32, Vec<u32>, usize)> = Vec::new();

        let grow = |v: &mut Vec<u32>, n: usize, fill: u32| {
            if v.len() < n {
                v.resize(n, fill);
            }
        };

        macro_rules! visit {
            ($s:expr) => {{
                let s = $s;
                let x = self.expand(s);
                let n = self.states.len();
                grow(&mut index, n, UNVISITED);
                grow(&mut low, n, UNVISITED);
                if on_stack.len() < n {
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

    /// `comp` is in reverse discovery order: states found deepest (closest to the end) come first.
    fn solve_component(&mut self, comp: &[u32], xs: &[Expansion]) {
        self.stats.sccs += 1;
        self.stats.largest_scc = self.stats.largest_scc.max(comp.len());
        let self_loop = comp.len() == 1 && xs[0].succ.iter().any(|&(t, _)| t == comp[0]);
        if comp.len() == 1 && !self_loop {
            self.value[comp[0] as usize] = self.state_value(&xs[0]);
            self.solved[comp[0] as usize] = true;
            self.component_stats(comp, xs);
            return;
        }
        let mut sweeps = 0;
        let mut residual;
        loop {
            residual = 0.0f64;
            for (k, &s) in comp.iter().enumerate() {
                let v = self.state_value(&xs[k]);
                residual = residual.max((v - self.value[s as usize]).abs());
                self.value[s as usize] = v;
            }
            sweeps += 1;
            if residual < self.tol || sweeps >= self.max_sweeps {
                break;
            }
        }
        self.stats.sweeps += sweeps;
        self.stats.max_residual = self.stats.max_residual.max(residual);
        for &s in comp {
            self.solved[s as usize] = true;
        }
        self.component_stats(comp, xs);
    }

    /// `EqStats` for a solved component: the equilibrium strategies are fixed, so outcome
    /// probabilities, final HP and length are linear and found by the same kind of Gauss-Seidel
    /// iteration, starting from zero (mass that never reaches an end is the endless-battle mass).
    fn component_stats(&mut self, comp: &[u32], xs: &[Expansion]) {
        if !self.track_stats {
            return;
        }
        // Joint probability of each cell under the equilibrium mixes.
        let joint: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| {
                let (r, c) = (x.rows.len(), x.cols.len());
                let (row, col) = if self.uniform {
                    (vec![1.0 / r as f64; r], vec![1.0 / c as f64; c])
                } else {
                    let sol = nash::solve(&self.matrix(x), r, c);
                    (sol.row, sol.col)
                };
                (0..r * c).map(|k| row[k / c] * col[k % c]).collect()
            })
            .collect();
        // Per state: [p_win, p_double_ko, hp0, hp1, p_end, turn mass (sum of turns over ending paths)].
        let mut acc: Vec<[f64; 6]> = vec![[0.0; 6]; comp.len()];
        let pos: FxHashMap<u32, usize> = comp.iter().enumerate().map(|(k, &s)| (s, k)).collect();
        let of = |eq: &EqStats| -> [f64; 6] {
            let p_end = 1.0 - eq.p_endless;
            let turn_mass = if p_end > 0.0 { eq.turns * p_end } else { 0.0 };
            [eq.p_win, eq.p_double_ko, eq.hp[0], eq.hp[1], p_end, turn_mass]
        };
        for _ in 0..self.max_sweeps {
            let mut change = 0.0f64;
            for (k, x) in xs.iter().enumerate() {
                let mut v = [0.0; 6];
                for (cell, &w) in joint[k].iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let e = x.ends[cell];
                    let ended = e.win + e.lose + e.double_ko;
                    let mut cv = [e.win, e.double_ko, e.hp[0], e.hp[1], ended, ended];
                    for &(t, p) in x.cell(cell) {
                        let tv = match pos.get(&t) {
                            Some(&j) => acc[j],
                            None => of(&self.eq[t as usize]),
                        };
                        for i in 0..5 {
                            cv[i] += p * tv[i];
                        }
                        cv[5] += p * (tv[5] + tv[4]); // every ending path from t takes this turn too
                    }
                    for i in 0..6 {
                        v[i] += w * cv[i];
                    }
                }
                for i in 0..6 {
                    change = change.max((v[i] - acc[k][i]).abs() / if i == 5 { 1.0 + v[5] } else { 1.0 });
                }
                acc[k] = v;
            }
            if change < self.tol.max(1e-12) {
                break;
            }
        }
        for (k, &s) in comp.iter().enumerate() {
            let a = acc[k];
            self.eq[s as usize] = EqStats {
                p_win: a[0],
                p_double_ko: a[1],
                p_endless: (1.0 - a[4]).max(0.0),
                hp: [a[2], a[3]],
                turns: if a[4] > 0.0 { a[5] / a[4] } else { f64::INFINITY },
            };
        }
    }

    /// Optimal mixed strategies at a solved state.
    pub fn policy(&mut self, b: &Battle) -> (Expansion, nash::GameSolution) {
        let mut b = b.clone();
        self.eng.canonicalize(&mut b);
        let id = self.intern(b);
        let x = self.expand(id);
        let m = self.matrix(&x);
        let sol = nash::solve(&m, x.rows.len(), x.cols.len());
        (x, sol)
    }
}
