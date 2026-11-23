//! Python bindings: a vector of independent 6v6 battles (`VecEnv`) for self-play training.
//!
//! Actions are integers: 0-3 a move slot, 4 Struggle, 5-10 switch to team slot 0-5, 11 pass.
//! State comes back as numpy arrays: the active pair's raw battle bytes (decoded in Python with the
//! dataset schema), and per team member a row of `TEAM_FIELDS`.

use gen1::battle6::{max_pp, Action, Battle6, TeamMon};
use gen1::chance::{Enumerator, Rng, Sampler};
use std::collections::HashMap;
use gen1::cli::parse_mon;
use gen1::data::Dex;
use gen1::engine::STRUGGLE_SLOT;
use gen1::state::{Battle, Mon, Outcome, Rules};
use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub const N_ACTIONS: usize = 12;
const PASS: usize = 11;
const BATTLE_BYTES: usize = std::mem::size_of::<Battle>();
/// Per team member: species, move ids x4, hp, max_hp, status, pp x4, revealed, moves_revealed,
/// active, alive, slept_by_foe, present (1 for a real slot, 0 for padding).
pub const TEAM_FIELDS: usize = 18;

fn to_action(a: usize) -> Option<Action> {
    match a {
        0..=3 => Some(Action::Move(a as u8)),
        4 => Some(Action::Move(STRUGGLE_SLOT)),
        5..=10 => Some(Action::Switch((a - 5) as u8)),
        PASS => Some(Action::Pass),
        _ => None,
    }
}

fn to_index(a: Action) -> usize {
    match a {
        Action::Move(STRUGGLE_SLOT) => 4,
        Action::Move(s) => s as usize,
        Action::Switch(i) => 5 + i as usize,
        Action::Pass => PASS,
    }
}

struct Env {
    battle: Option<Battle6>,
    ch: Sampler,
}

/// N independent 6v6 battles under Smogon RBY OU rules.
#[pyclass]
struct VecEnv {
    dex: Dex,
    rules: Rules,
    envs: Vec<Env>,
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed = 1, rules = "smogon"))]
    fn new(n: usize, seed: u64, rules: &str) -> PyResult<Self> {
        // GEN1PY_RULES replaces the default ruleset for every VecEnv of the process that doesn't name one: how a
        // player (agent/showdown_player.py --engine-rules) turns on `smogon-sd` for all of its search's VecEnvs.
        let from_env = std::env::var("GEN1PY_RULES").ok().filter(|v| !v.is_empty() && rules == "smogon");
        let rules = from_env.as_deref().unwrap_or(rules);
        let rules = Rules::named(rules).ok_or_else(|| PyValueError::new_err(format!("unknown ruleset {rules}")))?;
        let envs = (0..n).map(|i| Env { battle: None, ch: Sampler { rng: Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64)) } }).collect();
        Ok(VecEnv { dex: Dex::load_default(), rules, envs })
    }

    fn __len__(&self) -> usize {
        self.envs.len()
    }

    /// Starts a new battle in env i. Each team is a list of "SPECIES:MOVE,MOVE,..." strings.
    fn reset(&mut self, i: usize, team_a: Vec<String>, team_b: Vec<String>) -> PyResult<()> {
        let parse = |t: &[String]| -> PyResult<Vec<_>> {
            if t.is_empty() || t.len() > 6 {
                return Err(PyValueError::new_err("teams need 1-6 Pokemon"));
            }
            t.iter().map(|m| parse_mon(&self.dex, m, &self.rules).map_err(PyValueError::new_err)).collect()
        };
        let teams = [parse(&team_a)?, parse(&team_b)?];
        let env = self.envs.get_mut(i).ok_or_else(|| PyValueError::new_err("env index out of range"))?;
        env.battle = Some(Battle6::new(&self.dex, &self.rules, teams));
        Ok(())
    }

    /// Plays one decision point in every running battle. `actions` is (n, 2); ignored for envs
    /// that are finished or never reset. Raises ValueError on an illegal action.
    fn step(&mut self, py: Python<'_>, actions: PyReadonlyArray2<'_, i64>) -> PyResult<()> {
        let a = actions.as_array();
        if a.shape() != [self.envs.len(), 2] {
            return Err(PyValueError::new_err(format!("actions must have shape ({}, 2)", self.envs.len())));
        }
        let acts: Vec<[usize; 2]> = (0..self.envs.len()).map(|i| [a[[i, 0]] as usize, a[[i, 1]] as usize]).collect();
        let dex = &self.dex;
        let envs = &mut self.envs;
        let errors: Vec<String> = py.detach(|| {
            envs.par_iter_mut()
                .zip(acts.par_iter())
                .enumerate()
                .filter_map(|(i, (env, act))| {
                    let b = env.battle.as_mut()?;
                    if b.result.is_some() {
                        return None;
                    }
                    let mut chosen = [Action::Pass; 2];
                    for s in 0..2 {
                        match to_action(act[s]) {
                            Some(x) if b.legal_actions(dex, s).contains(&x) => chosen[s] = x,
                            _ => return Some(format!("env {i}: illegal action {} for side {s}", act[s])),
                        }
                    }
                    b.step(dex, chosen, &mut env.ch, false);
                    None
                })
                .collect()
        });
        match errors.first() {
            Some(e) => Err(PyValueError::new_err(e.clone())),
            None => Ok(()),
        }
    }

    /// Plays one decision point in env i and returns the turn log (for inspection).
    fn step_logged(&mut self, i: usize, a0: usize, a1: usize) -> PyResult<Vec<String>> {
        let env = &mut self.envs[i];
        let b = env.battle.as_mut().ok_or_else(|| PyValueError::new_err("env not reset"))?;
        let mut chosen = [Action::Pass; 2];
        for (s, a) in [a0, a1].into_iter().enumerate() {
            chosen[s] = to_action(a).filter(|x| b.legal_actions(&self.dex, s).contains(x)).ok_or_else(|| PyValueError::new_err(format!("illegal action {a} for side {s}")))?;
        }
        Ok(b.step(&self.dex, chosen, &mut env.ch, true))
    }

    /// Lookahead search support: env dst[k] becomes a copy of `other`'s env src[k], with its own
    /// random stream (from `seed` and k), so copies of one position roll different outcomes.
    fn copy_from(&mut self, other: PyRef<'_, VecEnv>, src: Vec<usize>, dst: Vec<usize>, seed: u64) -> PyResult<()> {
        if src.len() != dst.len() {
            return Err(PyValueError::new_err("src and dst need the same length"));
        }
        for (k, (&a, &b)) in src.iter().zip(&dst).enumerate() {
            let from = other.envs.get(a).ok_or_else(|| PyValueError::new_err("src index out of range"))?;
            let to = self.envs.get_mut(b).ok_or_else(|| PyValueError::new_err("dst index out of range"))?;
            to.battle = from.battle.clone();
            to.ch = Sampler { rng: Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k as u64)) };
        }
        Ok(())
    }

    /// Exact lookahead: for each job (src, a0, a1), plays the joint action in a copy of `other`'s env
    /// src under every chance outcome (damage rolls grouped into `buckets` equally likely buckets,
    /// 0 = all 39), merges outcomes that leave identical battles, and writes them into this VecEnv's
    /// envs 0, 1, ... in job order (the rest are cleared). Returns (job index, probability) per
    /// written env. A job with more than `max_outcomes` distinct outcomes keeps the most likely ones,
    /// renormalized. Raises ValueError on an illegal action or when the outcomes don't fit.
    #[pyo3(signature = (other, jobs, buckets = 3, max_outcomes = 64))]
    fn branch(&mut self, py: Python<'_>, other: PyRef<'_, VecEnv>, jobs: Vec<(usize, usize, usize)>, buckets: u8,
              max_outcomes: usize) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let dex = &self.dex;
        let src_envs = &other.envs;
        let per_job: Vec<Result<Vec<(Battle6, f64)>, String>> = py.detach(|| {
            jobs.par_iter()
                .map(|&(src, a0, a1)| {
                    let b0 = src_envs.get(src).and_then(|e| e.battle.as_ref()).ok_or(format!("src env {src} not running"))?;
                    let mut chosen = [Action::Pass; 2];
                    for (s, a) in [a0, a1].into_iter().enumerate() {
                        match to_action(a) {
                            Some(x) if b0.legal_actions(dex, s).contains(&x) => chosen[s] = x,
                            _ => return Err(format!("env {src}: illegal action {a} for side {s}")),
                        }
                    }
                    let mut merged: HashMap<Vec<u64>, usize> = HashMap::new();
                    let mut outs: Vec<(Battle6, f64)> = Vec::new();
                    let mut e = Enumerator::new();
                    let mut paths = 0usize;
                    loop {
                        let mut b = b0.clone();
                        b.roll_buckets = buckets;
                        b.step(dex, chosen, &mut e, false);
                        b.roll_buckets = 0;
                        let p = e.prob();
                        match merged.entry(b.outcome_key()) {
                            std::collections::hash_map::Entry::Occupied(o) => outs[*o.get()].1 += p,
                            std::collections::hash_map::Entry::Vacant(v) => {
                                v.insert(outs.len());
                                outs.push((b, p));
                            }
                        }
                        paths += 1;
                        if paths > 200_000 {
                            return Err(format!("env {src}: more than 200000 chance paths"));
                        }
                        if !e.advance() {
                            break;
                        }
                    }
                    if outs.len() > max_outcomes {
                        outs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                        outs.truncate(max_outcomes);
                    }
                    let total: f64 = outs.iter().map(|x| x.1).sum();
                    Ok(outs.into_iter().map(|(b, p)| (b, p / total)).collect())
                })
                .collect()
        });
        let mut idx = Vec::new();
        let mut probs = Vec::new();
        let mut k = 0;
        for e in &mut self.envs {
            e.battle = None;
        }
        for (j, r) in per_job.into_iter().enumerate() {
            for (b, p) in r.map_err(PyValueError::new_err)? {
                let env = self.envs.get_mut(k).ok_or_else(|| PyValueError::new_err("branch: outcomes don't fit in this VecEnv"))?;
                env.battle = Some(b);
                env.ch = Sampler { rng: Rng::new(0x5EED ^ k as u64) };
                idx.push(j);
                probs.push(p);
                k += 1;
            }
        }
        Ok((idx, probs))
    }

    /// Empties every env (step then ignores them), e.g. before reusing a search buffer.
    /// Puts a battle already under way into env i, from the arrays `team_state` and `battle_bytes`
    /// return (team: (2, 6, TEAM_FIELDS), battle: BATTLE_BYTES bytes). Members not PRESENT are left
    /// out; a member with species 0 and nothing revealed is a placeholder for `set_members` to fill.
    /// The active Pokemon's HP and status are taken from `battle`. A side whose active Pokemon has
    /// fainted must send out a replacement.
    fn load(&mut self, i: usize, team: PyReadonlyArray3<'_, i64>, battle: Vec<u8>, turn: u32) -> PyResult<()> {
        let t = team.as_array();
        if t.shape() != [2, 6, TEAM_FIELDS] || battle.len() != BATTLE_BYTES {
            return Err(PyValueError::new_err(format!("need team (2, 6, {TEAM_FIELDS}) and {BATTLE_BYTES} battle bytes")));
        }
        // Side::sleep_used is a bool: every other byte of Battle is a plain integer.
        for s in 0..2 {
            if battle[s * std::mem::size_of::<gen1::state::Side>() + 48] > 1 {
                return Err(PyValueError::new_err("sleep_used byte must be 0 or 1"));
            }
        }
        // SAFETY: Battle is repr(C), padding-free, made of integers and one bool per side (checked above).
        let b: Battle = unsafe { std::ptr::read_unaligned(battle.as_ptr() as *const Battle) };
        let mut teams: [Vec<TeamMon>; 2] = [Vec::new(), Vec::new()];
        let mut active = [0usize; 2];
        for s in 0..2 {
            for j in 0..6 {
                let r = |f: usize| t[[s, j, f]];
                if r(17) == 0 {
                    continue;
                }
                if r(14) != 0 {
                    active[s] = teams[s].len();
                }
                let moves: Vec<u8> = (1..5).map(|f| r(f) as u8).collect();
                let mon = Mon::new(&self.dex, r(0) as usize, &moves, &self.rules);
                teams[s].push(TeamMon {
                    hp: r(5) as u16,
                    status: r(7) as u8,
                    slept_by_foe: r(16) != 0,
                    pp: [r(8) as u8, r(9) as u8, r(10) as u8, r(11) as u8],
                    revealed: r(12) != 0,
                    moves_revealed: r(13) as u8,
                    mon,
                });
            }
            if teams[s].is_empty() {
                return Err(PyValueError::new_err("each side needs a present Pokemon"));
            }
        }
        let need_replace = [b.sides[0].hp == 0, b.sides[1].hp == 0];
        let env = self.envs.get_mut(i).ok_or_else(|| PyValueError::new_err("env index out of range"))?;
        env.battle = Some(Battle6::from_parts(teams, active, b, &self.rules, turn, need_replace));
        Ok(())
    }

    fn clear(&mut self) {
        for e in &mut self.envs {
            e.battle = None;
        }
    }

    /// Lookahead search support: replaces hidden information with a guess. Each edit (i, s, j, set)
    /// puts the set "SPECIES:MOVE,..." in env i, side s, team slot j. A member the opponent hasn't
    /// seen is replaced outright (full HP, no status, full PP). A seen one keeps its species, HP,
    /// status and every move slot it has revealed (the set must list those moves in those slots);
    /// its unrevealed slots take the set's moves with full PP.
    fn set_members(&mut self, edits: Vec<(usize, usize, usize, String)>) -> PyResult<()> {
        for (i, s, j, set) in edits {
            let mon = parse_mon(&self.dex, &set, &self.rules).map_err(PyValueError::new_err)?;
            let dex = &self.dex;
            let b = self.envs.get_mut(i).and_then(|e| e.battle.as_mut()).ok_or_else(|| PyValueError::new_err("env not running"))?;
            let active = b.active[s] == j;
            let t = b.teams.get_mut(s).and_then(|t| t.get_mut(j)).ok_or_else(|| PyValueError::new_err("no such team member"))?;
            if !t.revealed {
                t.hp = mon.max_hp;
                t.status = 0;
                t.slept_by_foe = false;
                t.pp = mon.moves.map(|m| max_pp(dex, m));
                t.moves_revealed = 0;
                t.mon = mon;
            } else {
                if mon.species != t.mon.species {
                    return Err(PyValueError::new_err(format!("{set}: a seen Pokemon keeps its species")));
                }
                for k in 0..4 {
                    if t.moves_revealed >> k & 1 == 1 {
                        if mon.moves[k] != t.mon.moves[k] {
                            return Err(PyValueError::new_err(format!("{set}: slot {k} was revealed with another move")));
                        }
                    } else {
                        t.mon.moves[k] = mon.moves[k];
                        t.pp[k] = max_pp(dex, mon.moves[k]);
                    }
                }
                // (a transformed Pokemon keeps the moves it copied: its own set is what it has on the bench)
                if active && b.b.sides[s].bs3 & gen1::state::TRANSFORMED == 0 {
                    b.b.sides[s].moves = t.mon.moves;
                }
            }
        }
        Ok(())
    }

    /// (n, 2, N_ACTIONS) bool: legal actions for each side (all false when finished).
    fn legal_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let mut m = Array3::<bool>::from_elem((self.envs.len(), 2, N_ACTIONS), false);
        for (i, env) in self.envs.iter().enumerate() {
            if let Some(b) = &env.battle {
                for s in 0..2 {
                    for a in b.legal_actions(&self.dex, s) {
                        m[[i, s, to_index(a)]] = true;
                    }
                }
            }
        }
        m.into_pyarray(py)
    }

    /// (n,) int8: -1 running (or never reset), 0 or 1 the winner, 2 a draw.
    fn results<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        Array1::from_iter(self.envs.iter().map(|e| match e.battle.as_ref().and_then(|b| b.result) {
            None => -1,
            Some(Outcome::Win(s)) => s as i8,
            Some(_) => 2,
        }))
        .into_pyarray(py)
    }

    /// (n,) int32 turn counts.
    fn turns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        Array1::from_iter(self.envs.iter().map(|e| e.battle.as_ref().map_or(0, |b| b.turn as i32))).into_pyarray(py)
    }

    /// (n, 2) bool: the side must send out a replacement this decision.
    fn need_replace<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        let mut m = Array2::<bool>::from_elem((self.envs.len(), 2), false);
        for (i, e) in self.envs.iter().enumerate() {
            if let Some(b) = &e.battle {
                m[[i, 0]] = b.need_replace[0];
                m[[i, 1]] = b.need_replace[1];
            }
        }
        m.into_pyarray(py)
    }

    /// (n, BATTLE_BYTES) uint8: the active pair's state, laid out as in the dataset schema.
    fn battle_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let mut m = Array2::<u8>::zeros((self.envs.len(), BATTLE_BYTES));
        for (i, e) in self.envs.iter().enumerate() {
            if let Some(b) = &e.battle {
                let bytes: Vec<u8> = b.b.pack().iter().flat_map(|w| w.to_le_bytes()).collect();
                for (k, &x) in bytes[..BATTLE_BYTES].iter().enumerate() {
                    m[[i, k]] = x;
                }
            }
        }
        m.into_pyarray(py)
    }

    /// (n, 2, 6, TEAM_FIELDS) int32: every team member's lasting state (see TEAM_FIELDS).
    /// The active Pokemon's HP and status are current.
    fn team_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<i32>> {
        let mut m = Array4::<i32>::zeros((self.envs.len(), 2, 6, TEAM_FIELDS));
        for (i, e) in self.envs.iter().enumerate() {
            let Some(b) = &e.battle else { continue };
            for s in 0..2 {
                for (k, t) in b.teams[s].iter().enumerate() {
                    let active = k == b.active[s];
                    let (hp, status) = if active { (b.b.sides[s].hp, b.b.sides[s].status) } else { (t.hp, t.status) };
                    let mut row = vec![t.mon.species as i32];
                    row.extend(t.mon.moves.iter().map(|&x| x as i32));
                    row.extend([hp as i32, t.mon.max_hp as i32, status as i32]);
                    row.extend(t.pp.iter().map(|&x| x as i32));
                    row.extend([t.revealed as i32, t.moves_revealed as i32, active as i32, (hp > 0) as i32, t.slept_by_foe as i32, 1]);
                    assert_eq!(row.len(), TEAM_FIELDS);
                    for (f, v) in row.into_iter().enumerate() {
                        m[[i, s, k, f]] = v;
                    }
                }
            }
        }
        m.into_pyarray(py)
    }
}

#[pymodule]
fn gen1py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add("N_ACTIONS", N_ACTIONS)?;
    m.add("TEAM_FIELDS", TEAM_FIELDS)?;
    m.add("PASS", PASS)?;
    Ok(())
}
