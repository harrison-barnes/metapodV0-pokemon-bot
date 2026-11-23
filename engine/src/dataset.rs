//! The 1v1 dataset for the 6v6 agent (see docs/DATASET.md): solved positions with outcome
//! statistics, the value of every pair of move choices, and both sides' optimal mixes.
//!
//! Each set pair is solved once from many fresh-entry roots (HP, major status and clause flags
//! carried in from the rest of a 6v6 battle), then mid-battle positions are sampled by playing
//! the equilibrium from random roots. Rows are fixed-width little-endian records (`ROW_BYTES`)
//! described by `schema()`.

use crate::chance::{Rng, Sampler};
use crate::data::Dex;
use crate::engine::Engine;
use crate::meta::{Precision, Set};
use crate::solver::{EqStats, Solver};
use crate::state::*;
use rustc_hash::FxHashSet;
use std::mem::offset_of;

/// Fallbacks for one set pair, finest first. A whole pair is solved at one setting. All its
/// fresh-entry roots share one solver, which makes the game several times larger than a single
/// full-HP solve; at about 600 bytes a state with statistics tracking, run few enough pairs at once
/// that a handful of 10M-state solves fit in memory.
pub const LADDER: &[Precision] = &[
    Precision::new(3, 48, 5_000_000),
    Precision::new(3, 24, 8_000_000),
    Precision::new(1, 12, 10_000_000),
    Precision::new(1, 6, 12_000_000),
    Precision::new(1, 3, 16_000_000),
];

/// HP fractions and major statuses a Pokemon can bring into a 1v1 position.
pub const HP_FRACTIONS: &[f64] = &[1.0, 0.75, 0.5, 0.25, 0.1];
/// None, PAR, BRN, PSN, asleep with 2 turns left (as after Rest), asleep with 5 turns left.
pub const STATUSES: &[u8] = &[0, PAR, BRN, PSN, 2, 5];
pub const CLAUSES: &[u8] = &[0, SLEEP_BLOCKED, FREEZE_BLOCKED, SLEEP_BLOCKED | FREEZE_BLOCKED];

pub const KIND_ENTRY: u8 = 0;
pub const KIND_MID: u8 = 1;
/// `Row::choices` marker: this side doesn't choose a move this turn (asleep, recharging, ...).
pub const NO_CHOICE: u8 = 254;
pub const UNUSED: u8 = 255;

pub struct Config {
    pub ladder: &'static [Precision],
    pub discount: f64,
    pub draw_value: f64,
    /// Mid-battle rows wanted per set pair.
    pub mid_rows: usize,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config { ladder: LADDER, discount: 1.0 - 1e-7, draw_value: 0.5, mid_rows: 500, seed: 1 }
    }
}

/// One labelled position.
pub struct Row {
    pub a: u16,
    pub b: u16,
    pub kind: u8,
    pub precision: Precision,
    pub battle: Battle,
    pub value: f64,
    pub eq: EqStats,
    /// Move slot of each matrix row (side 0) and column (side 1); `NO_CHOICE` or `UNUSED`.
    pub choices: [[u8; 4]; 2],
    /// q[i][j]: value if side 0 picks row i and side 1 column j, then both play the equilibrium.
    pub q: [[f64; 4]; 4],
    pub mix: [[f64; 4]; 2],
}

pub const ROW_BYTES: usize = 244;

impl Row {
    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&self.a.to_le_bytes());
        out.extend_from_slice(&self.b.to_le_bytes());
        out.push(self.kind);
        out.push(self.precision.buckets);
        out.extend_from_slice(&self.precision.hp_levels.to_le_bytes());
        let packed = self.battle.pack();
        let bytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        out.extend_from_slice(&bytes[..BATTLE_BYTES]);
        out.extend_from_slice(&[0, 0]);
        let e = &self.eq;
        for x in [self.value, e.p_win, e.p_double_ko, e.p_endless, e.hp[0], e.hp[1], e.turns] {
            out.extend_from_slice(&(x as f32).to_le_bytes());
        }
        out.extend_from_slice(&self.choices[0]);
        out.extend_from_slice(&self.choices[1]);
        for x in self.q.iter().flatten().chain(self.mix.iter().flatten()) {
            out.extend_from_slice(&(*x as f32).to_le_bytes());
        }
        debug_assert_eq!(out.len() - start, ROW_BYTES);
    }
}

const BATTLE_BYTES: usize = std::mem::size_of::<Battle>();

/// numpy-style description of a row and of the battle bytes inside it, for the Python loader.
pub fn schema() -> serde_json::Value {
    let side = |name: &str, off: usize, ty: &str, n: usize| serde_json::json!([name, off, ty, n]);
    serde_json::json!({
        "row_bytes": ROW_BYTES,
        "row": [
            ["a", 0, "<u2", 1], ["b", 2, "<u2", 1], ["kind", 4, "u1", 1], ["buckets", 5, "u1", 1],
            ["hp_levels", 6, "<u2", 1], ["battle", 8, "u1", BATTLE_BYTES],
            ["value", 112, "<f4", 1], ["p_win", 116, "<f4", 1], ["p_double_ko", 120, "<f4", 1],
            ["p_endless", 124, "<f4", 1], ["hp_left", 128, "<f4", 2], ["turns", 136, "<f4", 1],
            ["choices", 140, "u1", 8], ["q", 148, "<f4", 16], ["mix", 212, "<f4", 8]
        ],
        "battle": {
            "side_bytes": std::mem::size_of::<Side>(),
            "damage": [offset_of!(Battle, damage), "<u2", 1],
            "side": [
                side("hp", offset_of!(Side, hp), "<u2", 1),
                side("stats", offset_of!(Side, stats), "<u2", 4),
                side("unmod", offset_of!(Side, unmod), "<u2", 4),
                side("bide_dmg", offset_of!(Side, bide_dmg), "<u2", 1),
                side("status", offset_of!(Side, status), "u1", 1),
                side("mods", offset_of!(Side, mods), "u1", 6),
                side("bs1", offset_of!(Side, bs1), "u1", 1),
                side("bs2", offset_of!(Side, bs2), "u1", 1),
                side("bs3", offset_of!(Side, bs3), "u1", 1),
                side("confused", offset_of!(Side, confused), "u1", 1),
                side("toxic", offset_of!(Side, toxic), "u1", 1),
                side("disabled", offset_of!(Side, disabled), "u1", 1),
                side("disabled_move", offset_of!(Side, disabled_move), "u1", 1),
                side("attacks_left", offset_of!(Side, attacks_left), "u1", 1),
                side("sub_hp", offset_of!(Side, sub_hp), "u1", 1),
                side("selected", offset_of!(Side, selected), "u1", 1),
                side("ram", offset_of!(Side, ram), "u1", 4),
                side("types", offset_of!(Side, types), "u1", 2),
                side("moves", offset_of!(Side, moves), "u1", 4),
                side("species", offset_of!(Side, species), "u1", 1),
                side("sleep_used", offset_of!(Side, sleep_used), "u1", 1),
                side("clauses", offset_of!(Side, clauses), "u1", 1)
            ]
        }
    })
}

/// Fresh-entry roots for a matchup, deduplicated after canonicalisation (clause flags that can't
/// matter collapse, as do statuses a type can't have).
pub fn entry_roots(dex: &Dex, eng: &Engine, m: &Matchup) -> Vec<Battle> {
    let mut seen = FxHashSet::default();
    let mut out = Vec::new();
    let allowed = |s: usize, st: u8| {
        let types = dex.species[m.mons[s].species as usize].types;
        !(st == PSN && types.contains(&crate::data::ty::POISON) || st == BRN && types.contains(&crate::data::ty::FIRE))
    };
    for &f0 in HP_FRACTIONS {
        for &f1 in HP_FRACTIONS {
            let hp = [f0, f1].iter().enumerate().map(|(s, f)| ((m.mons[s].max_hp as f64 * f).round() as u16).max(1)).collect::<Vec<_>>();
            for &s0 in STATUSES.iter().filter(|&&s| allowed(0, s)) {
                for &s1 in STATUSES.iter().filter(|&&s| allowed(1, s)) {
                    for &c0 in CLAUSES {
                        for &c1 in CLAUSES {
                            let mut b = m.entry(dex, [hp[0], hp[1]], [s0, s1], [c0, c1]);
                            eng.canonicalize(&mut b);
                            if seen.insert(b.pack()) {
                                out.push(b);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Where one pair's time went, in seconds.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Timing {
    pub solve: f64,
    pub label: f64,
    pub mid: f64,
}

/// Solves one set pair and returns its rows, or None if every precision setting hit its cap.
pub fn generate_pair(dex: &Dex, rules: &Rules, sets: &[Set], a: usize, b: usize, cfg: &Config) -> Option<(Vec<Row>, usize, Timing)> {
    let m = Matchup { mons: [sets[a].mon(dex, rules), sets[b].mon(dex, rules)], rules: rules.clone() };
    for &p in cfg.ladder {
        let mut eng = Engine::new(dex, &m);
        eng.roll_buckets = p.buckets;
        let roots = entry_roots(dex, &eng, &m);
        let mut s = Solver::new(eng);
        s.max_states = p.max_states;
        s.set_hp_levels(p.hp_levels);
        s.discount = cfg.discount;
        s.draw_value = cfg.draw_value;
        s.track_stats = true;
        let t = std::time::Instant::now();
        if !roots.iter().all(|r| s.try_solve(r).is_some()) {
            continue;
        }
        let mut timing = Timing { solve: t.elapsed().as_secs_f64(), ..Timing::default() };
        let t = std::time::Instant::now();
        let mut rows = Vec::with_capacity(roots.len() + cfg.mid_rows);
        for r in &roots {
            rows.push(label(&mut s, r, a, b, KIND_ENTRY, p));
        }
        timing.label = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        sample_mid(&mut s, &roots, a, b, p, cfg, &mut rows);
        timing.mid = t.elapsed().as_secs_f64();
        return Some((rows, s.states.len(), timing));
    }
    None
}

/// Labels a solved position: statistics, the move-pair values and the optimal mixes.
fn label(s: &mut Solver, b: &Battle, a: usize, bi: usize, kind: u8, p: Precision) -> Row {
    let (x, sol) = s.policy(b);
    let id = s.id_of(b).expect("labelled positions are solved");
    let mut choices = [[UNUSED; 4]; 2];
    let mut q = [[0.0; 4]; 4];
    let mut mix = [[0.0; 4]; 2];
    let m = s.matrix_of(&x);
    for (side, (opts, strat)) in [(&x.rows, &sol.row), (&x.cols, &sol.col)].into_iter().enumerate() {
        assert!(opts.len() <= 4, "more than 4 choices");
        let chooses = s.eng.needs_choice(b, side);
        for (k, &c) in opts.iter().enumerate() {
            choices[side][k] = if chooses { c } else { NO_CHOICE };
            mix[side][k] = strat[k];
        }
    }
    for i in 0..x.rows.len() {
        for j in 0..x.cols.len() {
            q[i][j] = m[i * x.cols.len() + j];
        }
    }
    Row { a: a as u16, b: bi as u16, kind, precision: p, battle: s.states[id as usize].clone(), value: s.value[id as usize], eq: s.eq[id as usize], choices, q, mix }
}

/// Plays the equilibrium from random roots and labels a random selection of the positions visited,
/// so mid-battle rows follow how often optimal play actually reaches them.
fn sample_mid(s: &mut Solver, roots: &[Battle], a: usize, b: usize, p: Precision, cfg: &Config, rows: &mut Vec<Row>) {
    let mut rng = Rng::new(cfg.seed ^ ((a as u64) << 32 | b as u64));
    let mut picked = FxHashSet::default();
    const KEEP: f64 = 0.3; // chance of labelling each visited position, to spread rows across battles
    for _ in 0..cfg.mid_rows * 20 {
        if rows.len() >= roots.len() + cfg.mid_rows {
            break;
        }
        let mut cur = roots[rng.below(roots.len() as u64) as usize].clone();
        for _turn in 0..200 {
            let (x, sol) = s.policy(&cur);
            let pick = |strat: &[f64], rng: &mut Rng| {
                let mut r = rng.f64();
                for (k, &w) in strat.iter().enumerate() {
                    if r < w {
                        return k;
                    }
                    r -= w;
                }
                strat.len() - 1
            };
            let choice = [x.rows[pick(&sol.row, &mut rng)], x.cols[pick(&sol.col, &mut rng)]];
            let mut ch = Sampler { rng: rng.clone() };
            let (mut nb, o) = s.eng.turn(&cur, choice, &mut ch);
            rng = ch.rng;
            if o != Outcome::Ongoing {
                break;
            }
            s.eng.canonicalize(&mut nb);
            cur = s.sample_snap(nb, &mut rng);
            if s.id_of(&cur).is_none() {
                break; // can't happen when the snap matches the solver's; stop rather than guess
            }
            if rng.f64() < KEEP && picked.insert(cur.pack()) {
                rows.push(label(s, &cur, a, b, KIND_MID, p));
                if rows.len() >= roots.len() + cfg.mid_rows {
                    break;
                }
            }
        }
    }
}
