//! Tournament-level tools: move pools, movesets, and solving matchups between sets.

use crate::data::{eff, Dex};
use crate::engine::Engine;
use crate::solver::Solver;
use crate::state::{Matchup, Mon, Rules};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A Pokemon and its four (or fewer) moves.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Set {
    pub species: String,
    pub moves: Vec<String>,
}

impl Set {
    pub fn mon(&self, dex: &Dex, rules: &Rules) -> Mon {
        let si = dex.species_index(&self.species).unwrap_or_else(|| panic!("unknown species {}", self.species));
        let ids: Vec<u8> = self.moves.iter().map(|m| dex.move_id(m).unwrap_or_else(|| panic!("unknown move {m}"))).collect();
        Mon::new(dex, si, &ids, rules)
    }

    pub fn label(&self) -> String {
        format!("{}:{}", self.species, self.moves.join(","))
    }
}

/// Moves that are never considered: unsupported by the engine, banned by the rules, or with no
/// effect in a 1v1 trainer battle.
pub const EXCLUDED: &[&str] = &[
    "METRONOME", "MIRROR_MOVE", "MIMIC", // not implemented
    "EXPLOSION", "SELFDESTRUCT", // banned, as in the reference tournament
    "TELEPORT", "ROAR", "WHIRLWIND", "SPLASH", "STRUGGLE", // do nothing here
    "FOCUS_ENERGY", // bugged: quarters the critical hit rate
];

/// Effects that come with a cost, so a higher-power move with one does not dominate a plain move.
fn has_drawback(e: u8) -> bool {
    matches!(
        e,
        eff::EXPLODE | eff::RECOIL | eff::HYPER_BEAM | eff::CHARGE | eff::FLY | eff::THRASH | eff::JUMP_KICK
            | eff::TRAPPING | eff::OHKO | eff::SPECIAL_DAMAGE | eff::SUPER_FANG | eff::TWO_TO_FIVE
            | eff::ATTACK_TWICE | eff::TWINEEDLE | eff::RAGE | eff::DRAIN_HP | eff::DREAM_EATER | eff::SWIFT
            | eff::BIDE | eff::PAY_DAY
    )
}

/// Learnable moves worth considering, with plain damaging moves dominated by a same-type move
/// (at least as strong, as accurate, with the same or no secondary effect) removed.
pub fn move_pool(dex: &Dex, rules: &Rules, species: usize) -> Vec<u8> {
    let excluded: Vec<u8> = EXCLUDED.iter().chain(rules.banned_moves).filter_map(|n| dex.move_id(n)).collect();
    let learn: Vec<u8> = dex.species[species].learnset.iter().copied().filter(|m| !excluded.contains(m)).collect();
    let dominated = |a: u8| {
        let ma = dex.mv(a);
        if ma.power == 0 || has_drawback(ma.effect_id) {
            return false;
        }
        learn.iter().any(|&b| {
            let mb = dex.mv(b);
            b != a
                && mb.mtype == ma.mtype
                && !has_drawback(mb.effect_id)
                && (mb.effect_id == ma.effect_id || ma.effect_id == eff::NONE)
                && mb.power >= ma.power
                && mb.accuracy >= ma.accuracy
                && (mb.power > ma.power || mb.accuracy > ma.accuracy || b < a)
        })
    };
    learn.iter().copied().filter(|&m| !dominated(m)).collect()
}

/// Precision of one solve: damage-roll buckets and HP levels (0 = exact), and a state cap.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Precision {
    pub buckets: u8,
    pub hp_levels: u16,
    pub max_states: usize,
}

impl Precision {
    pub const fn new(buckets: u8, hp_levels: u16, max_states: usize) -> Precision {
        Precision { buckets, hp_levels, max_states }
    }

    pub fn is_exact(&self) -> bool {
        self.buckets == 0 && self.hp_levels == 0
    }
}

/// Fast screening: coarse rolls and HP, falling back to coarser settings for huge games.
/// Measured against exact solves, (3 buckets, 24 HP levels) is typically within ~0.003; the
/// fallbacks are coarser (around 0.03-0.04) and only used for the largest stall games.
pub const SCREEN: &[Precision] = &[
    Precision::new(3, 24, 200_000),
    Precision::new(1, 12, 200_000),
    Precision::new(1, 6, 1_000_000),
    Precision::new(1, 3, 4_000_000),
];

/// Very fast first-pass ranking of many candidate sets (biased by ~0.03, but similarly for
/// sets of the same Pokemon); the best few are re-scored with `SCREEN`.
pub const COARSE: &[Precision] =
    &[Precision::new(1, 12, 60_000), Precision::new(1, 6, 200_000), Precision::new(1, 3, 4_000_000)];

/// Final tables: exact when it fits, otherwise the finest approximation that does. Measured on
/// Tauros/Starmie against the exact 0.4074: (0, 96) +0.0006, (3, 48) -0.0017, (3, 24) +0.0027.
pub const FINAL: &[Precision] = &[
    Precision::new(0, 0, 600_000),
    Precision::new(0, 96, 1_000_000),
    Precision::new(3, 48, 1_500_000),
    Precision::new(3, 24, 2_000_000),
    Precision::new(1, 12, 3_000_000),
    Precision::new(1, 6, 6_000_000),
    Precision::new(1, 3, 8_000_000),
];

/// Second pass for the stall games `FINAL` gives up on (double Double Team plus Rest, sleep and
/// stat drops can reach ~10M states even on the coarsest grid). Needs ~500 bytes per state, so run
/// it with few threads: two concurrent 16M-state solves fit in about 17 GB.
pub const BIG: &[Precision] = &[Precision::new(1, 6, 16_000_000), Precision::new(1, 3, 24_000_000)];

/// Result of solving one matchup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Eval {
    /// Side 0's value: P(win) + P(draw)/2 under optimal play.
    pub value: f64,
    pub states: usize,
    pub secs: f64,
    pub precision: Precision,
    /// Optimal opening mixed strategies: (move, probability) for each side.
    pub opening: [Vec<(String, f64)>; 2],
}

/// Solves a matchup, trying each precision in turn until one finishes within its state cap.
pub fn evaluate(dex: &Dex, rules: &Rules, a: &Set, b: &Set, ladder: &[Precision]) -> Option<Eval> {
    let m = Matchup { mons: [a.mon(dex, rules), b.mon(dex, rules)], rules: rules.clone() };
    let t = Instant::now();
    for &p in ladder {
        let mut eng = Engine::new(dex, &m);
        eng.roll_buckets = p.buckets;
        let mut s = Solver::new(eng);
        s.max_states = p.max_states;
        s.set_hp_levels(p.hp_levels);
        if let Some(v) = s.try_solve(&m.initial(dex)) {
            let (x, sol) = s.policy(&m.initial(dex));
            let name = |side: usize, c: u8| dex.mv(m.mons[side].moves[c as usize]).name.clone();
            let opening = [
                x.rows.iter().zip(&sol.row).map(|(&c, &p)| (name(0, c), p)).collect(),
                x.cols.iter().zip(&sol.col).map(|(&c, &p)| (name(1, c), p)).collect(),
            ];
            return Some(Eval { value: v, states: s.states.len(), secs: t.elapsed().as_secs_f64(), precision: p, opening });
        }
    }
    None
}

/// Rough fraction of the defender's HP one use of a move removes (for building starting sets).
pub fn damage_fraction(dex: &Dex, rules: &Rules, atk: &Mon, mv: u8, def: &Mon) -> f64 {
    let d = dex.mv(mv);
    if d.power == 0 && d.effect_id != eff::SPECIAL_DAMAGE && d.effect_id != eff::SUPER_FANG {
        return 0.0;
    }
    let types = dex.species[def.species as usize].types;
    let mut mult = 1.0;
    for &m in dex.type_multipliers(d.mtype, types[0], types[1]) {
        mult *= m as f64 / 10.0;
    }
    let acc = d.accuracy as f64 / 256.0;
    let raw = match d.effect_id {
        eff::SPECIAL_DAMAGE => {
            if d.name == "SONICBOOM" {
                20.0
            } else if d.name == "DRAGON_RAGE" {
                40.0
            } else {
                rules.level as f64
            }
        }
        eff::SUPER_FANG => def.max_hp as f64 / 2.0 * 0.6,
        _ => {
            let special = dex.is_special(d.mtype);
            let (a, df) = if special { (atk.stats[3], def.stats[3]) } else { (atk.stats[0], def.stats[1]) };
            let base = (2.0 * rules.level as f64 / 5.0 + 2.0) * d.power as f64 * a as f64 / df as f64 / 50.0 + 2.0;
            let stab = if dex.species[atk.species as usize].types.contains(&d.mtype) { 1.5 } else { 1.0 };
            let shape = match d.effect_id {
                eff::TWO_TO_FIVE => 3.0,
                eff::ATTACK_TWICE | eff::TWINEEDLE => 2.0,
                eff::HYPER_BEAM => 0.75,
                eff::CHARGE => 0.5,
                eff::FLY => 0.6,
                eff::OHKO | eff::DREAM_EATER => 0.0,
                _ => 1.0,
            };
            base * stab * mult * shape
        }
    };
    if mult == 0.0 && d.effect_id != eff::SPECIAL_DAMAGE {
        return 0.0;
    }
    (raw * acc / def.max_hp as f64).min(1.0)
}

/// Greedy starting set: maximize summed best-move damage against the field, plus small bonuses for
/// the strongest kinds of status moves.
pub fn heuristic_set(dex: &Dex, rules: &Rules, species: usize, pool: &[u8], field: &[Mon]) -> Vec<u8> {
    let me = Mon::new(dex, species, &pool[..1], rules);
    let status_bonus = |mv: u8| -> f64 {
        let d = dex.mv(mv);
        match d.effect_id {
            eff::SLEEP => 0.25 * d.accuracy as f64 / 255.0,
            eff::PARALYZE => 0.12 * d.accuracy as f64 / 255.0,
            eff::HEAL => 0.06,
            eff::ATTACK_UP2 | 0x35 => 0.06, // Swords Dance, Amnesia (SPECIAL_UP2)
            eff::EVASION_UP1 => 0.03,
            _ => 0.0,
        }
    };
    let n = field.len().max(1) as f64;
    let score = |set: &[u8]| -> f64 {
        let dmg: f64 = field
            .iter()
            .map(|def| set.iter().map(|&m| damage_fraction(dex, rules, &me, m, def)).fold(0.0, f64::max))
            .sum::<f64>()
            / n;
        // Count only the best move of each status kind.
        let mut kinds: Vec<(u8, f64)> = Vec::new();
        for &m in set {
            let (k, v) = (dex.mv(m).effect_id, status_bonus(m));
            match kinds.iter_mut().find(|x| x.0 == k) {
                Some(x) => x.1 = x.1.max(v),
                None => kinds.push((k, v)),
            }
        }
        dmg + kinds.iter().map(|x| x.1).sum::<f64>()
    };
    let mut set: Vec<u8> = Vec::new();
    while set.len() < 4.min(pool.len()) {
        let best = pool
            .iter()
            .filter(|m| !set.contains(m))
            .map(|&m| {
                let mut t = set.clone();
                t.push(m);
                (m, score(&t))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        set.push(best.0);
    }
    set
}

pub fn to_set(dex: &Dex, species: usize, moves: &[u8]) -> Set {
    Set {
        species: dex.species[species].name.clone(),
        moves: moves.iter().map(|&m| dex.mv(m).name.clone()).collect(),
    }
}

/// Prior usefulness of a non-damaging move (or a damaging move whose value isn't raw damage).
fn status_prior(dex: &Dex, mv: u8) -> f64 {
    let d = dex.mv(mv);
    let name = d.name.as_str();
    let acc = d.accuracy as f64 / 255.0;
    match d.effect_id {
        eff::SLEEP => 1.0 * acc,
        eff::PARALYZE => 0.9 * acc,
        eff::HEAL => if name == "REST" { 0.6 } else { 0.85 },
        eff::ATTACK_UP2 | 0x35 => 0.7, // Swords Dance, Amnesia
        0x34 => 0.5,                   // Agility
        eff::EVASION_UP1 => 0.6,       // Double Team, Minimize
        eff::SUBSTITUTE => 0.5,
        eff::POISON => 0.45 * acc,
        eff::LEECH_SEED => 0.45 * acc,
        eff::REFLECT | eff::LIGHT_SCREEN => 0.4,
        eff::CONFUSION => 0.4 * acc,
        eff::TRAPPING => 0.6 * acc,
        eff::SPECIAL_DAMAGE | eff::SUPER_FANG => 0.5,
        eff::TRANSFORM => 0.5,
        eff::DISABLE => 0.3 * acc,
        eff::HAZE => 0.3,
        eff::ACCURACY_DOWN1_ID | eff::ACCURACY_DOWN2_ID => 0.35 * acc,
        eff::BIDE | eff::RAGE | eff::CONVERSION | eff::MIST => 0.15,
        _ if name == "COUNTER" => 0.5,
        eff::ATTACK_UP1..=eff::EVASION_UP1 => 0.3,
        _ => 0.1,
    }
}

/// A smaller pool for local search: the best damaging move of each type (and runners-up for the
/// strongest types), plus the most promising status moves. `keep` moves are always included.
pub fn focused_pool(dex: &Dex, rules: &Rules, species: usize, pool: &[u8], field: &[Mon], keep: &[u8], size: usize) -> Vec<u8> {
    let me = Mon::new(dex, species, &pool[..1], rules);
    let n = field.len().max(1) as f64;
    let utility = |m: u8| -> f64 { field.iter().map(|d| damage_fraction(dex, rules, &me, m, d)).sum::<f64>() / n };
    let plain = |m: u8| {
        let d = dex.mv(m);
        d.power > 0 && !matches!(d.effect_id, eff::TRAPPING | eff::OHKO | eff::DREAM_EATER)
    };
    let mut attacks: Vec<(u8, f64)> = pool.iter().copied().filter(|&m| plain(m)).map(|m| (m, utility(m))).collect();
    attacks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut chosen: Vec<u8> = keep.to_vec();
    let mut per_type: Vec<(u8, usize)> = Vec::new();
    // Best per type first, then a second move for the types that already rank highest.
    for pass in 0..2 {
        for &(m, _) in &attacks {
            let t = dex.mv(m).mtype;
            let count = per_type.iter().find(|x| x.0 == t).map(|x| x.1).unwrap_or(0);
            if count == pass && !chosen.contains(&m) && chosen.len() < size / 2 + keep.len() {
                chosen.push(m);
                match per_type.iter_mut().find(|x| x.0 == t) {
                    Some(x) => x.1 += 1,
                    None => per_type.push((t, 1)),
                }
            }
        }
    }
    let mut others: Vec<(u8, f64)> =
        pool.iter().copied().filter(|m| !chosen.contains(m) && !plain(*m)).map(|m| (m, status_prior(dex, m))).collect();
    others.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (m, _) in others {
        if chosen.len() >= size {
            break;
        }
        chosen.push(m);
    }
    for &(m, _) in &attacks {
        if chosen.len() >= size {
            break;
        }
        if !chosen.contains(&m) {
            chosen.push(m);
        }
    }
    chosen
}
