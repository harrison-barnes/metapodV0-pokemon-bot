//! One battle turn, ported from engine/battle/core.asm and engine/battle/effects.asm.
//!
//! The code follows the "player" side routines of the cartridge (ExecutePlayerMove and friends) for
//! both sides, which is what a link battle does on each Game Boy. Every random event goes through a
//! `Chance`, so a turn can be sampled or exhaustively enumerated.
//!
//! Deliberate departures from the cartridge (all documented in docs/ENGINE.md):
//! - PP is not tracked (every battle uses max PP, which is never exhausted in practice).
//! - Metronome, Mirror Move and Mimic are not supported (they act like Splash).
//! - Where the cartridge's player and enemy routines differ, the player version is used for both.

use crate::chance::Chance;
use crate::data::{eff, ty, Dex};
use crate::state::*;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;

/// Log names for stat indices ATK..EVA.
const STAT_NAMES: [&str; 6] = ["Attack", "Defense", "Speed", "Special", "accuracy", "evasion"];

/// `data/battle/stat_modifiers.asm`, indexed by stage 1..=13.
const STAT_RATIOS: [(u32, u32); 13] = [
    (25, 100),
    (28, 100),
    (33, 100),
    (40, 100),
    (50, 100),
    (66, 100),
    (1, 1),
    (15, 10),
    (2, 1),
    (25, 10),
    (3, 1),
    (35, 10),
    (4, 1),
];

/// A "choice" index meaning Struggle (no usable moves left).
pub const STRUGGLE_SLOT: u8 = 4;

pub struct Engine<'a> {
    pub dex: &'a Dex,
    pub m: &'a Matchup,
    has_bide: bool,
    /// may_counter[s]: side s can ever use Counter (directly, or by Transforming into a Counter user).
    may_counter: [bool; 2],
    /// charges_counterable[s]: side s has a Normal/Fighting two-turn move, whose charging turn loads
    /// counterable move data without touching wDamage.
    charges_counterable: [bool; 2],
    /// clause_bits[s]: the `Side::clauses` bits that can affect side s (it has a sleep move, or a
    /// move that can freeze, possibly by Transforming). Others are cleared by `canonicalize`.
    clause_bits: [u8; 2],
    /// Final-damage distributions keyed by (non-crit damage, crit damage, crit rate).
    damage_cache: RefCell<FxHashMap<(u16, u16, u16), Rc<DamageDist>>>,
    /// 0 = exact. Otherwise the 39 damage rolls are grouped into this many equally likely buckets,
    /// each represented by its middle roll (an approximation used for fast screening).
    pub roll_buckets: u8,
    /// Largest max HP a Counter can target (doubled damage at half of it always KOs).
    max_hp: u16,
    /// Canonicalise for a battle with switching (`set_teams`).
    team_mode: bool,
    /// Single-player rules (side 0 is the player, side 1 the game). None means link-battle rules.
    pub sp: Option<SinglePlayer>,
}

/// What differs from a link battle when playing against the game itself. See `trainer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinglePlayer {
    /// wObtainedBadges. Bits 0, 2, 4 and 6 boost attack, defense, speed and special by 1/8.
    pub badges: u8,
}

/// Distinct damage values after the random roll (and optionally the critical hit), with weights.
pub struct DamageDist {
    vals: Vec<u16>,
    weights: Vec<u32>,
    total: u64,
}

/// Move data loaded in battle RAM (wPlayerMoveNum, ..., wPlayerMoveAccuracy).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct Ram {
    num: u8,
    effect: u8,
    power: u8,
    mtype: u8,
    acc: u8,
}

/// Where execution continues after the status-condition checks (the `jp hl` targets).
enum Flow {
    Done,
    Normal,
    CanExecute,   // Rage
    CalcDamage,   // Thrash / Petal Dance continuing
    ApplyAgain,   // trapping move continuing
    IfMissed,     // Bide unleashing
}

impl<'a> Engine<'a> {
    pub fn new(dex: &'a Dex, m: &'a Matchup) -> Engine<'a> {
        let has_bide = m.either_has(dex.ids.bide);
        let transform = dex.move_id("TRANSFORM").unwrap();
        let has = |s: usize, mv: u8| m.mons[s].moves.contains(&mv);
        let may_counter = [0, 1].map(|s| has(s, dex.ids.counter) || (has(s, transform) && has(1 - s, dex.ids.counter)));
        let charges_counterable = [0, 1].map(|s: usize| {
            m.mons[s].moves.iter().any(|&mv| {
                let d = dex.mv(mv);
                mv != 0 && matches!(d.effect_id, eff::CHARGE | eff::FLY) && (d.mtype == ty::NORMAL || d.mtype == ty::FIGHTING)
            })
        });
        let can = |s: usize, effect: u8| {
            let any = |side: usize| m.mons[side].moves.iter().any(|&mv| mv != 0 && dex.mv(mv).effect_id == effect);
            any(s) || (has(s, transform) && any(1 - s))
        };
        let clause_bits = [0, 1].map(|s: usize| {
            (if can(s, eff::SLEEP) { SLEEP_BLOCKED } else { 0 })
                | (if can(s, eff::FREEZE_SIDE1) || can(s, eff::FREEZE_SIDE2) { FREEZE_BLOCKED } else { 0 })
                | (if m.heal_max_pp(dex, s).is_some() { HEAL_PP_MASK } else { 0 })
        });
        let max_hp = m.mons[0].max_hp.max(m.mons[1].max_hp);
        Engine { dex, m, has_bide, may_counter, charges_counterable, clause_bits, damage_cache: RefCell::new(FxHashMap::default()), roll_buckets: 0, max_hp, team_mode: false, sp: None }
    }

    /// Makes `canonicalize` lossless for a battle where these teams can switch in and out of the
    /// matchup (`endgame`): what may read stale state (Counter, Bide, two-turn moves) is judged over
    /// whole teams, a stale Quick Attack or Counter selection is kept even when the side picks a move
    /// (switching keeps it, and it still sets priority), and of the clause bits only the healing PP
    /// and a sleeping Pokemon's `SLEPT_BY_FOE` are kept: `SLEEP_BLOCKED` and `FREEZE_BLOCKED` are
    /// recomputed from both teams before every turn, as `battle6` does.
    pub fn set_teams(&mut self, teams: [&[Mon]; 2]) {
        let dex = self.dex;
        let transform = dex.move_id("TRANSFORM").unwrap();
        let has = |s: usize, mv: u8| teams[s].iter().any(|m| m.moves.contains(&mv));
        self.has_bide = has(0, dex.ids.bide) || has(1, dex.ids.bide);
        self.may_counter = [0, 1].map(|s| has(s, dex.ids.counter) || (has(s, transform) && has(1 - s, dex.ids.counter)));
        self.charges_counterable = [0, 1].map(|s: usize| {
            teams[s].iter().flat_map(|m| m.moves).any(|mv| {
                let d = dex.mv(mv);
                mv != 0 && matches!(d.effect_id, eff::CHARGE | eff::FLY) && (d.mtype == ty::NORMAL || d.mtype == ty::FIGHTING)
            })
        });
        self.max_hp = teams.iter().flat_map(|t| t.iter()).map(|m| m.max_hp).max().unwrap_or(0).max(self.max_hp);
        self.team_mode = true;
    }

    /// True when the side skips move selection this turn and reuses (or cannot use) a move.
    fn selection_skipped(b: &Battle, s: usize) -> bool {
        let me = &b.sides[s];
        me.bs2 & (NEEDS_TO_RECHARGE | USING_RAGE) != 0
            || me.bs1 & (THRASHING_ABOUT | CHARGING_UP) != 0
            || me.status & (FRZ | SLP_MASK) != 0
            || me.bs1 & (STORING_ENERGY | USING_TRAPPING_MOVE) != 0
    }

    fn trapped(b: &Battle, s: usize) -> bool {
        b.sides[1 - s].bs1 & USING_TRAPPING_MOVE != 0
    }

    /// Does this side pick a move this turn?
    pub fn needs_choice(&self, b: &Battle, s: usize) -> bool {
        !Self::selection_skipped(b, s) && !Self::trapped(b, s)
    }

    /// Move slots the side may pick (empty when `needs_choice` is false). Struggle when nothing is usable.
    pub fn legal_choices(&self, b: &Battle, s: usize) -> Vec<u8> {
        if !self.needs_choice(b, s) {
            return Vec::new();
        }
        let me = &b.sides[s];
        let disabled_slot = me.disabled >> 4;
        let mut v: Vec<u8> = (0..4u8)
            .filter(|&i| {
                let mv = me.moves[i as usize];
                mv != 0
                    && disabled_slot != i + 1
                    && !(self.m.rules.sleep_clause && me.sleep_used && self.dex.mv(mv).effect_id == eff::SLEEP)
                    && !(self.m.rules.heal_pp && me.clauses & HEAL_PP_MASK == 0 && self.dex.mv(mv).effect_id == eff::HEAL)
            })
            .collect();
        if v.is_empty() {
            v.push(STRUGGLE_SLOT);
        }
        v
    }

    /// Plays one turn. `choice[s]` is a slot from `legal_choices` (ignored when the side does not choose).
    pub fn turn<C: Chance>(&self, b: &Battle, choice: [u8; 2], ch: &mut C) -> (Battle, Outcome) {
        let mut t = Turn { e: self, b: b.clone(), ch, ram: [Ram::default(); 2], missed: false, switched: [false; 2], pp_spent: [false; 2], log: None };
        let o = t.run(choice);
        (t.b, o)
    }

    /// First half of `turn`: everything up to and including the first mover's action and its
    /// residual damage. Returns either the finished turn or the position halfway through, which
    /// `finish_turn` completes. Splitting lets an enumeration of the second half start from each
    /// distinct halfway position instead of replaying the first half for every outcome.
    pub fn start_turn<C: Chance>(&self, b: &Battle, choice: [u8; 2], ch: &mut C) -> Result<MidTurn, (Battle, Outcome)> {
        self.start_turn_switching(b, choice, [false; 2], ch)
    }

    /// `start_turn` for a turn in which the sides with `switched[s]` set have switched (see
    /// `turn_switching`).
    pub fn start_turn_switching<C: Chance>(&self, b: &Battle, choice: [u8; 2], switched: [bool; 2], ch: &mut C) -> Result<MidTurn, (Battle, Outcome)> {
        let mut t = Turn { e: self, b: b.clone(), ch, ram: [Ram::default(); 2], missed: false, switched, pp_spent: [false; 2], log: None };
        match t.run_first(choice) {
            Ok(first) => Ok(MidTurn { b: t.b, ram: t.ram, missed: t.missed, first: first as u8, switched }),
            Err(o) => Err((t.b, o)),
        }
    }

    /// Second half of `turn`, from a position returned by `start_turn`.
    pub fn finish_turn<C: Chance>(&self, mid: &MidTurn, ch: &mut C) -> (Battle, Outcome) {
        let mut t = Turn { e: self, b: mid.b.clone(), ch, ram: mid.ram, missed: mid.missed, switched: mid.switched, pp_spent: [false; 2], log: None };
        let o = t.run_second(mid.first as usize);
        (t.b, o)
    }

    /// A 6v6 turn: like `turn`, but a side with `switched[s]` set has already switched in this turn
    /// (see `battle6`) and executes no move. Optionally logged.
    pub fn turn_switching<C: Chance>(&self, b: &Battle, choice: [u8; 2], switched: [bool; 2], ch: &mut C, log: bool) -> TurnResult {
        let mut t = Turn { e: self, b: b.clone(), ch, ram: [Ram::default(); 2], missed: false, switched, pp_spent: [false; 2], log: log.then(Vec::new) };
        let o = t.run(choice);
        TurnResult { pp_spent: t.pp_spent, log: t.log.unwrap_or_default(), b: t.b, outcome: o }
    }

    /// Same as `turn`, also returning a human-readable log of what happened.
    pub fn turn_logged<C: Chance>(&self, b: &Battle, choice: [u8; 2], ch: &mut C) -> (Battle, Outcome, Vec<String>) {
        let mut t = Turn { e: self, b: b.clone(), ch, ram: [Ram::default(); 2], missed: false, switched: [false; 2], pp_spent: [false; 2], log: Some(Vec::new()) };
        let o = t.run(choice);
        (t.b, o, t.log.unwrap())
    }

    /// Single-player turn, first step: records selections and decides the order. `skip[s]` means side
    /// s already spent its turn (on a switch or item) and executes no move. See `trainer`.
    pub fn sp_begin<C: Chance>(&self, b: &Battle, choice: [u8; 2], skip: [bool; 2], ch: &mut C) -> SpTurn {
        let mut t = Turn { e: self, b: b.clone(), ch, ram: [Ram::default(); 2], missed: false, switched: skip, pp_spent: [false; 2], log: None };
        let first = t.begin(choice);
        SpTurn { first, b: t.b, ram: t.ram, missed: t.missed, skip: t.switched, pp_spent: [false; 2] }
    }

    /// Side s takes its slot in the turn: its move unless skipped, then its residual damage. Returns
    /// the outcome if someone fainted.
    pub fn sp_act<C: Chance>(&self, st: &mut SpTurn, s: usize, ch: &mut C) -> Option<Outcome> {
        let mut t = Turn { e: self, b: st.b.clone(), ch, ram: st.ram, missed: st.missed, switched: st.skip, pp_spent: [false; 2], log: None };
        let other = 1 - s;
        t.execute(s);
        let mut out = None;
        if t.b.sides[other].hp == 0 {
            out = Some(t.ko_outcome(s));
        } else {
            t.residual(s);
            if t.b.sides[s].hp == 0 {
                out = Some(Outcome::Win(other));
            }
        }
        st.b = t.b;
        st.ram = t.ram;
        st.missed = t.missed;
        st.pp_spent[0] |= t.pp_spent[0];
        st.pp_spent[1] |= t.pp_spent[1];
        out
    }

    /// `AIIncreaseStat`: the game's X item, which runs the stat-up effect without a move.
    pub fn sp_ai_stat_up<C: Chance>(&self, st: &mut SpTurn, stat: usize, ch: &mut C) {
        let mut t = Turn { e: self, b: st.b.clone(), ch, ram: st.ram, missed: st.missed, switched: st.skip, pp_spent: [false; 2], log: None };
        t.stat_up(1, stat, false);
        st.b = t.b;
    }

    /// End of a turn nobody fainted in: `CheckNumAttacksLeft`, and move RAM saved for next turn.
    pub fn sp_end(&self, st: &mut SpTurn) {
        for s in 0..2 {
            if st.b.sides[s].attacks_left == 0 {
                st.b.sides[s].bs1 &= !USING_TRAPPING_MOVE;
            }
            let r = st.ram[s];
            st.b.sides[s].ram = [r.num, r.power, r.mtype, r.acc];
        }
    }

    /// Saves move RAM after a turn cut short by a faint, as `sp_end` does for a whole turn.
    pub fn sp_save_ram(&self, st: &mut SpTurn) {
        for s in 0..2 {
            let r = st.ram[s];
            st.b.sides[s].ram = [r.num, r.power, r.mtype, r.acc];
        }
    }

    /// Zeroes state that can no longer affect the battle, so equivalent states hash equally.
    pub fn canonicalize(&self, b: &mut Battle) {
        let trapping = b.sides.iter().any(|s| s.bs1 & (USING_TRAPPING_MOVE | STORING_ENERGY) != 0);
        if !(self.has_bide || trapping) {
            // Only Counter can still read wDamage. Counter by side c fails unless the opponent's move
            // data in RAM is counterable, and the opponent overwrites wDamage whenever it loads new
            // move data and calculates damage. So a stale value only matters while the opponent's
            // current RAM is counterable, or if a charging turn could make it so without a damage
            // calculation.
            let matters = (0..2).any(|c| {
                let o = &b.sides[1 - c];
                let counterable = o.ram[1] != 0 && (o.ram[2] == ty::NORMAL || o.ram[2] == ty::FIGHTING);
                self.may_counter[c] && (counterable || self.charges_counterable[1 - c] || o.bs1 & CHARGING_UP != 0)
            });
            if !matters {
                b.damage = 0;
            } else {
                // Doubled damage at or above the target's max HP always KOs (or breaks a Substitute).
                let cap = (self.max_hp + 1) / 2;
                b.damage = b.damage.min(cap.max(128));
            }
        }
        let clear_flinch = (0..2).any(|s| b.sides[s].bs2 & (NEEDS_TO_RECHARGE | USING_RAGE) == 0);
        for s in 0..2 {
            let fresh = !Self::selection_skipped(b, s);
            let x = &mut b.sides[s];
            if clear_flinch {
                x.bs1 &= !FLINCHED;
            }
            x.bs1 &= !ATTACKING_MULTIPLE_TIMES;
            if x.bs2 & HAS_SUBSTITUTE_UP == 0 {
                x.sub_hp = 0;
            }
            if x.bs1 & STORING_ENERGY == 0 {
                x.bide_dmg = 0;
            }
            if x.bs1 & (STORING_ENERGY | THRASHING_ABOUT | USING_TRAPPING_MOVE) == 0 {
                x.attacks_left = 0;
            }
            if x.bs1 & CONFUSED == 0 {
                x.confused = 0;
            }
            if x.bs3 & BADLY_POISONED == 0 {
                x.toxic = 0;
            }
            // A stale selection is only read for priority (Quick Attack, Counter), by the opponent's
            // Counter check, or when a charging move executes on its second turn.
            let ids = self.dex.ids;
            let priority = x.selected == ids.quick_attack || x.selected == ids.counter;
            if (fresh && !(self.team_mode && priority)) || (x.bs1 & CHARGING_UP == 0 && !priority) {
                x.selected = 0;
            }
            // A side's leftover move data is reused when locked into Rage, Thrash or a trapping move.
            // Otherwise it is reloaded before use, and only the opponent's Counter can read it, which
            // only cares whether it was a damaging Normal/Fighting move.
            let locked = x.bs2 & USING_RAGE != 0 || x.bs1 & (THRASHING_ABOUT | USING_TRAPPING_MOVE) != 0;
            if !locked {
                let counterable = x.ram[1] != 0 && (x.ram[2] == ty::NORMAL || x.ram[2] == ty::FIGHTING);
                x.ram = if self.may_counter[1 - s] && counterable { [0, 1, ty::NORMAL, 0] } else { [0; 4] };
            } else if x.bs2 & USING_RAGE == 0 && x.bs1 & THRASHING_ABOUT == 0 {
                x.ram[3] = 0; // accuracy only matters while locked into Rage or Thrash
            }
            let slept_by_foe = x.clauses & SLEPT_BY_FOE != 0 && x.status & SLP_MASK != 0;
            x.clauses &= self.clause_bits[s];
            if self.team_mode {
                x.clauses &= !(SLEEP_BLOCKED | FREEZE_BLOCKED);
                if slept_by_foe {
                    x.clauses |= SLEPT_BY_FOE;
                }
            }
            if !self.m.rules.sleep_clause
                || !x.moves.iter().any(|&m| m != 0 && self.dex.mv(m).effect_id == eff::SLEEP)
            {
                x.sleep_used = false;
            }
        }
    }
}

/// Appends a line to the turn log when logging is enabled (formatting is skipped otherwise).
macro_rules! say {
    ($t:expr, $($arg:tt)*) => {
        if $t.log.is_some() {
            let line = format!($($arg)*);
            $t.log.as_mut().unwrap().push(line);
        }
    };
}

/// Result of `Engine::turn_switching`.
pub struct TurnResult {
    pub b: Battle,
    pub outcome: Outcome,
    /// pp_spent[s]: side s used up a PP of its selected move this turn.
    pub pp_spent: [bool; 2],
    pub log: Vec<String>,
}

/// A turn stopped halfway: after the first mover's action (see `Engine::start_turn`).
#[derive(Clone, PartialEq, Eq)]
pub struct MidTurn {
    b: Battle,
    ram: [Ram; 2],
    missed: bool,
    first: u8,
    switched: [bool; 2],
}

impl std::hash::Hash for MidTurn {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        // The packed battle words hash far faster than the derived field-by-field hash.
        self.b.pack().hash(h);
        let r = |x: &Ram| u64::from_le_bytes([x.num, x.effect, x.power, x.mtype, x.acc, 0, 0, 0]);
        let sw = (self.switched[0] as u64) << 60 ^ (self.switched[1] as u64) << 61;
        h.write_u64(r(&self.ram[0]) ^ r(&self.ram[1]).rotate_left(40) ^ (self.missed as u64) << 62 ^ (self.first as u64) << 63 ^ sw);
    }
}

/// `ApplyBadgeStatBoosts` on one set of stats.
pub fn apply_badges(x: &mut Side, badges: u8) {
    for i in 0..4 {
        if badges & (1 << (2 * i)) != 0 {
            x.stats[i] = (x.stats[i] + (x.stats[i] >> 3)).min(999);
        }
    }
}

/// A single-player turn in progress (see `Engine::sp_begin`).
#[derive(Clone)]
pub struct SpTurn {
    pub first: usize,
    pub b: Battle,
    ram: [Ram; 2],
    missed: bool,
    pub skip: [bool; 2],
    pub pp_spent: [bool; 2],
}

struct Turn<'e, 'a, C: Chance> {
    e: &'e Engine<'a>,
    b: Battle,
    ch: &'e mut C,
    ram: [Ram; 2],
    missed: bool,
    /// switched[s]: side s spent this turn switching (6v6). It still takes its place in the turn
    /// order and its residual damage, but executes no move (`ExecutePlayerMoveDone`).
    switched: [bool; 2],
    /// pp_spent[s]: side s reached `DecrementPP` this turn (the 1v1 solver ignores PP; 6v6 tracks it).
    pp_spent: [bool; 2],
    log: Option<Vec<String>>,
}

impl<'e, 'a, C: Chance> Turn<'e, 'a, C> {
    /// Under `Rules::heal_pp`, one use of the healing move: drop a healing-PP level with probability
    /// 1 / (uses the level stands for minus uses the next one down stands for), so the expected
    /// number of uses left falls by exactly one, as the HP grid does with damage.
    fn spend_heal_pp(&mut self, me: usize) {
        let Some(max) = self.e.m.heal_max_pp(self.e.dex, me) else { return };
        let l = self.b.sides[me].clauses >> HEAL_PP_SHIFT;
        if l == 0 {
            return;
        }
        let gap = (heal_level_pp(l, max) - heal_level_pp(l - 1, max)) as u32;
        if gap <= 1 || self.ch.pick(&[1, gap - 1]) == 0 {
            let x = &mut self.b.sides[me];
            x.clauses = (x.clauses & !HEAL_PP_MASK) | ((l - 1) << HEAL_PP_SHIFT);
        }
    }

    fn name(&self, s: usize) -> String {
        format!("{}#{}", self.e.dex.species[self.e.m.mons[s].species as usize].name, s)
    }

    fn mv_name(&self, id: u8) -> String {
        if id == CANNOT_MOVE { "(cannot move)".into() } else { self.e.dex.mv(id).name.clone() }
    }

    fn load_ram(&mut self, s: usize, mv: u8) {
        let d = self.e.dex.mv(mv);
        self.ram[s] = Ram { num: mv, effect: d.effect_id, power: d.power, mtype: d.mtype, acc: d.accuracy };
    }

    // ---------------------------------------------------------------- MainInBattleLoop

    fn run(&mut self, choice: [u8; 2]) -> Outcome {
        match self.run_first(choice) {
            Ok(first) => self.run_second(first),
            Err(o) => o,
        }
    }

    /// Restores move RAM, resets flinches and records selections. Returns who moves first.
    fn begin(&mut self, choice: [u8; 2]) -> usize {
        // Restore the move data each side still has in RAM from earlier turns.
        for s in 0..2 {
            let [mv, power, mtype, acc] = self.b.sides[s].ram;
            if mv != 0 {
                self.load_ram(s, mv);
            }
            // Canonical states may keep only the fields that still matter (see `canonicalize`).
            self.ram[s].power = power;
            self.ram[s].mtype = mtype;
            self.ram[s].acc = acc;
        }
        let fresh = [Engine::selection_skipped(&self.b, 0), Engine::selection_skipped(&self.b, 1)].map(|x| !x);
        let trapped = [Engine::trapped(&self.b, 0), Engine::trapped(&self.b, 1)];
        // On the cartridge only the player's state gates this; link play is symmetric.
        let resets = if self.e.sp.is_some() {
            self.b.sides[0].bs2 & (NEEDS_TO_RECHARGE | USING_RAGE) == 0
        } else {
            (0..2).any(|s| self.b.sides[s].bs2 & (NEEDS_TO_RECHARGE | USING_RAGE) == 0)
        };
        if resets {
            for s in 0..2 {
                self.b.sides[s].bs1 &= !FLINCHED;
            }
        }
        for s in 0..2 {
            // A side that switched keeps its stale selection (it still decides Quick Attack and
            // Counter priority, as wPlayerSelectedMove does on the cartridge).
            if !fresh[s] || self.switched[s] {
                continue;
            }
            self.b.sides[s].selected = if trapped[s] {
                CANNOT_MOVE
            } else if choice[s] == STRUGGLE_SLOT {
                self.e.dex.ids.struggle
            } else {
                let mv = self.b.sides[s].moves[choice[s] as usize];
                assert!(mv != 0, "illegal choice {} for side {s}", choice[s]);
                mv
            };
        }
        self.first_mover()
    }

    /// Up to and including the first mover's action and residual damage. Returns the first mover,
    /// or the outcome if the battle ended.
    fn run_first(&mut self, choice: [u8; 2]) -> Result<usize, Outcome> {
        let first = self.begin(choice);
        let second = 1 - first;

        self.execute(first);
        if self.b.sides[second].hp == 0 {
            return Err(self.ko_outcome(first));
        }
        self.residual(first);
        if self.b.sides[first].hp == 0 {
            return Err(Outcome::Win(second));
        }
        Ok(first)
    }

    fn run_second(&mut self, first: usize) -> Outcome {
        let second = 1 - first;
        self.execute(second);
        if self.b.sides[first].hp == 0 {
            return self.ko_outcome(second);
        }
        self.residual(second);
        if self.b.sides[second].hp == 0 {
            return Outcome::Win(first);
        }
        // CheckNumAttacksLeft
        for s in 0..2 {
            if self.b.sides[s].attacks_left == 0 {
                self.b.sides[s].bs1 &= !USING_TRAPPING_MOVE;
            }
        }
        for s in 0..2 {
            let r = self.ram[s];
            self.b.sides[s].ram = [r.num, r.power, r.mtype, r.acc];
        }
        Outcome::Ongoing
    }

    /// `attacker` knocked out its target; it may have fainted too (Explosion, recoil).
    fn ko_outcome(&self, attacker: usize) -> Outcome {
        if self.b.sides[attacker].hp == 0 { Outcome::Draw } else { Outcome::Win(attacker) }
    }

    fn first_mover(&mut self) -> usize {
        let ids = self.e.dex.ids;
        let sel = [self.b.sides[0].selected, self.b.sides[1].selected];
        let qa = sel.map(|m| m == ids.quick_attack);
        let ctr = sel.map(|m| m == ids.counter);
        if qa[0] && !qa[1] {
            return 0;
        }
        if qa[1] && !qa[0] {
            return 1;
        }
        if !(qa[0] && qa[1]) {
            if ctr[0] && !ctr[1] {
                return 1;
            }
            if ctr[1] && !ctr[0] {
                return 0;
            }
        }
        let (a, b) = (self.b.sides[0].stats[SPE], self.b.sides[1].stats[SPE]);
        if a != b {
            return if a > b { 0 } else { 1 };
        }
        // cp 50 percent + 1
        if self.ch.below(128) { 0 } else { 1 }
    }

    // ---------------------------------------------------------------- ExecutePlayerMove

    fn execute(&mut self, me: usize) {
        let op = 1 - me;
        if self.switched[me] {
            return;
        }
        if self.b.sides[me].selected == CANNOT_MOVE {
            say!(self, "  (trapped, cannot move)");
            return;
        }
        self.missed = false;
        let flow = self.status_checks(me);
        // Each arm falls through to the next stage, like the labels in the assembly.
        let mut stage = match flow {
            Flow::Done => return,
            Flow::Normal => 0,
            Flow::CanExecute => 1,
            Flow::CalcDamage => 2,
            Flow::IfMissed => 3,
            Flow::ApplyAgain => 4,
        };
        if stage == 0 {
            let sel = self.b.sides[me].selected;
            self.load_ram(me, sel);
            let x = &mut self.b.sides[me];
            if x.bs1 & CHARGING_UP != 0 {
                x.bs1 &= !(CHARGING_UP | INVULNERABLE);
            } else if matches!(self.ram[me].effect, eff::CHARGE | eff::FLY) {
                say!(self, "  {} charges {}", self.name(me), self.mv_name(sel));
                self.charge_effect(me);
                return;
            }
            stage = 1;
        }
        if stage == 1 {
            say!(self, "  {} used {}", self.name(me), self.mv_name(self.ram[me].num));
            // PlayerCanExecuteMove -> DecrementPP, which skips Struggle, Bide, Thrash, multi-hit
            // follow-ups and Rage.
            let x = &self.b.sides[me];
            self.pp_spent[me] = x.selected != self.e.dex.ids.struggle
                && x.bs1 & (STORING_ENERGY | THRASHING_ABOUT | ATTACKING_MULTIPLE_TIMES) == 0
                && x.bs2 & USING_RAGE == 0;
            if self.pp_spent[me] && self.e.dex.mv(x.selected).effect_id == eff::HEAL {
                self.spend_heal_pp(me);
            }
            let e = self.ram[me].effect;
            if eff::residual1(e) {
                self.run_effect(me);
                return;
            }
            if eff::special_cont(e) {
                self.run_effect(me);
            }
            stage = 2;
        }
        if stage == 2 {
            stage = 3;
            if !self.calc_move_damage(me) {
                stage = 4; // 0-power move or failed OHKO: skip the hit test, go straight to MirrorMoveCheck
            }
        }
        if stage == 3 {
            stage = 4; // HandleIfPlayerMoveMissed only selects an animation
        }
        let _ = stage;
        // MirrorMoveCheck onwards; loops for multi-hit moves.
        loop {
            let e = self.ram[me].effect;
            if eff::residual2(e) {
                self.run_effect(me);
                return;
            }
            let mut hit = false;
            if self.missed {
                say!(self, "  ...but it missed");
                self.move_failure(me);
                if e != eff::EXPLODE {
                    return;
                }
            } else {
                self.apply_attack(me);
                hit = true;
            }
            let _ = hit;
            let e = self.ram[me].effect;
            if eff::always_happen(e) {
                self.run_effect(me);
            }
            if self.b.sides[op].hp == 0 {
                return;
            }
            self.building_rage(me);
            let x = &mut self.b.sides[me];
            if x.bs1 & ATTACKING_MULTIPLE_TIMES != 0 {
                x.attacks_left = x.attacks_left.wrapping_sub(1);
                if x.attacks_left != 0 {
                    continue;
                }
                x.bs1 &= !ATTACKING_MULTIPLE_TIMES;
            }
            let e = self.ram[me].effect;
            if e != 0 && !eff::special(e) {
                self.run_effect(me);
            }
            return;
        }
    }

    /// PlayerCalcMoveDamage through MoveHitTest. Returns false when the hit test is skipped.
    fn calc_move_damage(&mut self, me: usize) -> bool {
        let e = self.ram[me].effect;
        if eff::set_damage(e) {
            self.move_hit_test(me);
            return true;
        }
        if self.ram[me].num == self.e.dex.ids.counter {
            self.counter(me);
            return true;
        }
        if e == eff::OHKO {
            // CalculateDamage jumps to the OHKO effect before looking at base power.
            self.b.damage = 0;
            let (a, b) = (self.b.sides[me].stats[SPE], self.b.sides[1 - me].stats[SPE]);
            if a < b {
                self.missed = true;
                return false;
            }
            self.b.damage = 0xFFFF;
            let (d, immune) = self.adjust_for_type(me, 0xFFFF);
            self.b.damage = d;
            if immune {
                self.missed = true;
            }
            self.randomize_then_hit_test(me);
            return true;
        }
        if self.ram[me].power == 0 && !matches!(e, eff::TWO_TO_FIVE | eff::EFFECT_1E) {
            // CriticalHitTest and GetDamageVars do nothing visible except zeroing wDamage.
            self.b.damage = 0;
            return false;
        }
        // Damage before randomization, for a normal hit and a critical hit.
        let crit_rate = self.crit_rate(me);
        let nocrit = self.damage_before_random(me, false);
        let crit = self.damage_before_random(me, true);
        if (nocrit.0 == 0) == (crit.0 == 0) {
            // Crit only changes the amount, so test accuracy first and enumerate (crit, roll) jointly.
            if nocrit.0 == 0 {
                self.missed = true;
                self.b.damage = 0;
                if self.ram[me].effect == eff::TRAPPING {
                    self.move_hit_test(me); // still decides whether the trap sticks
                }
                return true;
            }
            if !self.move_hit_test(me) {
                return true;
            }
            self.b.damage = self.joint_damage_roll(crit_rate, nocrit.0, crit.0);
            true
        } else {
            let is_crit = self.ch.below(crit_rate);
            let d = if is_crit { crit.0 } else { nocrit.0 };
            self.b.damage = d;
            if d == 0 {
                self.missed = true;
            }
            self.randomize_then_hit_test(me);
            true
        }
    }

    fn randomize_then_hit_test(&mut self, me: usize) {
        let hit = self.move_hit_test(me);
        if hit && !self.missed && self.b.damage >= 2 {
            let d = self.b.damage;
            self.b.damage = self.joint_damage_roll(0, d, d);
        }
    }

    /// Final damage over (critical hit, random roll), merged by value. `crit_rate` 0 means no crit roll.
    fn joint_damage_roll(&mut self, crit_rate: u32, nocrit: u16, crit: u16) -> u16 {
        let key = (nocrit, if crit_rate == 0 { 0 } else { crit }, crit_rate as u16);
        let dist = {
            let mut cache = self.e.damage_cache.borrow_mut();
            let k = self.e.roll_buckets;
            cache.entry(key).or_insert_with(|| Rc::new(damage_dist(crit_rate, nocrit, crit, k))).clone()
        };
        if dist.vals.len() == 1 {
            return dist.vals[0];
        }
        dist.vals[self.ch.pick_positive(&dist.weights, dist.total)]
    }

    // ---------------------------------------------------------------- status checks

    fn status_checks(&mut self, me: usize) -> Flow {
        let op = 1 - me;
        let x = &mut self.b.sides[me];
        if x.status & SLP_MASK != 0 {
            x.status = (x.status & SLP_MASK) - 1;
            let woke = x.status == 0;
            say!(self, "  {} {}", self.name(me), if woke { "woke up" } else { "is fast asleep" });
            return Flow::Done;
        }
        if x.status & FRZ != 0 {
            say!(self, "  {} is frozen solid", self.name(me));
            return Flow::Done;
        }
        if self.b.sides[op].bs1 & USING_TRAPPING_MOVE != 0 {
            say!(self, "  {} can't move", self.name(me));
            return Flow::Done;
        }
        let x = &mut self.b.sides[me];
        if x.bs1 & FLINCHED != 0 {
            x.bs1 &= !FLINCHED;
            say!(self, "  {} flinched", self.name(me));
            return Flow::Done;
        }
        if x.bs2 & NEEDS_TO_RECHARGE != 0 {
            x.bs2 &= !NEEDS_TO_RECHARGE;
            say!(self, "  {} must recharge", self.name(me));
            return Flow::Done;
        }
        if x.disabled != 0 {
            x.disabled -= 1;
            if x.disabled & 0xF == 0 {
                x.disabled = 0;
                x.disabled_move = 0;
            }
        }
        let mut hurt_or_para = false;
        if x.bs1 & CONFUSED != 0 {
            x.confused = x.confused.wrapping_sub(1);
            if x.confused == 0 {
                x.bs1 &= !CONFUSED;
            } else if !self.ch.below(128) {
                self.b.sides[me].bs1 &= CONFUSED;
                say!(self, "  {} hurt itself in confusion", self.name(me));
                self.self_confusion_damage(me);
                hurt_or_para = true;
            }
        }
        if !hurt_or_para {
            let x = &mut self.b.sides[me];
            if x.disabled_move != 0 && x.selected == x.disabled_move {
                x.bs1 &= !CHARGING_UP;
                return Flow::Done;
            }
            if x.status & PAR != 0 && self.ch.below(63) {
                say!(self, "  {} is fully paralyzed", self.name(me));
                hurt_or_para = true;
            }
        }
        if hurt_or_para {
            self.b.sides[me].bs1 &= !(STORING_ENERGY | THRASHING_ABOUT | CHARGING_UP | USING_TRAPPING_MOVE);
            return Flow::Done;
        }
        let damage = self.b.damage;
        let x = &mut self.b.sides[me];
        if x.bs1 & STORING_ENERGY != 0 {
            x.bide_dmg = x.bide_dmg.wrapping_add(damage);
            x.attacks_left = x.attacks_left.wrapping_sub(1);
            if x.attacks_left != 0 {
                say!(self, "  {} is storing energy", self.name(me));
                return Flow::Done;
            }
            x.bs1 &= !STORING_ENERGY;
            let d = x.bide_dmg.wrapping_mul(2);
            x.bide_dmg = 0;
            self.b.damage = d;
            self.load_ram(me, self.e.dex.ids.bide);
            self.ram[me].power = 1;
            self.ram[me].effect = 0;
            if d == 0 {
                self.missed = true;
            }
            say!(self, "  {} unleashed energy", self.name(me));
            return Flow::IfMissed;
        }
        if x.bs1 & THRASHING_ABOUT != 0 {
            x.attacks_left = x.attacks_left.wrapping_sub(1);
            if x.attacks_left == 0 {
                x.bs1 &= !THRASHING_ABOUT;
                x.bs1 |= CONFUSED;
                let r = self.ch.pick(&[1, 1, 1, 1]) as u8;
                self.b.sides[me].confused = r + 2;
            }
            say!(self, "  {} is thrashing about ({})", self.name(me), self.mv_name(self.ram[me].num));
            return Flow::CalcDamage;
        }
        if x.bs1 & USING_TRAPPING_MOVE != 0 {
            x.attacks_left = x.attacks_left.wrapping_sub(1);
            say!(self, "  {}'s attack continues", self.name(me));
            return Flow::ApplyAgain;
        }
        if x.bs2 & USING_RAGE != 0 {
            self.ram[me].effect = 0;
            return Flow::CanExecute;
        }
        Flow::Normal
    }

    fn self_confusion_damage(&mut self, me: usize) {
        let op = 1 - me;
        // Own attack against own defense, but the *opponent's* Reflect is checked.
        let x = &self.b.sides[me];
        let mut def = x.stats[DEF] as u32;
        if self.b.sides[op].bs3 & HAS_REFLECT_UP != 0 {
            def *= 2;
        }
        let (atk, def) = scale_stats(x.stats[ATK] as u32, def);
        let lvl = self.e.m.mons[me].level as u32;
        self.b.damage = calculate_damage(lvl, 40, atk, def, false);
        // The move's power and type are left as 40 / Normal in RAM (only the effect is restored).
        self.ram[me].power = 40;
        self.ram[me].mtype = ty::NORMAL;
        self.apply_damage_to_self(me);
    }

    // ---------------------------------------------------------------- damage

    fn crit_rate(&self, me: usize) -> u32 {
        let base_speed = self.e.dex.species[self.b.sides[me].species as usize].base.spe as u32;
        let mut b = base_speed >> 1;
        if self.b.sides[me].bs2 & GETTING_PUMPED != 0 {
            b >>= 1; // the Focus Energy bug
        } else {
            b = (b << 1).min(255);
        }
        if self.e.dex.ids.high_crit.contains(&self.ram[me].num) {
            b = (b << 1).min(255);
            b = (b << 1).min(255);
        } else {
            b >>= 1;
        }
        b
    }

    /// GetDamageVars + CalculateDamage + AdjustDamageForMoveType. Returns (damage, missed due to type).
    fn damage_before_random(&self, me: usize, crit: bool) -> (u16, bool) {
        let op = 1 - me;
        let r = self.ram[me];
        let (x, y) = (&self.b.sides[me], &self.b.sides[op]);
        let special = self.e.dex.is_special(r.mtype);
        let (si, screen) = if special { (SPC, HAS_LIGHT_SCREEN_UP) } else { (ATK, HAS_REFLECT_UP) };
        let di = if special { SPC } else { DEF };
        let (atk, def) = if crit {
            (self.e.m.mons[me].stats[si] as u32, self.e.m.mons[op].stats[di] as u32)
        } else {
            let mut d = y.stats[di] as u32;
            if y.bs3 & screen != 0 {
                d *= 2;
            }
            (x.stats[si] as u32, d)
        };
        let (atk, mut def) = scale_stats(atk, def);
        if r.effect == eff::EXPLODE {
            def >>= 1;
            if def == 0 {
                def = 1;
            }
        }
        let lvl = self.e.m.mons[me].level as u32 * if crit { 2 } else { 1 };
        let d = calculate_damage(lvl & 0xFF, r.power as u32, atk, def, false);
        let (d, immune) = self.adjust_for_type(me, d);
        (d, immune)
    }

    fn adjust_for_type(&self, me: usize, mut d: u16) -> (u16, bool) {
        let op = 1 - me;
        let t = self.ram[me].mtype;
        let (x, y) = (&self.b.sides[me], &self.b.sides[op]);
        if t == x.types[0] || t == x.types[1] {
            d = d.wrapping_add(d >> 1);
        }
        let mut missed = false;
        for &mult in self.e.dex.type_multipliers(t, y.types[0], y.types[1]) {
            d = ((d as u32 * mult as u32 / 10) & 0xFFFF) as u16;
            if d == 0 {
                missed = true;
            }
        }
        (d, missed)
    }

    fn counter(&mut self, me: usize) {
        let op = 1 - me;
        self.missed = true;
        let y = &self.b.sides[op];
        let opp_ram = self.ram[op];
        if y.selected == self.e.dex.ids.counter || opp_ram.power == 0 {
            return;
        }
        if opp_ram.mtype != ty::NORMAL && opp_ram.mtype != ty::FIGHTING {
            // `xor a; ret` leaves wMoveMissed set: Counter fails.
            return;
        }
        if self.b.damage == 0 {
            return;
        }
        self.b.damage = self.b.damage.saturating_mul(2);
        self.missed = false;
        self.move_hit_test(me);
    }

    /// MoveHitTest. Returns true if the move hit.
    fn move_hit_test(&mut self, me: usize) -> bool {
        let op = 1 - me;
        let e = self.ram[me].effect;
        let y = &self.b.sides[op];
        let mut miss = false;
        if e == eff::DREAM_EATER && y.status & SLP_MASK == 0 {
            miss = true;
        } else if e == eff::SWIFT {
            return true;
        } else if y.bs1 & INVULNERABLE != 0 {
            miss = true;
        } else if (matches!(e, eff::ATTACK_DOWN1..=eff::HAZE) || matches!(e, eff::ATTACK_DOWN2..=eff::REFLECT))
            && y.bs2 & PROTECTED_BY_MIST != 0
        {
            miss = true;
        } else {
            let acc = self.calc_hit_chance(me);
            self.ram[me].acc = acc;
            if !self.ch.below(acc as u32) {
                miss = true;
            }
        }
        if miss {
            self.b.damage = 0;
            self.missed = true;
            self.b.sides[me].bs1 &= !USING_TRAPPING_MOVE;
        }
        !miss
    }

    fn calc_hit_chance(&self, me: usize) -> u8 {
        let op = 1 - me;
        let acc_mod = self.b.sides[me].mods[ACC];
        let eva_mod = 14 - self.b.sides[op].mods[EVA];
        let mut a = self.ram[me].acc as u32;
        for m in [acc_mod, eva_mod] {
            let (n, d) = STAT_RATIOS[(m - 1) as usize];
            a = a * n / d;
            if a == 0 {
                a = 1;
            }
        }
        a.min(255) as u8
    }

    // ---------------------------------------------------------------- applying damage

    fn apply_attack(&mut self, me: usize) {
        let op = 1 - me;
        let r = self.ram[me];
        match r.effect {
            eff::OHKO => {}
            eff::SUPER_FANG => {
                self.b.damage = (self.b.sides[op].hp >> 1).max(1);
            }
            eff::SPECIAL_DAMAGE => {
                let ids = self.e.dex.ids;
                let lvl = self.e.m.mons[me].level;
                self.b.damage = if r.num == ids.seismic_toss || r.num == ids.night_shade {
                    lvl as u16
                } else if r.num == ids.sonicboom {
                    20
                } else if r.num == ids.dragon_rage {
                    40
                } else {
                    // Psywave: uniform in [1, level * 1.5)
                    let b = lvl.wrapping_add(lvl >> 1) as usize;
                    let w = vec![1u32; b.saturating_sub(1).max(1)];
                    self.ch.pick(&w) as u16 + 1
                };
            }
            _ => {
                if r.power == 0 {
                    return;
                }
            }
        }
        let d = self.b.damage;
        say!(self, "  -> {d} damage");
        self.apply_damage_to(me, op);
    }

    /// ApplyDamageToEnemyPokemon (from `me`'s perspective).
    fn apply_damage_to(&mut self, me: usize, target: usize) {
        if self.b.damage == 0 {
            return;
        }
        if self.b.sides[target].bs2 & HAS_SUBSTITUTE_UP != 0 {
            self.attack_substitute(me);
            return;
        }
        let y = &mut self.b.sides[target];
        if self.b.damage >= y.hp {
            self.b.damage = y.hp;
            y.hp = 0;
        } else {
            y.hp -= self.b.damage;
        }
    }

    /// ApplyDamageToPlayerPokemon on one's own turn (confusion, crash damage). A Substitute on the
    /// user redirects the damage to the *opponent's* Substitute (cartridge bug).
    fn apply_damage_to_self(&mut self, me: usize) {
        if self.b.damage == 0 {
            return;
        }
        if self.b.sides[me].bs2 & HAS_SUBSTITUTE_UP != 0 {
            self.attack_substitute(me);
            return;
        }
        let x = &mut self.b.sides[me];
        if self.b.damage >= x.hp {
            self.b.damage = x.hp;
            x.hp = 0;
        } else {
            x.hp -= self.b.damage;
        }
    }

    /// AttackSubstitute with hWhoseTurn = me: always hits the opponent's Substitute.
    fn attack_substitute(&mut self, me: usize) {
        let op = 1 - me;
        let d = self.b.damage;
        let y = &mut self.b.sides[op];
        let broke = if d > 0xFF {
            true
        } else {
            let (v, borrow) = y.sub_hp.overflowing_sub(d as u8);
            y.sub_hp = v;
            borrow
        };
        if broke {
            y.bs2 &= !HAS_SUBSTITUTE_UP;
            self.ram[me].effect = 0;
            say!(self, "  {}'s substitute broke", self.name(op));
        } else {
            say!(self, "  the substitute took the damage");
        }
    }

    fn building_rage(&mut self, me: usize) {
        let op = 1 - me;
        if self.b.sides[op].bs2 & USING_RAGE == 0 || self.b.sides[op].mods[ATK] == 13 {
            return;
        }
        self.stat_up(op, ATK, false);
        self.ram[op].effect = 0;
        self.ram[op].num = self.e.dex.ids.rage;
    }

    /// PrintMoveFailureText: Jump Kick / Hi Jump Kick crash for 1 HP.
    fn move_failure(&mut self, me: usize) {
        if self.ram[me].effect == eff::JUMP_KICK {
            self.b.damage = 1;
            say!(self, "  {} kept going and crashed", self.name(me));
            self.apply_damage_to_self(me);
        }
    }

    // ---------------------------------------------------------------- residual damage

    fn residual(&mut self, me: usize) {
        let op = 1 - me;
        let max = self.e.m.mons[me].max_hp;
        if self.b.sides[me].status & (BRN | PSN) != 0 {
            let d = self.residual_damage(me, max);
            let x = &mut self.b.sides[me];
            x.hp = x.hp.saturating_sub(d);
            say!(self, "  {} is hurt by poison/burn ({d})", self.name(me));
        }
        if self.b.sides[me].bs2 & SEEDED != 0 {
            let d = self.residual_damage(me, max);
            let x = &mut self.b.sides[me];
            x.hp = x.hp.saturating_sub(d);
            let omax = self.e.m.mons[op].max_hp;
            let y = &mut self.b.sides[op];
            y.hp = (y.hp as u32 + d as u32).min(omax as u32) as u16;
            say!(self, "  leech seed drains {} ({d})", self.name(me));
        }
    }

    fn residual_damage(&mut self, me: usize, max: u16) -> u16 {
        let mut d = (((max >> 2) & 0xFF) >> 2).max(1);
        let x = &mut self.b.sides[me];
        if x.bs3 & BADLY_POISONED != 0 {
            x.toxic = x.toxic.wrapping_add(1);
            d = d.wrapping_mul(x.toxic as u16);
        }
        d
    }

    // ---------------------------------------------------------------- move effects

    fn run_effect(&mut self, me: usize) {
        let op = 1 - me;
        let e = self.ram[me].effect;
        match e {
            0x01 | eff::SLEEP => self.sleep_effect(me),
            eff::POISON_SIDE1 | eff::POISON_SIDE2 | eff::POISON => self.poison_effect(me),
            eff::DRAIN_HP | eff::DREAM_EATER => {
                let d = (self.b.damage >> 1).max(1);
                self.b.damage = d;
                let max = self.e.m.mons[me].max_hp;
                let x = &mut self.b.sides[me];
                x.hp = (x.hp as u32 + d as u32).min(max as u32) as u16;
            }
            eff::BURN_SIDE1 | eff::FREEZE_SIDE1 | eff::PARALYZE_SIDE1 | eff::BURN_SIDE2 | eff::FREEZE_SIDE2
            | eff::PARALYZE_SIDE2 => self.freeze_burn_paralyze(me),
            eff::EXPLODE => {
                let x = &mut self.b.sides[me];
                x.hp = 0;
                x.status = 0;
                x.bs2 &= !SEEDED;
            }
            eff::ATTACK_UP1..=eff::EVASION_UP1 => self.stat_up(me, (e - eff::ATTACK_UP1) as usize, false),
            eff::ATTACK_UP2..=eff::EVASION_UP2 => self.stat_up(me, (e - eff::ATTACK_UP2) as usize, true),
            eff::ATTACK_DOWN1..=eff::EVASION_DOWN1 | eff::ATTACK_DOWN2..=eff::EVASION_DOWN2 => self.stat_down(me),
            eff::ATTACK_DOWN_SIDE..=0x4B => self.stat_down(me),
            eff::CONVERSION => {
                if self.b.sides[op].bs1 & INVULNERABLE == 0 {
                    self.b.sides[me].types = self.b.sides[op].types;
                }
            }
            eff::HAZE => self.haze(me),
            eff::BIDE => {
                let x = &mut self.b.sides[me];
                x.bs1 |= STORING_ENERGY;
                x.bide_dmg = 0;
                self.ram[me].effect = 0;
                self.ram[op].effect = 0;
                let r = self.ch.pick(&[1, 1]) as u8;
                self.b.sides[me].attacks_left = r + 2;
            }
            eff::THRASH => {
                self.b.sides[me].bs1 |= THRASHING_ABOUT;
                let r = self.ch.pick(&[1, 1]) as u8;
                self.b.sides[me].attacks_left = r + 2;
            }
            eff::TWO_TO_FIVE | eff::EFFECT_1E | eff::ATTACK_TWICE | eff::TWINEEDLE => {
                let x = &mut self.b.sides[me];
                if x.bs1 & ATTACKING_MULTIPLE_TIMES != 0 {
                    return;
                }
                x.bs1 |= ATTACKING_MULTIPLE_TIMES;
                let hits = if e == eff::TWINEEDLE {
                    self.ram[me].effect = eff::POISON_SIDE1;
                    2
                } else if e == eff::ATTACK_TWICE {
                    2
                } else {
                    // 2 and 3 hits 3/8 each, 4 and 5 hits 1/8 each
                    [2u8, 3, 4, 5][self.ch.pick(&[3, 3, 1, 1])]
                };
                self.b.sides[me].attacks_left = hits;
            }
            eff::FLINCH_SIDE1 | eff::FLINCH_SIDE2 => {
                if self.b.sides[op].bs2 & HAS_SUBSTITUTE_UP != 0 {
                    return;
                }
                let thr = if e == eff::FLINCH_SIDE1 { 26 } else { 77 };
                if self.ch.below(thr) {
                    let y = &mut self.b.sides[op];
                    y.bs1 |= FLINCHED;
                    y.bs2 &= !NEEDS_TO_RECHARGE;
                }
            }
            eff::CHARGE | eff::FLY => self.charge_effect(me),
            eff::TRAPPING => {
                if self.b.sides[me].bs1 & USING_TRAPPING_MOVE != 0 {
                    return;
                }
                self.b.sides[op].bs2 &= !NEEDS_TO_RECHARGE;
                self.b.sides[me].bs1 |= USING_TRAPPING_MOVE;
                let n = [1u8, 2, 3, 4][self.ch.pick(&[3, 3, 1, 1])];
                self.b.sides[me].attacks_left = n;
            }
            eff::MIST => self.b.sides[me].bs2 |= PROTECTED_BY_MIST,
            eff::FOCUS_ENERGY => self.b.sides[me].bs2 |= GETTING_PUMPED,
            eff::RECOIL => {
                let mut r = self.b.damage >> 1;
                if self.ram[me].num != self.e.dex.ids.struggle {
                    r >>= 1;
                }
                let r = r.max(1);
                let x = &mut self.b.sides[me];
                x.hp = x.hp.saturating_sub(r);
                say!(self, "  {} is hit with recoil ({r})", self.name(me));
            }
            eff::CONFUSION => {
                if self.b.sides[op].bs2 & HAS_SUBSTITUTE_UP != 0 || !self.move_hit_test(me) {
                    return;
                }
                self.confuse(op);
            }
            eff::CONFUSION_SIDE => {
                if self.ch.below(25) {
                    self.confuse(op);
                }
            }
            eff::HEAL => self.heal(me),
            eff::TRANSFORM => {
                let y = self.b.sides[op].clone();
                let x = &mut self.b.sides[me];
                x.species = y.species;
                x.types = y.types;
                x.moves = y.moves;
                x.stats = y.stats;
                x.unmod = y.unmod;
                x.mods = y.mods;
                x.bs3 |= TRANSFORMED;
            }
            eff::LIGHT_SCREEN => self.b.sides[me].bs3 |= HAS_LIGHT_SCREEN_UP,
            eff::REFLECT => self.b.sides[me].bs3 |= HAS_REFLECT_UP,
            eff::PARALYZE => self.paralyze_effect(me),
            eff::SUBSTITUTE => {
                let max = self.e.m.mons[me].max_hp;
                let x = &mut self.b.sides[me];
                if x.bs2 & HAS_SUBSTITUTE_UP != 0 {
                    return;
                }
                let cost = ((max >> 2) & 0xFF) as u16;
                if x.hp < cost {
                    return;
                }
                x.hp -= cost;
                x.sub_hp = cost as u8;
                x.bs2 |= HAS_SUBSTITUTE_UP;
            }
            eff::HYPER_BEAM => self.b.sides[me].bs2 |= NEEDS_TO_RECHARGE,
            eff::RAGE => self.b.sides[me].bs2 |= USING_RAGE,
            eff::LEECH_SEED => {
                if !self.move_hit_test(me) {
                    return;
                }
                let y = &mut self.b.sides[op];
                if y.types.contains(&ty::GRASS) || y.bs2 & SEEDED != 0 {
                    return;
                }
                y.bs2 |= SEEDED;
            }
            eff::DISABLE => self.disable(me),
            // Pay Day, Swift, Splash, Teleport/Roar/Whirlwind, Mimic, Metronome, Mirror Move: no battle effect here.
            _ => {}
        }
    }

    fn charge_effect(&mut self, me: usize) {
        let fly = self.ram[me].effect == eff::FLY || self.ram[me].num == self.e.dex.ids.dig;
        let x = &mut self.b.sides[me];
        x.bs1 |= CHARGING_UP;
        if fly {
            x.bs1 |= INVULNERABLE;
        }
    }

    fn sleep_effect(&mut self, me: usize) {
        let op = 1 - me;
        self.b.sides[me].sleep_used = true;
        if self.b.sides[me].clauses & SLEEP_BLOCKED != 0 {
            // Showdown's Sleep Clause Mod: the move fails outright.
            say!(self, "  Sleep Clause Mod activated");
            return;
        }
        let y = &mut self.b.sides[op];
        if y.bs2 & NEEDS_TO_RECHARGE != 0 {
            // A recharging target is put to sleep without any accuracy or status check.
            y.bs2 &= !NEEDS_TO_RECHARGE;
        } else {
            if y.status != 0 {
                return;
            }
            if !self.move_hit_test(me) {
                return;
            }
        }
        let n = self.ch.pick(&[1; 7]) as u8 + 1;
        self.b.sides[op].status = n;
        self.b.sides[op].clauses |= SLEPT_BY_FOE;
        say!(self, "  {} fell asleep ({n})", self.name(op));
    }

    fn poison_effect(&mut self, me: usize) {
        let op = 1 - me;
        let e = self.ram[me].effect;
        let y = &self.b.sides[op];
        if y.bs2 & HAS_SUBSTITUTE_UP != 0 || y.status != 0 || y.types.contains(&ty::POISON) {
            return;
        }
        let ok = match e {
            eff::POISON_SIDE1 => self.ch.below(52),
            eff::POISON_SIDE2 => self.ch.below(103),
            _ => self.move_hit_test(me),
        };
        if !ok {
            return;
        }
        let toxic = self.ram[me].num == self.e.dex.ids.toxic;
        let y = &mut self.b.sides[op];
        y.status |= PSN;
        if toxic {
            y.bs3 |= BADLY_POISONED;
            y.toxic = 0;
        }
        say!(self, "  {} was poisoned", self.name(op));
    }

    fn freeze_burn_paralyze(&mut self, me: usize) {
        let op = 1 - me;
        let r = self.ram[me];
        let y = &self.b.sides[op];
        if y.bs2 & HAS_SUBSTITUTE_UP != 0 {
            return;
        }
        if y.status != 0 {
            // CheckDefrost: fire moves with a burn chance thaw a frozen target.
            if y.status & FRZ != 0 && r.mtype == ty::FIRE {
                self.b.sides[op].status = 0;
                say!(self, "  {} was defrosted", self.name(op));
            }
            return;
        }
        if r.mtype == y.types[0] || r.mtype == y.types[1] {
            return;
        }
        let (thr, base) = if r.effect <= eff::PARALYZE_SIDE1 {
            (26, r.effect)
        } else {
            (77, r.effect - (eff::PARALYZE_SIDE2 - eff::PARALYZE_SIDE1))
        };
        if base == eff::FREEZE_SIDE1 && self.b.sides[me].clauses & FREEZE_BLOCKED != 0 {
            return; // Showdown's Freeze Clause Mod: the freeze chance does nothing
        }
        if !self.ch.below(thr) {
            return;
        }
        let y = &mut self.b.sides[op];
        match base {
            eff::BURN_SIDE1 => {
                y.status = BRN;
                y.stats[ATK] = (y.stats[ATK] >> 1).max(1);
            }
            eff::FREEZE_SIDE1 => {
                y.bs2 &= !NEEDS_TO_RECHARGE;
                y.status = FRZ;
            }
            _ => {
                y.status = PAR;
                y.stats[SPE] = (y.stats[SPE] >> 2).max(1);
            }
        }
        let s = status_name(self.b.sides[op].status);
        say!(self, "  {} is now {s}", self.name(op));
    }

    fn paralyze_effect(&mut self, me: usize) {
        let op = 1 - me;
        let y = &self.b.sides[op];
        if y.status != 0 {
            return;
        }
        if self.ram[me].mtype == ty::ELECTRIC && y.types.contains(&ty::GROUND) {
            return;
        }
        if !self.move_hit_test(me) {
            return;
        }
        let y = &mut self.b.sides[op];
        y.status |= PAR;
        y.stats[SPE] = (y.stats[SPE] >> 2).max(1);
        say!(self, "  {} is paralyzed", self.name(op));
    }

    fn confuse(&mut self, target: usize) {
        if self.b.sides[target].bs1 & CONFUSED != 0 {
            return;
        }
        self.b.sides[target].bs1 |= CONFUSED;
        let r = self.ch.pick(&[1, 1, 1, 1]) as u8;
        self.b.sides[target].confused = r + 2;
        say!(self, "  {} became confused", self.name(target));
    }

    /// Re-applies paralysis/burn stat penalties to `target` (QuarterSpeedDueToParalysis etc.).
    fn penalties(&mut self, target: usize) {
        let y = &mut self.b.sides[target];
        if y.status & PAR != 0 {
            y.stats[SPE] = (y.stats[SPE] >> 2).max(1);
        }
        if y.status & BRN != 0 {
            y.stats[ATK] = (y.stats[ATK] >> 1).max(1);
        }
    }

    fn stat_up(&mut self, user: usize, stat: usize, two: bool) {
        let x = &mut self.b.sides[user];
        let mut m = x.mods[stat] + 1;
        if m > 13 {
            return;
        }
        if two {
            m = (m + 1).min(13);
        }
        x.mods[stat] = m;
        if stat < 4 {
            if x.stats[stat] == 999 {
                x.mods[stat] -= 1;
                return;
            }
            let (n, d) = STAT_RATIOS[(m - 1) as usize];
            x.stats[stat] = (x.unmod[stat] as u32 * n / d).min(999) as u16;
        }
        say!(self, "  {}'s {} rose", self.name(user), STAT_NAMES[stat]);
        if user == 0 {
            self.reapply_badges();
        }
        self.penalties(1 - user);
    }

    /// `ApplyBadgeStatBoosts`, which the cartridge reapplies to every stat after the player's stat-up
    /// moves and the opponent's stat-down moves, so the boosts stack.
    fn reapply_badges(&mut self) {
        let Some(sp) = self.e.sp else { return };
        apply_badges(&mut self.b.sides[0], sp.badges);
    }

    fn stat_down(&mut self, me: usize) {
        let op = 1 - me;
        let e = self.ram[me].effect;
        // Regular battles only: the opponent's stat-down moves have a flat 25% chance to miss first.
        if self.e.sp.is_some() && me == 1 && self.ch.below(64) {
            return;
        }
        if self.b.sides[op].bs2 & HAS_SUBSTITUTE_UP != 0 {
            return;
        }
        let stat = if e >= eff::ATTACK_DOWN_SIDE {
            if !self.ch.below(85) {
                return;
            }
            (e - eff::ATTACK_DOWN_SIDE) as usize
        } else {
            if !self.move_hit_test(me) || self.b.sides[op].bs1 & INVULNERABLE != 0 {
                return;
            }
            if e >= eff::ATTACK_DOWN2 { (e - eff::ATTACK_DOWN2) as usize } else { (e - eff::ATTACK_DOWN1) as usize }
        };
        let y = &mut self.b.sides[op];
        let mut m = y.mods[stat] - 1;
        if m == 0 {
            return;
        }
        if (0x24..eff::ATTACK_DOWN_SIDE).contains(&e) {
            m -= 1;
            if m == 0 {
                m = 1;
            }
        }
        y.mods[stat] = m;
        if stat < 4 {
            if y.stats[stat] == 1 {
                y.mods[stat] += 1;
                return;
            }
            let (n, d) = STAT_RATIOS[(m - 1) as usize];
            y.stats[stat] = ((y.unmod[stat] as u32 * n / d) as u16).max(1);
        }
        say!(self, "  {}'s {} fell", self.name(op), STAT_NAMES[stat]);
        if me == 1 {
            self.reapply_badges();
        }
        self.penalties(op);
    }

    fn haze(&mut self, me: usize) {
        let op = 1 - me;
        for s in 0..2 {
            let x = &mut self.b.sides[s];
            x.mods = [7; 6];
            x.stats = x.unmod;
            x.disabled = 0;
            x.disabled_move = 0;
            x.bs1 &= !CONFUSED;
            x.bs2 &= !(1 | PROTECTED_BY_MIST | GETTING_PUMPED | SEEDED);
            x.bs3 &= !(BADLY_POISONED | HAS_LIGHT_SCREEN_UP | HAS_REFLECT_UP);
        }
        let y = &mut self.b.sides[op];
        if y.status & (FRZ | SLP_MASK) != 0 {
            y.selected = CANNOT_MOVE;
        }
        y.status = 0;
    }

    fn heal(&mut self, me: usize) {
        let max = self.e.m.mons[me].max_hp;
        let rest = self.ram[me].num == self.e.dex.ids.rest;
        let x = &mut self.b.sides[me];
        // Only the low bytes are compared (with the borrow from the high bytes): fails at full HP and
        // whenever max - current is 255 or 511.
        let borrow = (x.hp >> 8) < (max >> 8);
        if (x.hp as u8).wrapping_sub(max as u8).wrapping_sub(borrow as u8) == 0 {
            return;
        }
        let amount = if rest {
            x.status = 2;
            // Rest's sleep is self-inflicted: it doesn't count for Sleep Clause.
            x.clauses &= !SLEPT_BY_FOE;
            max
        } else {
            max >> 1
        };
        x.hp = (x.hp as u32 + amount as u32).min(max as u32) as u16;
        say!(self, "  {} restored HP", self.name(me));
    }

    fn disable(&mut self, me: usize) {
        let op = 1 - me;
        if !self.move_hit_test(me) || self.b.sides[op].disabled != 0 {
            return;
        }
        let y = &self.b.sides[op];
        let slots: Vec<usize> = (0..4).filter(|&i| y.moves[i] != 0).collect();
        let slot = slots[self.ch.pick(&vec![1; slots.len()])];
        let turns = self.ch.pick(&[1; 8]) as u8 + 1;
        let y = &mut self.b.sides[op];
        y.disabled = ((slot as u8 + 1) << 4) + turns;
        y.disabled_move = y.moves[slot];
    }
}

/// Stat scaling in GetDamageVars: if either stat exceeds a byte, divide both by 4 (keeping only the low byte).
fn scale_stats(atk: u32, def: u32) -> (u32, u32) {
    if atk > 0xFF || def > 0xFF {
        let mut a = atk >> 2;
        let d = (def >> 2) & 0xFF;
        if a == 0 {
            a = 1;
        }
        // Defense 0 freezes the real game; it cannot happen at the levels simulated here.
        (a & 0xFF, d.max(1))
    } else {
        (atk, def.max(1))
    }
}

/// CalculateDamage: ((2L/5 + 2) * power * atk / def) / 50, capped at 997, plus 2.
fn calculate_damage(level: u32, power: u32, atk: u32, def: u32, _explode: bool) -> u16 {
    let base = level * 2 / 5 + 2;
    let q = base * power * atk / def / 50;
    (q.min(997) + 2) as u16
}

/// RandomizeDamage: damage * r / 255 for r uniform in 217..=255 (no randomization below 2).
fn roll(d: u16, r: u32) -> u16 {
    if d < 2 { d } else { ((d as u32 * r / 255) & 0xFFFF) as u16 }
}

/// Merges the non-crit rolls (weight 256 - rate each) and crit rolls (weight `rate` each) by value.
fn damage_dist(crit_rate: u32, nocrit: u16, crit: u16, buckets: u8) -> DamageDist {
    // (roll, multiplicity): every roll once, or one representative roll per bucket.
    let rolls: Vec<(u32, u32)> = if buckets == 0 {
        (217..=255).map(|r| (r, 1)).collect()
    } else {
        let k = buckets as u32;
        (0..k)
            .map(|i| {
                let (lo, hi) = (217 + 39 * i / k, 217 + 39 * (i + 1) / k);
                ((lo + hi - 1) / 2, hi - lo)
            })
            .collect()
    };
    let mut pts: Vec<(u16, u32)> = rolls.iter().map(|&(r, n)| (roll(nocrit, r), n * (256 - crit_rate))).collect();
    if crit_rate > 0 {
        pts.extend(rolls.iter().map(|&(r, n)| (roll(crit, r), n * crit_rate)));
    }
    pts.sort_unstable_by_key(|p| p.0);
    let mut d = DamageDist { vals: Vec::new(), weights: Vec::new(), total: 0 };
    for (v, w) in pts {
        if d.vals.last() == Some(&v) {
            *d.weights.last_mut().unwrap() += w;
        } else {
            d.vals.push(v);
            d.weights.push(w);
        }
    }
    d.total = d.weights.iter().map(|&w| w as u64).sum();
    d
}
