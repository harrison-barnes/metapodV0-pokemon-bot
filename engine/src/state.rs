//! Battle state for a single-Pokemon (1v1) battle, mirroring the cartridge's battle RAM.

use crate::data::Dex;

// wBattleMonStatus bits (constants/battle_constants.asm). Bits 0-2 hold the sleep counter.
pub const SLP_MASK: u8 = 0b111;
pub const PSN: u8 = 1 << 3;
pub const BRN: u8 = 1 << 4;
pub const FRZ: u8 = 1 << 5;
pub const PAR: u8 = 1 << 6;

// wPlayerBattleStatus1
pub const STORING_ENERGY: u8 = 1 << 0; // Bide
pub const THRASHING_ABOUT: u8 = 1 << 1;
pub const ATTACKING_MULTIPLE_TIMES: u8 = 1 << 2;
pub const FLINCHED: u8 = 1 << 3;
pub const CHARGING_UP: u8 = 1 << 4;
pub const USING_TRAPPING_MOVE: u8 = 1 << 5;
pub const INVULNERABLE: u8 = 1 << 6; // Fly/Dig
pub const CONFUSED: u8 = 1 << 7;

// wPlayerBattleStatus2
pub const PROTECTED_BY_MIST: u8 = 1 << 1;
pub const GETTING_PUMPED: u8 = 1 << 2; // Focus Energy
pub const HAS_SUBSTITUTE_UP: u8 = 1 << 4;
pub const NEEDS_TO_RECHARGE: u8 = 1 << 5;
pub const USING_RAGE: u8 = 1 << 6;
pub const SEEDED: u8 = 1 << 7;

// wPlayerBattleStatus3
pub const BADLY_POISONED: u8 = 1 << 0;
pub const HAS_LIGHT_SCREEN_UP: u8 = 1 << 1;
pub const HAS_REFLECT_UP: u8 = 1 << 2;
pub const TRANSFORMED: u8 = 1 << 3;

/// `Side::clauses` bits: Smogon clause mods that depend on the rest of the team. Set means the
/// rest of the opposing team already has a Pokemon asleep (frozen), so this side's sleep moves fail
/// (its attacks can't freeze), as with Showdown's Sleep Clause Mod and Freeze Clause Mod.
pub const SLEEP_BLOCKED: u8 = 1 << 0;
pub const FREEZE_BLOCKED: u8 = 1 << 1;
/// Marks a Pokemon put to sleep by the opponent's move, not by Rest (what Sleep Clause counts).
/// Only the 6v6 layer reads it; `Engine::canonicalize` drops it from 1v1 states.
pub const SLEPT_BY_FOE: u8 = 1 << 2;
/// Bits 3-7 of `Side::clauses`, used only under `Rules::heal_pp`: how much PP the side's healing
/// move (Recover, Softboiled, Rest) has left, as a level. Level L stands for `heal_level_pp(L)`
/// uses: 0, 1, 2, 4, 8, ... capped at the move's maximum PP.
pub const HEAL_PP_SHIFT: u8 = 3;
pub const HEAL_PP_MASK: u8 = 0b1111_1000;

/// The number of uses healing-PP level `level` stands for, for a move with `max_pp` PP.
pub fn heal_level_pp(level: u8, max_pp: u8) -> u8 {
    if level == 0 { 0 } else { (1u16 << (level - 1)).min(max_pp as u16) as u8 }
}

/// The level that stands for a full `max_pp`: the first whose count reaches it.
pub fn heal_top_level(max_pp: u8) -> u8 {
    (1..=8).find(|&l| heal_level_pp(l, max_pp) >= max_pp).unwrap()
}

/// Value of wPlayerSelectedMove when a trapped Pokemon cannot act.
pub const CANNOT_MOVE: u8 = 0xFF;

pub const ATK: usize = 0;
pub const DEF: usize = 1;
pub const SPE: usize = 2;
pub const SPC: usize = 3;
pub const ACC: usize = 4;
pub const EVA: usize = 5;

/// Fixed per-battle facts about one side (never change during the battle, so not part of the hashed state).
#[derive(Debug, Clone)]
pub struct Mon {
    pub species: u8, // index into Dex::species
    pub level: u8,
    pub max_hp: u16,
    /// Party (unmodified) attack, defense, speed, special. Critical hits read these.
    pub stats: [u16; 4],
    pub moves: [u8; 4],
}

/// Everything about one battler that can change during a battle.
///
/// `repr(C)` with all u16 fields first and an explicit pad byte: the struct has no padding, so a
/// whole `Battle` can be copied byte for byte into a hash key (see `Battle::pack`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Side {
    pub hp: u16,
    /// Current (modified) attack, defense, speed, special as stored in battle RAM.
    pub stats: [u16; 4],
    /// "Unmodified" stats that stat-stage recalculation starts from (changed only by Transform).
    pub unmod: [u16; 4],
    pub bide_dmg: u16,
    pub status: u8,
    /// Stat stages for atk, def, spe, spc, acc, eva; 7 is neutral, range 1..=13.
    pub mods: [u8; 6],
    pub bs1: u8,
    pub bs2: u8,
    pub bs3: u8,
    pub confused: u8,
    pub toxic: u8,
    /// High nibble: disabled slot + 1, low nibble: turns left.
    pub disabled: u8,
    pub disabled_move: u8,
    pub attacks_left: u8,
    pub sub_hp: u8,
    /// wPlayerSelectedMove. Survives turns where no move is chosen (sleep, recharge, ...).
    pub selected: u8,
    /// Move data left in battle RAM from this side's last action: move id, power, type and the
    /// (possibly rescaled) accuracy. Rage/Thrash reuse it, and the opponent's Counter reads it.
    pub ram: [u8; 4],
    pub types: [u8; 2],
    pub moves: [u8; 4],
    /// Species used for the critical hit rate (changes on Transform).
    pub species: u8,
    /// Sleep clause bookkeeping: this side has already attempted a sleep move.
    pub sleep_used: bool,
    /// `SLEEP_BLOCKED` / `FREEZE_BLOCKED` (fills what would otherwise be a padding byte).
    pub clauses: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Battle {
    pub sides: [Side; 2],
    /// wDamage: shared by both sides, read by Counter, Bide and trapping-move continuation.
    pub damage: u16,
}

/// Battle packed into fixed-size words: a fast hash-map key for the solver.
pub type Packed = [u64; 13];

const _: () = assert!(std::mem::size_of::<Side>() == 50 && std::mem::size_of::<Battle>() == 102);

impl Battle {
    pub fn pack(&self) -> Packed {
        let mut out = [0u64; 13];
        // SAFETY: Battle is repr(C), made only of integers and bools, and has no padding bytes
        // (checked by the size assertion above), so all 102 bytes are initialized.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self as *const Battle as *const u8,
                out.as_mut_ptr() as *mut u8,
                std::mem::size_of::<Battle>(),
            );
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    Win(usize),
    Draw,
}

/// How a battle is set up: the same for every Pokemon in a tournament.
#[derive(Debug, Clone)]
pub struct Rules {
    pub level: u8,
    /// DV used for atk/def/spe/spc (HP DV is derived from these as on the cartridge).
    pub dv: u8,
    pub stat_exp: u16,
    /// Each side may attempt a sleep-inducing move only once per battle.
    pub sleep_clause: bool,
    /// Moves left out of every moveset, on top of `meta::EXCLUDED`.
    pub banned_moves: &'static [&'static str],
    /// Species left out of the field.
    pub banned_species: &'static [&'static str],
    /// Track the healing move's PP (coarsely, see `HEAL_PP_MASK`), so recovery stalls end when it
    /// runs out instead of looping forever. Other moves keep unlimited PP.
    pub heal_pp: bool,
    /// 6v6 only (`battle6`): three places where Pokemon Showdown's Gen 1 OU differs from what `battle6` did, found by
    /// replaying Showdown's own battles turn by turn (agent/test_engine_vs_showdown.py). Off by default, so existing
    /// results don't move.
    /// - Sleep Clause is judged when the sleep move lands, not at the start of the turn: a Pokemon that woke up
    ///   earlier in the turn can be put back to sleep.
    /// - The target of a Wrap/Clamp/Bind/Fire Spin switches out: the trapper uses that move again, afresh, on the
    ///   Pokemon that comes in (`battle6` played the move in its first slot).
    /// - The trapper switches out: the Pokemon it held had no move to choose and makes none this turn (`battle6`
    ///   played the move in its first slot).
    pub showdown: bool,
}

impl Default for Rules {
    /// The reference video's rules: level 50, Explosion banned (in `meta::EXCLUDED`), one sleep
    /// attempt per side.
    fn default() -> Self {
        Rules { level: 50, dv: 15, stat_exp: 65535, sleep_clause: true, banned_moves: &[], banned_species: &[], heal_pp: false, showdown: false }
    }
}

impl Rules {
    /// Smogon's RBY OU rules, as far as they apply to a 1v1 battle: level 100, Mew and Mewtwo
    /// banned, Evasion Clause and OHKO Clause. Sleep Clause and Freeze Clause limit how many
    /// *different* opposing Pokemon may be asleep or frozen, so with one Pokemon a side they never
    /// apply and sleep moves can be used repeatedly. Explosion stays banned (`meta::EXCLUDED`):
    /// with one Pokemon a side it only ever ends the battle in a double KO.
    pub fn smogon() -> Self {
        Rules {
            level: 100,
            sleep_clause: false,
            banned_moves: &["DOUBLE_TEAM", "MINIMIZE", "FISSURE", "HORN_DRILL", "GUILLOTINE"],
            banned_species: &["MEW", "MEWTWO"],
            ..Rules::default()
        }
    }

    /// `smogon` with the healing move's PP tracked (`heal_pp`).
    pub fn smogon_pp() -> Self {
        Rules { heal_pp: true, ..Rules::smogon() }
    }

    /// `video` (the default), `smogon`, `smogon-pp`, or `smogon-sd` (`smogon` with `showdown` set).
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "video" => Some(Rules::default()),
            "smogon" => Some(Rules::smogon()),
            "smogon-pp" => Some(Rules::smogon_pp()),
            "smogon-sd" => Some(Rules { showdown: true, ..Rules::smogon() }),
            _ => None,
        }
    }
}

/// `CalcStat` from home/move_mon.asm.
pub fn calc_stat(base: u8, dv: u8, stat_exp: u16, level: u8, is_hp: bool) -> u16 {
    let mut b: u32 = 0;
    // ceil(sqrt(stat_exp)), capped at 255 by the loop in CalcStat
    loop {
        b += 1;
        if b == 255 || b * b >= stat_exp as u32 {
            break;
        }
    }
    if stat_exp == 0 {
        b = 0;
    }
    let v = ((base as u32 + dv as u32) * 2 + b / 4) * level as u32 / 100;
    (if is_hp { v + level as u32 + 10 } else { v + 5 }) as u16
}

impl Mon {
    pub fn new(dex: &Dex, species: usize, moves: &[u8], rules: &Rules) -> Mon {
        let s = &dex.species[species];
        let dv = rules.dv;
        // HP DV is built from the low bit of each other DV.
        let hp_dv = ((dv & 1) << 3) | ((dv & 1) << 2) | ((dv & 1) << 1) | (dv & 1);
        let st = |b| calc_stat(b, dv, rules.stat_exp, rules.level, false);
        let mut mv = [0u8; 4];
        for (i, &m) in moves.iter().take(4).enumerate() {
            mv[i] = m;
        }
        Mon {
            species: species as u8,
            level: rules.level,
            max_hp: calc_stat(s.base.hp, hp_dv, rules.stat_exp, rules.level, true),
            stats: [st(s.base.atk), st(s.base.def), st(s.base.spe), st(s.base.spc)],
            moves: mv,
        }
    }

    pub fn fresh_side(&self, dex: &Dex) -> Side {
        Side {
            hp: self.max_hp,
            stats: self.stats,
            unmod: self.stats,
            bide_dmg: 0,
            status: 0,
            mods: [7; 6],
            bs1: 0,
            bs2: 0,
            bs3: 0,
            confused: 0,
            toxic: 0,
            disabled: 0,
            disabled_move: 0,
            attacks_left: 0,
            sub_hp: 0,
            selected: 0,
            ram: [0; 4],
            types: dex.species[self.species as usize].types,
            moves: self.moves,
            species: self.species,
            sleep_used: false,
            clauses: 0,
        }
    }
}

/// The two Pokemon in a battle plus the rules. Immutable for the whole battle.
#[derive(Debug, Clone)]
pub struct Matchup {
    pub mons: [Mon; 2],
    pub rules: Rules,
}

impl Matchup {
    pub fn initial(&self, dex: &Dex) -> Battle {
        let mut b = Battle { sides: [self.mons[0].fresh_side(dex), self.mons[1].fresh_side(dex)], damage: 0 };
        for s in 0..2 {
            b.sides[s].clauses |= self.heal_pp_bits(dex, s);
        }
        b
    }

    /// Side s's healing move's maximum PP (with PP Ups), if it has one and `heal_pp` is on.
    pub fn heal_max_pp(&self, dex: &Dex, s: usize) -> Option<u8> {
        if !self.rules.heal_pp {
            return None;
        }
        self.mons[s].moves.iter().find(|&&mv| mv != 0 && dex.mv(mv).effect_id == crate::data::eff::HEAL).map(|&mv| dex.mv(mv).pp / 5 * 8)
    }

    /// The `Side::clauses` bits for a full healing move.
    fn heal_pp_bits(&self, dex: &Dex, s: usize) -> u8 {
        self.heal_max_pp(dex, s).map_or(0, |pp| heal_top_level(pp) << HEAL_PP_SHIFT)
    }

    /// A position as it stands right after both Pokemon have been sent out mid-battle: the given HP
    /// and major status (sleep counter 1-7 in the low bits, or PSN / BRN / FRZ / PAR) carried over,
    /// and everything volatile fresh.
    ///
    /// As on the cartridge (`LoadBattleMonFromParty` calls `ApplyBurnAndParalysisPenaltiesToPlayer`)
    /// and in Showdown (`setSpecies`), a Pokemon entering the battle has its stats reloaded with the
    /// paralysis Speed quartering and burn Attack halving applied. Toxic has already become ordinary
    /// poison (switching out clears it).
    pub fn entry(&self, dex: &Dex, hp: [u16; 2], status: [u8; 2], clauses: [u8; 2]) -> Battle {
        let mut b = self.initial(dex);
        for s in 0..2 {
            assert!(hp[s] >= 1 && hp[s] <= self.mons[s].max_hp, "HP {} out of range", hp[s]);
            let x = &mut b.sides[s];
            x.hp = hp[s];
            x.status = status[s];
            x.clauses = clauses[s] | self.heal_pp_bits(dex, s);
            if status[s] & PAR != 0 {
                x.stats[SPE] = (x.stats[SPE] >> 2).max(1);
            }
            if status[s] & BRN != 0 {
                x.stats[ATK] = (x.stats[ATK] >> 1).max(1);
            }
        }
        b
    }

    pub fn either_has(&self, mv: u8) -> bool {
        self.mons.iter().any(|m| m.moves.contains(&mv))
    }
}

pub fn status_name(s: u8) -> &'static str {
    if s & SLP_MASK != 0 {
        "SLP"
    } else if s & PSN != 0 {
        "PSN"
    } else if s & BRN != 0 {
        "BRN"
    } else if s & FRZ != 0 {
        "FRZ"
    } else if s & PAR != 0 {
        "PAR"
    } else {
        ""
    }
}
