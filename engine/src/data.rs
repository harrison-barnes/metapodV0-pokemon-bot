//! Gen 1 data loaded from `data/gen1.json` (extracted from the pret/pokered disassembly).

use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::path::Path;

/// Move effect ids, numerically identical to `constants/move_effect_constants.asm`.
pub mod eff {
    pub const NONE: u8 = 0x00;
    pub const POISON_SIDE1: u8 = 0x02;
    pub const DRAIN_HP: u8 = 0x03;
    pub const BURN_SIDE1: u8 = 0x04;
    pub const FREEZE_SIDE1: u8 = 0x05;
    pub const PARALYZE_SIDE1: u8 = 0x06;
    pub const EXPLODE: u8 = 0x07;
    pub const DREAM_EATER: u8 = 0x08;
    pub const MIRROR_MOVE: u8 = 0x09;
    pub const ATTACK_UP1: u8 = 0x0A;
    pub const EVASION_UP1: u8 = 0x0F;
    pub const PAY_DAY: u8 = 0x10;
    pub const SWIFT: u8 = 0x11;
    pub const ATTACK_DOWN1: u8 = 0x12;
    pub const EVASION_DOWN1: u8 = 0x17;
    pub const CONVERSION: u8 = 0x18;
    pub const ACCURACY_DOWN1_ID: u8 = 0x16;
    pub const HAZE: u8 = 0x19;
    pub const BIDE: u8 = 0x1A;
    pub const THRASH: u8 = 0x1B;
    pub const SWITCH_TELEPORT: u8 = 0x1C;
    pub const TWO_TO_FIVE: u8 = 0x1D;
    pub const EFFECT_1E: u8 = 0x1E;
    pub const FLINCH_SIDE1: u8 = 0x1F;
    pub const SLEEP: u8 = 0x20;
    pub const POISON_SIDE2: u8 = 0x21;
    pub const BURN_SIDE2: u8 = 0x22;
    pub const FREEZE_SIDE2: u8 = 0x23;
    pub const PARALYZE_SIDE2: u8 = 0x24;
    pub const FLINCH_SIDE2: u8 = 0x25;
    pub const OHKO: u8 = 0x26;
    pub const CHARGE: u8 = 0x27;
    pub const SUPER_FANG: u8 = 0x28;
    pub const SPECIAL_DAMAGE: u8 = 0x29;
    pub const TRAPPING: u8 = 0x2A;
    pub const FLY: u8 = 0x2B;
    pub const ATTACK_TWICE: u8 = 0x2C;
    pub const JUMP_KICK: u8 = 0x2D;
    pub const MIST: u8 = 0x2E;
    pub const FOCUS_ENERGY: u8 = 0x2F;
    pub const RECOIL: u8 = 0x30;
    pub const CONFUSION: u8 = 0x31;
    pub const ATTACK_UP2: u8 = 0x32;
    pub const EVASION_UP2: u8 = 0x37;
    pub const HEAL: u8 = 0x38;
    pub const TRANSFORM: u8 = 0x39;
    pub const ATTACK_DOWN2: u8 = 0x3A;
    pub const ACCURACY_DOWN2_ID: u8 = 0x3E;
    pub const EVASION_DOWN2: u8 = 0x3F;
    pub const LIGHT_SCREEN: u8 = 0x40;
    pub const REFLECT: u8 = 0x41;
    pub const POISON: u8 = 0x42;
    pub const PARALYZE: u8 = 0x43;
    pub const ATTACK_DOWN_SIDE: u8 = 0x44;
    pub const SPECIAL_DOWN_SIDE: u8 = 0x47;
    pub const CONFUSION_SIDE: u8 = 0x4C;
    pub const TWINEEDLE: u8 = 0x4D;
    pub const SUBSTITUTE: u8 = 0x4F;
    pub const HYPER_BEAM: u8 = 0x50;
    pub const RAGE: u8 = 0x51;
    pub const MIMIC: u8 = 0x52;
    pub const METRONOME: u8 = 0x53;
    pub const LEECH_SEED: u8 = 0x54;
    pub const SPLASH: u8 = 0x55;
    pub const DISABLE: u8 = 0x56;

    /// `data/battle/residual_effects_1.asm`: skip damage calculation and accuracy entirely.
    pub fn residual1(e: u8) -> bool {
        matches!(
            e,
            CONVERSION | HAZE | SWITCH_TELEPORT | MIST | FOCUS_ENERGY | CONFUSION | HEAL | TRANSFORM
                | LIGHT_SCREEN | REFLECT | POISON | PARALYZE | SUBSTITUTE | MIMIC | LEECH_SEED | SPLASH
        )
    }

    /// `data/battle/residual_effects_2.asm`: stat changes, sleep and Bide.
    pub fn residual2(e: u8) -> bool {
        matches!(e, 0x01 | ATTACK_UP1..=EVASION_UP1 | ATTACK_DOWN1..=EVASION_DOWN1 | BIDE | SLEEP
            | ATTACK_UP2..=EVASION_UP2 | ATTACK_DOWN2..=EVASION_DOWN2)
    }

    /// `SpecialEffectsCont`: run before damage calculation.
    pub fn special_cont(e: u8) -> bool {
        matches!(e, THRASH | TRAPPING)
    }

    /// `SpecialEffects` (which falls through into `SpecialEffectsCont`).
    pub fn special(e: u8) -> bool {
        matches!(
            e,
            DRAIN_HP | EXPLODE | DREAM_EATER | PAY_DAY | SWIFT | TWO_TO_FIVE | EFFECT_1E | CHARGE | SUPER_FANG
                | SPECIAL_DAMAGE | FLY | ATTACK_TWICE | JUMP_KICK | RECOIL
        ) || special_cont(e)
    }

    /// `AlwaysHappenSideEffects`: still run when the attack faints the target.
    pub fn always_happen(e: u8) -> bool {
        matches!(
            e,
            DRAIN_HP | EXPLODE | DREAM_EATER | PAY_DAY | TWO_TO_FIVE | EFFECT_1E | ATTACK_TWICE | RECOIL | TWINEEDLE
                | RAGE
        )
    }

    pub fn set_damage(e: u8) -> bool {
        matches!(e, SUPER_FANG | SPECIAL_DAMAGE)
    }
}

/// Dense type indices produced by the extractor (see `data/gen1.json` "types").
pub mod ty {
    pub const NORMAL: u8 = 0;
    pub const FIGHTING: u8 = 1;
    pub const POISON: u8 = 3;
    pub const GROUND: u8 = 4;
    pub const BUG: u8 = 7;
    pub const GHOST: u8 = 8;
    pub const FIRE: u8 = 9;
    pub const GRASS: u8 = 11;
    pub const ELECTRIC: u8 = 12;
}

#[derive(Debug, Clone, Deserialize)]
pub struct MoveData {
    pub id: u8,
    pub name: String,
    pub effect_id: u8,
    pub power: u8,
    #[serde(rename = "type")]
    pub mtype: u8,
    /// Accuracy on the cartridge's 0-255 scale.
    pub accuracy: u8,
    pub pp: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BaseStats {
    pub hp: u8,
    pub atk: u8,
    pub def: u8,
    pub spe: u8,
    pub spc: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Species {
    pub name: String,
    pub dex: u8,
    pub base: BaseStats,
    pub types: [u8; 2],
    pub learnset: Vec<u8>,
    pub fully_evolved: bool,
}

#[derive(Deserialize)]
struct RawData {
    types: Vec<String>,
    special_threshold: u8,
    type_chart: Vec<[u8; 3]>,
    moves: Vec<Option<MoveData>>,
    species: Vec<Species>,
}

/// Move ids the engine special-cases by number, looked up by name at load time.
#[derive(Debug, Clone, Copy)]
pub struct MoveIds {
    pub quick_attack: u8,
    pub counter: u8,
    pub seismic_toss: u8,
    pub night_shade: u8,
    pub sonicboom: u8,
    pub dragon_rage: u8,
    pub rest: u8,
    pub toxic: u8,
    pub dig: u8,
    pub struggle: u8,
    pub rage: u8,
    pub thrash: u8,
    pub bide: u8,
    pub high_crit: [u8; 4],
}

pub struct Dex {
    pub types: Vec<String>,
    pub special_threshold: u8,
    /// (attacking type, defending type, multiplier x10), in cartridge table order.
    pub type_chart: Vec<[u8; 3]>,
    /// Indexed by move id; index 0 is a placeholder for "no move".
    pub moves: Vec<MoveData>,
    /// Indexed by dex number - 1.
    pub species: Vec<Species>,
    pub ids: MoveIds,
    /// For (move type, defender type 1, defender type 2): the multipliers (x10) that apply, in table order.
    type_mults: Vec<[u8; 3]>,
    move_by_name: FxHashMap<String, u8>,
    species_by_name: FxHashMap<String, usize>,
}

impl Dex {
    pub fn load(path: &Path) -> Result<Dex, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let raw: RawData = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut moves = vec![MoveData {
            id: 0,
            name: "NO_MOVE".into(),
            effect_id: 0,
            power: 0,
            mtype: 0,
            accuracy: 0,
            pp: 0,
        }];
        moves.extend(raw.moves.into_iter().flatten());
        let move_by_name: FxHashMap<String, u8> = moves.iter().map(|m| (m.name.clone(), m.id)).collect();
        let species_by_name = raw.species.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
        let id = |n: &str| *move_by_name.get(n).unwrap_or_else(|| panic!("missing move {n}"));
        let ids = MoveIds {
            quick_attack: id("QUICK_ATTACK"),
            counter: id("COUNTER"),
            seismic_toss: id("SEISMIC_TOSS"),
            night_shade: id("NIGHT_SHADE"),
            sonicboom: id("SONICBOOM"),
            dragon_rage: id("DRAGON_RAGE"),
            rest: id("REST"),
            toxic: id("TOXIC"),
            dig: id("DIG"),
            struggle: id("STRUGGLE"),
            rage: id("RAGE"),
            thrash: id("THRASH"),
            bide: id("BIDE"),
            high_crit: [id("KARATE_CHOP"), id("RAZOR_LEAF"), id("CRABHAMMER"), id("SLASH")],
        };
        let n = raw.types.len();
        let mut type_mults = vec![[0u8; 3]; n * n * n];
        for a in 0..n {
            for d1 in 0..n {
                for d2 in 0..n {
                    let e = &mut type_mults[(a * n + d1) * n + d2];
                    for &[ta, td, m] in &raw.type_chart {
                        if ta as usize == a && (td as usize == d1 || td as usize == d2) {
                            e[0] += 1;
                            e[e[0] as usize] = m;
                        }
                    }
                }
            }
        }
        Ok(Dex {
            type_mults,
            types: raw.types,
            special_threshold: raw.special_threshold,
            type_chart: raw.type_chart,
            moves,
            species: raw.species,
            ids,
            move_by_name,
            species_by_name,
        })
    }

    /// Loads `data/gen1.json` relative to the repository root.
    pub fn load_default() -> Dex {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/gen1.json");
        Dex::load(&root).expect("load gen1.json")
    }

    pub fn move_id(&self, name: &str) -> Option<u8> {
        self.move_by_name.get(&name.to_ascii_uppercase().replace([' ', '-'], "_")).copied()
    }

    pub fn species_index(&self, name: &str) -> Option<usize> {
        self.species_by_name.get(&name.to_ascii_uppercase()).copied()
    }

    /// Type-chart multipliers (x10) for a move type against a defender's two types, in cartridge order.
    pub fn type_multipliers(&self, atk: u8, d1: u8, d2: u8) -> &[u8] {
        let n = self.types.len();
        let e = &self.type_mults[(atk as usize * n + d1 as usize) * n + d2 as usize];
        &e[1..1 + e[0] as usize]
    }

    pub fn mv(&self, id: u8) -> &MoveData {
        &self.moves[id as usize]
    }

    pub fn is_special(&self, t: u8) -> bool {
        t >= self.special_threshold
    }
}
