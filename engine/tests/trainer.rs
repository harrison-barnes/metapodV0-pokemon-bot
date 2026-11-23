//! Hand-derived checks of single-player rules against engine/battle/{core,effects,trainer_ai}.asm.
use gen1::chance::{Enumerator, Sampler, Rng};
use gen1::data::Dex;
use gen1::state::*;
use gen1::trainer::*;
use std::sync::Arc;

fn dex() -> Dex {
    Dex::load_default()
}

fn mon(d: &Dex, sp: &str, level: u8, hp: u16, stats: [u16; 4], moves: [&str; 4]) -> Mon {
    let m = moves.map(|n| if n.is_empty() { 0 } else { d.move_id(n).unwrap_or_else(|| panic!("{n}")) });
    Mon { species: d.species_index(sp).unwrap() as u8, level, max_hp: hp, stats, moves: m }
}

fn ai(layers: &[u8], rules: Vec<AiRule>, uses: u8) -> Option<Arc<AiSpec>> {
    Some(Arc::new(AiSpec { layers: layers.to_vec(), max_uses: uses, rules, type_effects: vec![(10, 9, 20), (9, 10, 5)] }))
}

const WATER: u8 = 10;
const FIRE: u8 = 9;

#[test]
fn wild_move_rolls_are_not_quite_uniform() {
    let d = dex();
    let t = TrainerBattle::new(&d, vec![mon(&d, "RATTATA", 5, 20, [10; 4], ["TACKLE", "", "", ""])], vec![mon(&d, "PIDGEY", 5, 20, [10; 4], ["TACKLE", "GUST", "SAND_ATTACK", "QUICK_ATTACK"])], None, 0, vec![]);
    assert_eq!(t.enemy_move_weights(&d), [63, 64, 63, 66]);
}

#[test]
fn layer_one_avoids_status_when_player_has_one() {
    let d = dex();
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "RATTATA", 5, 20, [10; 4], ["TACKLE", "", "", ""])], vec![mon(&d, "GENGAR", 50, 100, [50; 4], ["HYPNOSIS", "NIGHT_SHADE", "", ""])], ai(&[1], vec![], 1), 0, vec![]);
    assert_eq!(t.enemy_move_weights(&d), [63, 64, 0, 0], "no status yet: both moves");
    t.b.sides[0].status = PSN;
    assert_eq!(t.enemy_move_weights(&d), [0, 64, 0, 0], "player poisoned: Hypnosis avoided");
}

#[test]
fn layer_two_prefers_setup_only_on_the_second_action() {
    let d = dex();
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "RATTATA", 5, 20, [10; 4], ["TACKLE", "", "", ""])], vec![mon(&d, "ARBOK", 50, 100, [50; 4], ["GLARE", "SCREECH", "BITE", ""])], ai(&[2], vec![], 1), 0, vec![]);
    assert_eq!(t.enemy_move_weights(&d), [63, 64, 63, 0]);
    t.layer2 = 1;
    // Screech's effect is DEFENSE_DOWN2, which is in the preferred range; Glare (PARALYZE) and Bite are not.
    let w = t.enemy_move_weights(&d);
    assert_eq!(w, [0, 64, 0, 0], "only Screech should remain, got {w:?}");
}

#[test]
fn layer_three_uses_first_type_match() {
    let d = dex();
    // Player is Fire: Surf is super effective, so it alone remains among the moves.
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "CHARMANDER", 5, 20, [10; 4], ["SCRATCH", "", "", ""])], vec![mon(&d, "BLASTOISE", 50, 100, [50; 4], ["SURF", "TACKLE", "", ""])], ai(&[3], vec![], 1), 0, vec![]);
    t.b.sides[0].types = [FIRE, FIRE];
    assert_eq!(t.enemy_move_weights(&d), [63, 0, 0, 0]);
    // Player is Water: Ember is resisted and Tackle is another option, so Ember is avoided.
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "SQUIRTLE", 5, 20, [10; 4], ["TACKLE", "", "", ""])], vec![mon(&d, "CHARIZARD", 50, 100, [50; 4], ["EMBER", "SCRATCH", "", ""])], ai(&[3], vec![], 1), 0, vec![]);
    t.b.sides[0].types = [WATER, WATER];
    assert_eq!(t.enemy_move_weights(&d), [0, 64, 0, 0]);
}

/// Sums the probability of each outcome of one step.
fn enumerate(d: &Dex, t: &TrainerBattle, a: PlayerAction, mut f: impl FnMut(&TrainerBattle, f64)) {
    let mut e = Enumerator::new();
    loop {
        let mut x = t.clone();
        x.step(d, a, &mut e);
        f(&x, e.prob());
        if !e.advance() {
            break;
        }
    }
}

#[test]
fn agatha_style_action_rates() {
    let d = dex();
    let rules = vec![
        AiRule { below: Some(20), hp_below: None, needs_status: false, action: AiAction::Switch },
        AiRule { below: Some(128), hp_below: Some(4), needs_status: false, action: AiAction::Heal(50) },
    ];
    let enemy = vec![mon(&d, "GENGAR", 56, 100, [50; 4], ["SPLASH", "", "", ""]), mon(&d, "GOLBAT", 56, 100, [50; 4], ["SPLASH", "", "", ""])];
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["SPLASH", "", "", ""])], enemy, ai(&[1], rules, 2), 0, vec![]);
    t.b.sides[1].hp = 24; // Below 1/4 of 100.
    let (mut switched, mut healed, mut neither, mut total) = (0.0, 0.0, 0.0, 0.0);
    enumerate(&d, &t, PlayerAction::Move(0), |x, p| {
        total += p;
        if x.active[1] == 1 {
            switched += p;
        } else if x.b.sides[1].hp == 74 {
            healed += p;
        } else {
            assert_eq!(x.b.sides[1].hp, 24);
            neither += p;
        }
    });
    assert!((total - 1.0).abs() < 1e-9);
    assert!((switched - 20.0 / 256.0).abs() < 1e-9, "switch {switched}");
    assert!((healed - 108.0 / 256.0).abs() < 1e-9, "heal {healed}");
    assert!((neither - 128.0 / 256.0).abs() < 1e-9, "neither {neither}");
}

#[test]
fn switch_needs_another_living_pokemon() {
    let d = dex();
    let rules = vec![AiRule { below: Some(256), hp_below: None, needs_status: false, action: AiAction::Switch }];
    let enemy = vec![mon(&d, "GENGAR", 56, 100, [50; 4], ["SPLASH", "", "", ""])];
    let t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["SPLASH", "", "", ""])], enemy, ai(&[], rules, 2), 0, vec![]);
    enumerate(&d, &t, PlayerAction::Move(0), |x, _| assert_eq!(x.active[1], 0));
}

#[test]
fn enemy_stat_down_misses_a_quarter_of_the_time() {
    let d = dex();
    // Screech at 100% accuracy-equivalent stages: 85% accuracy, then a 64/256 flat miss first.
    let t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["SPLASH", "", "", ""])], vec![mon(&d, "ARBOK", 50, 100, [50; 4], ["SCREECH", "", "", ""])], None, 0, vec![]);
    let mut landed = 0.0;
    enumerate(&d, &t, PlayerAction::Move(0), |x, p| {
        if x.b.sides[0].mods[DEF] < 7 {
            landed += p;
        }
    });
    // 3/4 pass the flat miss. Screech's own accuracy is 85%: threshold 216/256 (accuracy data is scaled to 255).
    let acc = d.mv(d.move_id("SCREECH").unwrap()).accuracy as f64;
    let expected = 192.0 / 256.0 * (acc / 256.0);
    assert!((landed - expected).abs() < 1e-9, "landed {landed}, expected {expected}");
}

#[test]
fn enemy_stat_down_stacks_badge_boosts_on_the_player() {
    let d = dex();
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100, 200, 300, 400], ["SPLASH", "", "", ""])], vec![mon(&d, "ARBOK", 50, 100, [50; 4], ["SCREECH", "", "", ""])], None, 0xFF, vec![]);
    assert_eq!(t.b.sides[0].stats, [112, 225, 337, 450], "boosted once at send-out");
    t.b.sides[1].mods = [7; 6];
    enumerate(&d, &t, PlayerAction::Move(0), |x, _| {
        let s = x.b.sides[0].stats;
        if x.b.sides[0].mods[DEF] < 7 {
            // Attack, speed and special boost again by 1/8 of their current values. Defense is
            // recalculated from the unmodified 200 at stage 5 (50%) = 100, then boosted to 112.
            assert_eq!(s, [126, 112, 379, 506]);
        } else {
            assert_eq!(s, [112, 225, 337, 450]);
        }
    });
}

#[test]
fn enemy_pp_is_never_spent_and_player_pp_is() {
    let d = dex();
    let t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["TACKLE", "", "", ""])], vec![mon(&d, "ARBOK", 50, 100, [50; 4], ["TACKLE", "", "", ""])], None, 0, vec![]);
    let start = t.teams[0][0].pp[0];
    let mut s = Sampler { rng: Rng::new(1) };
    let mut x = t.clone();
    x.step(&d, PlayerAction::Move(0), &mut s);
    assert_eq!(x.teams[0][0].pp[0], start - 1);
    assert_eq!(x.teams[1][0].pp[0], t.teams[1][0].pp[0]);
}

#[test]
fn replacement_is_first_living_and_faint_wins_when_last() {
    let d = dex();
    let enemy = vec![mon(&d, "PIDGEY", 5, 1, [5; 4], ["SPLASH", "", "", ""]), mon(&d, "RATTATA", 5, 20, [5; 4], ["SPLASH", "", "", ""]), mon(&d, "ZUBAT", 5, 20, [5; 4], ["SPLASH", "", "", ""])];
    let t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [200; 4], ["SWIFT", "", "", ""])], enemy, ai(&[], vec![], 3), 0, vec![]);
    enumerate(&d, &t, PlayerAction::Move(0), |x, _| {
        assert_eq!(x.active[1], 1, "Rattata comes next");
        assert_eq!(x.result, None);
    });
    let last = vec![mon(&d, "PIDGEY", 5, 1, [5; 4], ["SPLASH", "", "", ""])];
    let t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [200; 4], ["SWIFT", "", "", ""])], last, None, 0, vec![]);
    enumerate(&d, &t, PlayerAction::Move(0), |x, _| assert_eq!(x.result, Some(Finish::Win)));
}

#[test]
fn double_ko_on_both_last_pokemon_is_a_black_out_whoever_attacked() {
    let d = dex();
    for (p, e) in [
        (mon(&d, "ELECTRODE", 50, 1, [200; 4], ["EXPLOSION", "", "", ""]), mon(&d, "PIDGEY", 5, 1, [5; 4], ["SPLASH", "", "", ""])),
        (mon(&d, "PIDGEY", 5, 1, [5; 4], ["SPLASH", "", "", ""]), mon(&d, "ELECTRODE", 50, 1, [200; 4], ["EXPLOSION", "", "", ""])),
    ] {
        let t = TrainerBattle::new(&d, vec![p], vec![e], None, 0, vec![]);
        enumerate(&d, &t, PlayerAction::Move(0), |x, p| {
            if x.teams[0][0].hp == 0 && x.teams[1][0].hp == 0 && p > 0.0 {
                assert_eq!(x.result, Some(Finish::Loss));
            }
        });
    }
}

#[test]
fn after_a_double_ko_the_player_picks_first_then_the_game_replaces() {
    let d = dex();
    let player = vec![mon(&d, "ELECTRODE", 50, 1, [200; 4], ["EXPLOSION", "", "", ""]), mon(&d, "RATTATA", 50, 50, [50; 4], ["TACKLE", "", "", ""])];
    let enemy = vec![mon(&d, "PIDGEY", 5, 1, [5; 4], ["SPLASH", "", "", ""]), mon(&d, "ZUBAT", 5, 20, [5; 4], ["SPLASH", "", "", ""])];
    let t = TrainerBattle::new(&d, player, enemy, ai(&[], vec![], 3), 0, vec![]);
    let mut seen = false;
    enumerate(&d, &t, PlayerAction::Move(0), |x, _| {
        if x.teams[0][0].hp == 0 && x.teams[1][0].hp == 0 {
            seen = true;
            assert!(x.need_replace);
            assert_eq!(x.active[1], 0, "the fainted foe stays while the player chooses");
            assert_eq!(x.b.sides[1].hp, 0);
            let mut y = x.clone();
            let mut s = Sampler { rng: Rng::new(1) };
            y.step(&d, PlayerAction::Switch(1), &mut s);
            assert_eq!(y.active, [1, 1], "then both new Pokémon are in");
            assert_eq!(y.b.sides[1].hp, 20);
            assert!(!y.need_replace);
        }
    });
    assert!(seen);
}

#[test]
fn fainting_clears_the_active_status() {
    let d = dex();
    let player = vec![mon(&d, "PIDGEY", 5, 1, [5; 4], ["SPLASH", "", "", ""]), mon(&d, "RATTATA", 50, 50, [50; 4], ["TACKLE", "", "", ""])];
    let mut t = TrainerBattle::new(&d, player, vec![mon(&d, "ARBOK", 50, 100, [100; 4], ["SWIFT", "", "", ""])], None, 0, vec![]);
    t.b.sides[0].status = PAR;
    enumerate(&d, &t, PlayerAction::Move(0), |x, _| {
        if x.need_replace {
            assert_eq!(x.b.sides[0].status, 0);
        }
    });
}

#[test]
fn full_restore_on_a_hurt_pokemon_keeps_the_toxic_flag() {
    let d = dex();
    let bag = vec![BagItem { kind: ItemKind::FullRestore, count: 1 }];
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["SPLASH", "", "", ""])], vec![mon(&d, "PIDGEY", 5, 20, [5; 4], ["SPLASH", "", "", ""])], None, 0, bag.clone());
    t.b.sides[0].hp = 100;
    t.teams[0][0].hp = 100;
    t.b.sides[0].status = PSN;
    t.b.sides[0].bs3 |= BADLY_POISONED;
    let mut s = Sampler { rng: Rng::new(1) };
    t.clone().step(&d, PlayerAction::Item(0), &mut s);
    let mut x = t.clone();
    x.step(&d, PlayerAction::Item(0), &mut s);
    assert_eq!(x.b.sides[0].status, 0);
    assert_ne!(x.b.sides[0].bs3 & BADLY_POISONED, 0, "only the cure path clears it");
    // At full HP it goes down the cure path instead, which does clear the flag.
    let mut f = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["SPLASH", "", "", ""])], vec![mon(&d, "PIDGEY", 5, 20, [5; 4], ["SPLASH", "", "", ""])], None, 0, bag);
    f.b.sides[0].status = PSN;
    f.b.sides[0].bs3 |= BADLY_POISONED;
    f.step(&d, PlayerAction::Item(0), &mut s);
    assert_eq!(f.b.sides[0].bs3 & BADLY_POISONED, 0);
}

#[test]
fn items_heal_and_cure() {
    let d = dex();
    let bag = vec![BagItem { kind: ItemKind::Heal(50), count: 1 }, BagItem { kind: ItemKind::Cure(PSN), count: 1 }];
    let mut t = TrainerBattle::new(&d, vec![mon(&d, "SNORLAX", 50, 200, [100; 4], ["SPLASH", "", "", ""])], vec![mon(&d, "PIDGEY", 5, 20, [5; 4], ["SPLASH", "", "", ""])], None, 0, bag);
    t.b.sides[0].hp = 100;
    t.teams[0][0].hp = 100;
    let acts = t.player_actions(&d);
    assert!(acts.contains(&PlayerAction::Item(0)));
    assert!(!acts.contains(&PlayerAction::Item(1)), "nothing to cure");
    let mut s = Sampler { rng: Rng::new(1) };
    t.step(&d, PlayerAction::Item(0), &mut s);
    assert_eq!(t.b.sides[0].hp, 150);
    assert_eq!(t.bag[0].count, 0);
}
