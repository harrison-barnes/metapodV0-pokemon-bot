//! 6v6 mechanics: switching, fainting and clauses, checked against core.asm behaviour.

use gen1::battle6::{Action, Battle6};
use gen1::chance::{Rng, Sampler};
use gen1::cli::parse_mon;
use gen1::data::Dex;
use gen1::state::*;

fn team(d: &Dex, specs: &[&str]) -> Vec<Mon> {
    specs.iter().map(|s| parse_mon(d, s, &Rules::smogon()).unwrap()).collect()
}

fn battle(d: &Dex, a: &[&str], b: &[&str]) -> Battle6 {
    Battle6::new(d, &Rules::smogon(), [team(d, a), team(d, b)])
}

fn rng() -> Sampler {
    Sampler { rng: Rng::new(7) }
}

#[test]
fn switching_keeps_hp_and_status_and_applies_entry_penalties() {
    let d = Dex::load_default();
    let mut g = battle(&d, &["JOLTEON:SPLASH", "TAUROS:SPLASH"], &["CHANSEY:SPLASH"]);
    // Jolteon starts badly poisoned at reduced HP; switch it out, paralyse it on the bench, and
    // bring it back.
    g.b.sides[0].hp = 100;
    g.b.sides[0].status = PSN;
    g.b.sides[0].bs3 |= BADLY_POISONED;
    g.b.sides[0].toxic = 3;
    g.step(&d, [Action::Switch(1), Action::Move(0)], &mut rng(), false);
    assert_eq!(g.teams[0][0].hp, 100);
    assert_eq!(g.teams[0][0].status, PSN, "Toxic becomes ordinary poison");
    g.teams[0][0].status = PAR;
    g.step(&d, [Action::Switch(0), Action::Move(0)], &mut rng(), false);
    let x = &g.b.sides[0];
    assert_eq!(x.bs3 & BADLY_POISONED, 0);
    assert_eq!(x.stats[SPE], g.teams[0][0].mon.stats[SPE] / 4, "paralysis Speed drop applied on entry");
    assert_eq!(x.mods, [7; 6]);
}

#[test]
fn incoming_pokemon_takes_residual_damage_on_the_switch_turn() {
    let d = Dex::load_default();
    let mut g = battle(&d, &["JOLTEON:SPLASH", "TAUROS:SPLASH"], &["CHANSEY:SPLASH"]);
    g.teams[0][1].status = BRN;
    g.step(&d, [Action::Switch(1), Action::Move(0)], &mut rng(), false);
    let max = g.teams[0][1].mon.max_hp;
    assert_eq!(g.b.sides[0].hp, max - max / 16, "HandlePoisonBurnLeechSeed runs for the switcher");
}

#[test]
fn sleep_clause_blocks_a_second_sleep() {
    let d = Dex::load_default();
    let mut g = battle(&d, &["GENGAR:HYPNOSIS"], &["CHANSEY:SPLASH", "TAUROS:SPLASH"]);
    // Chansey (active) is asleep from an earlier Hypnosis; it switches out to Tauros.
    g.b.sides[1].status = 3;
    g.b.sides[1].clauses |= SLEPT_BY_FOE;
    g.step(&d, [Action::Move(0), Action::Switch(1)], &mut rng(), false);
    assert!(g.teams[1][0].slept_by_foe && g.teams[1][0].status == 3, "sleep and its source persist on the bench");
    assert_eq!(g.b.sides[1].status & SLP_MASK, 0, "Hypnosis on the incoming Tauros must fail");
    for seed in 0..200 {
        let mut h = g.clone();
        h.step(&d, [Action::Move(0), Action::Move(0)], &mut Sampler { rng: Rng::new(seed) }, false);
        assert_eq!(h.b.sides[1].status & SLP_MASK, 0, "Hypnosis must fail while Chansey sleeps");
    }
    // Once Chansey is awake (Rest-induced sleep doesn't count either), Hypnosis can land.
    g.teams[1][0].status = 0;
    g.teams[1][0].slept_by_foe = false;
    let slept = (0..200).any(|seed| {
        let mut h = g.clone();
        h.step(&d, [Action::Move(0), Action::Move(0)], &mut Sampler { rng: Rng::new(seed) }, false);
        h.b.sides[1].status & SLP_MASK != 0
    });
    assert!(slept);
}

#[test]
fn no_switching_while_recharging_but_allowed_while_asleep() {
    let d = Dex::load_default();
    let mut g = battle(&d, &["TAUROS:HYPER_BEAM", "CHANSEY:SPLASH"], &["SNORLAX:SPLASH"]);
    g.b.sides[0].bs2 |= NEEDS_TO_RECHARGE;
    assert_eq!(g.legal_actions(&d, 0), vec![Action::Pass]);
    g.b.sides[0].bs2 &= !NEEDS_TO_RECHARGE;
    g.b.sides[0].status = 2;
    assert_eq!(g.legal_actions(&d, 0), vec![Action::Pass, Action::Switch(1)]);
}

#[test]
fn a_faint_ends_the_turn_and_forces_a_replacement() {
    let d = Dex::load_default();
    let mut g = battle(&d, &["GENGAR:NIGHT_SHADE"], &["CHANSEY:SPLASH", "TAUROS:SPLASH"]);
    g.b.sides[1].hp = 1;
    g.b.sides[1].status = PSN;
    g.step(&d, [Action::Move(0), Action::Move(0)], &mut rng(), false);
    assert_eq!(g.teams[1][0].hp, 0);
    assert_eq!(g.need_replace, [false, true]);
    assert_eq!(g.legal_actions(&d, 0), vec![Action::Pass]);
    assert_eq!(g.legal_actions(&d, 1), vec![Action::Switch(1)]);
    g.step(&d, [Action::Pass, Action::Switch(1)], &mut rng(), false);
    assert_eq!(g.active[1], 1);
    assert_eq!(g.b.sides[1].hp, g.teams[1][1].mon.max_hp, "replacement doesn't take a turn");
    assert!(g.result.is_none());
}

#[test]
fn pp_runs_out_then_struggle() {
    let d = Dex::load_default();
    let mut g = battle(&d, &["CHANSEY:SOFTBOILED"], &["SNORLAX:SPLASH"]);
    assert_eq!(g.teams[0][0].pp[0], 16, "Softboiled: 10 PP, 16 with three PP Ups");
    for _ in 0..16 {
        g.step(&d, [Action::Move(0), Action::Move(0)], &mut rng(), false);
    }
    assert_eq!(g.teams[0][0].pp[0], 0);
    assert_eq!(g.legal_actions(&d, 0), vec![Action::Move(gen1::engine::STRUGGLE_SLOT)]);
}
