//! `Rules::heal_pp`: the healing move's PP, tracked as coarse levels (state.rs `HEAL_PP_MASK`).

use gen1::chance::{Enumerator, Rng, Sampler};
use gen1::cli::matchup;
use gen1::data::Dex;
use gen1::engine::Engine;
use gen1::state::*;

fn level(b: &Battle, s: usize) -> u8 {
    b.sides[s].clauses >> HEAL_PP_SHIFT
}

#[test]
fn levels_stand_for_powers_of_two_up_to_max_pp() {
    let counts = |max| (0..=heal_top_level(max)).map(|l| heal_level_pp(l, max)).collect::<Vec<_>>();
    assert_eq!(counts(16), vec![0, 1, 2, 4, 8, 16]);   // Softboiled, Rest: 10 PP + PP Ups
    assert_eq!(counts(32), vec![0, 1, 2, 4, 8, 16, 32]); // Recover: 20 PP + PP Ups
    assert_eq!(counts(24), vec![0, 1, 2, 4, 8, 16, 24]);
}

#[test]
fn off_by_default_and_only_for_the_healer() {
    let d = Dex::load_default();
    let m = matchup(&d, "CHANSEY:SOFTBOILED,SEISMIC_TOSS", "TAUROS:STRENGTH", Rules::smogon()).unwrap();
    assert_eq!(level(&m.initial(&d), 0), 0);
    let m = matchup(&d, "CHANSEY:SOFTBOILED,SEISMIC_TOSS", "TAUROS:STRENGTH", Rules::smogon_pp()).unwrap();
    let b = m.initial(&d);
    assert_eq!((level(&b, 0), level(&b, 1)), (5, 0));
    let e = m.entry(&d, [300, 300], [0, 0], [SLEEP_BLOCKED, 0]);
    assert_eq!((level(&e, 0), e.sides[0].clauses & SLEEP_BLOCKED), (5, SLEEP_BLOCKED));
}

#[test]
fn a_use_drops_a_level_with_probability_one_over_the_gap() {
    let d = Dex::load_default();
    let m = matchup(&d, "CHANSEY:SOFTBOILED,SEISMIC_TOSS", "TAUROS:STRENGTH", Rules::smogon_pp()).unwrap();
    let eng = Engine::new(&d, &m);
    for (from, p_drop) in [(5u8, 1.0 / 8.0), (4, 1.0 / 4.0), (3, 1.0 / 2.0), (2, 1.0), (1, 1.0)] {
        let mut b = m.entry(&d, [m.mons[0].max_hp, 300], [0, 0], [0, 0]);
        b.sides[0].clauses = (b.sides[0].clauses & !HEAL_PP_MASK) | (from << HEAL_PP_SHIFT);
        let mut e = Enumerator::new();
        let (mut drop, mut total) = (0.0, 0.0);
        loop {
            let (nb, _) = eng.turn(&b, [0, 0], &mut e);
            total += e.prob();
            if level(&nb, 0) == from - 1 {
                drop += e.prob();
            } else {
                assert_eq!(level(&nb, 0), from);
            }
            if !e.advance() {
                break;
            }
        }
        // At full HP Chansey survives Strength even on a critical hit, and Strength has no side
        // effect that could stop it moving, so Softboiled always reaches DecrementPP.
        assert!((drop / total - p_drop).abs() < 1e-9, "level {from}: {drop} vs {p_drop}");
    }
}

#[test]
fn an_empty_healing_move_is_not_selectable() {
    let d = Dex::load_default();
    let m = matchup(&d, "CHANSEY:SOFTBOILED,SEISMIC_TOSS", "TAUROS:STRENGTH", Rules::smogon_pp()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    assert_eq!(eng.legal_choices(&b, 0), vec![0, 1]);
    b.sides[0].clauses &= !HEAL_PP_MASK;
    assert_eq!(eng.legal_choices(&b, 0), vec![1]);
}

#[test]
fn sixteen_uses_on_average() {
    // Using Softboiled every turn from full PP, the number of uses before it runs out averages 16.
    let d = Dex::load_default();
    let m = matchup(&d, "CHANSEY:SOFTBOILED,SEISMIC_TOSS", "CHANSEY:SOFTBOILED,SEISMIC_TOSS", Rules::smogon_pp()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut rng = Sampler { rng: Rng::new(11) };
    let runs = 20_000;
    let mut uses = 0u64;
    for _ in 0..runs {
        let mut b = m.initial(&d);
        while level(&b, 0) > 0 {
            b = eng.turn(&b, [0, 0], &mut rng).0;
            uses += 1;
        }
    }
    let mean = uses as f64 / runs as f64;
    assert!((mean - 16.0).abs() < 0.15, "mean uses {mean}");
}
