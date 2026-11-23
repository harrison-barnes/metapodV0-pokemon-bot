//! The equilibrium statistics must reproduce the game value: under the solved strategies,
//! value = P(win) + draw_value * P(double KO) + 0.5 * P(endless). Strategies come from a slightly
//! discounted solve (see `Solver::discount`) and are checked against the undiscounted values.
use gen1::cli::matchup;
use gen1::data::Dex;
use gen1::engine::Engine;
use gen1::solver::Solver;
use gen1::state::Rules;

fn check(a: &str, b: &str, buckets: u8, hp_levels: u16, draw_value: f64) {
    let d = Dex::load_default();
    let m = matchup(&d, a, b, Rules::smogon()).unwrap();
    let solver = |discount: f64, stats: bool| {
        let mut eng = Engine::new(&d, &m);
        eng.roll_buckets = buckets;
        let mut s = Solver::new(eng);
        s.set_hp_levels(hp_levels);
        s.track_stats = stats;
        s.draw_value = draw_value;
        s.discount = discount;
        s.solve(&m.initial(&d));
        s
    };
    let exact = solver(1.0, false);
    let s = solver(1.0 - 1e-7, true);
    let v = exact.value[0];
    let mut worst = 0.0f64;
    for (id, eq) in s.eq.iter().enumerate() {
        let implied = eq.p_win + draw_value * eq.p_double_ko + 0.5 * eq.p_endless;
        let truth = exact.value_of(&s.states[id]).expect("same state space");
        worst = worst.max((implied - truth).abs());
        assert!(eq.hp.iter().all(|h| (0.0..=1.0 + 1e-9).contains(h)), "HP fraction out of range: {eq:?}");
        assert!(eq.p_endless > 1e-9 || eq.turns >= 1.0 - 1e-9, "a finished battle lasts at least a turn: {eq:?}");
    }
    assert!(worst < 1e-4, "{a} vs {b}: equilibrium play misses the value by {worst}");
    let root = s.eq[0];
    eprintln!("{a} vs {b}: value {v:.6} win {:.4} dko {:.4} endless {:.4} turns {:.2}", root.p_win, root.p_double_ko, root.p_endless, root.turns);
}

#[test]
fn stats_reproduce_values() {
    check("TAUROS:BODY_SLAM,HYPER_BEAM,EARTHQUAKE,BLIZZARD", "STARMIE:SURF,PSYCHIC,THUNDERBOLT,RECOVER", 3, 24, 0.5);
}

#[test]
fn double_ko_valued_separately() {
    // Explosion makes double KOs common; the identity must hold for any double-KO value.
    for dv in [0.3, 0.5, 0.7] {
        check("SNORLAX:BODY_SLAM,SELFDESTRUCT,EARTHQUAKE,REST", "GENGAR:HYPNOSIS,THUNDERBOLT,EXPLOSION,NIGHT_SHADE", 1, 12, dv);
    }
}

#[test]
fn cyclic_stall_game() {
    check("CHANSEY:SOFTBOILED,THUNDER_WAVE,ICE_BEAM,SING", "STARMIE:RECOVER,THUNDER_WAVE,PSYCHIC,BLIZZARD", 1, 6, 0.5);
}
