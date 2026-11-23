//! Mechanics tests: each computes an exact outcome distribution with the Enumerator and checks it
//! against values derived by hand from the pokered disassembly.

use gen1::chance::Enumerator;
use gen1::cli::matchup;
use gen1::data::Dex;
use gen1::engine::Engine;
use gen1::state::*;

fn dex() -> Dex {
    Dex::load_default()
}

fn rules() -> Rules {
    Rules { sleep_clause: false, ..Rules::default() }
}

/// All (state, outcome, probability) results of one turn.
fn dist(eng: &Engine, b: &Battle, choice: [u8; 2]) -> Vec<(Battle, Outcome, f64)> {
    let mut e = Enumerator::new();
    let mut out = Vec::new();
    loop {
        let (nb, o) = eng.turn(b, choice, &mut e);
        out.push((nb, o, e.prob()));
        if !e.advance() {
            break;
        }
    }
    let total: f64 = out.iter().map(|x| x.2).sum();
    assert!((total - 1.0).abs() < 1e-12, "probabilities sum to {total}");
    out
}

fn prob(d: &[(Battle, Outcome, f64)], f: impl Fn(&Battle, Outcome) -> bool) -> f64 {
    d.iter().filter(|x| f(&x.0, x.1)).map(|x| x.2).sum()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn stats_at_level_50() {
    let d = dex();
    let m = matchup(&d, "TAUROS:BODY_SLAM", "CHANSEY:SPLASH", rules()).unwrap();
    // ((75+15)*2 + 63) * 50/100 + 60 = 181; ((100+15)*2 + 63) * 50/100 + 5 = 151
    assert_eq!(m.mons[0].max_hp, 181);
    assert_eq!(m.mons[0].stats, [151, 146, 161, 121]);
    assert_eq!(m.mons[1].max_hp, 356);
}

#[test]
fn accuracy_is_255_of_256_and_crit_rate_is_base_speed_over_512() {
    let d = dex();
    let m = matchup(&d, "TAUROS:STRENGTH", "CHANSEY:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let b = m.initial(&d);
    let r = dist(&eng, &b, [0, 0]);
    close(prob(&r, |b, _| b.sides[1].hp == 356), 1.0 / 256.0);
    // Strength: base = 22*80*151/56/50 = 94 (+2) = 96, STAB 144 -> non-crit 122..144.
    // Crit: level 100 -> 42*80*151/56/50 = 181 (+2) = 183, STAB 274 -> 233..274.
    let min_crit = 274 * 217 / 255;
    close(prob(&r, |b, _| 356 - b.sides[1].hp >= min_crit), 255.0 / 256.0 * (55.0 / 256.0));
    close(prob(&r, |b, _| b.sides[1].hp == 356 - 144), 255.0 / 256.0 * (201.0 / 256.0) / 39.0);
}

#[test]
fn focus_energy_quarters_the_crit_rate() {
    let d = dex();
    let m = matchup(&d, "TAUROS:STRENGTH", "CHANSEY:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[0].bs2 |= GETTING_PUMPED;
    let r = dist(&eng, &b, [0, 0]);
    // 55 >> 1 = 27, then halved again for a normal move = 13
    close(prob(&r, |b, _| 356 - b.sides[1].hp >= 274 * 217 / 255), 255.0 / 256.0 * (13.0 / 256.0));
}

#[test]
fn wrap_on_a_ghost_still_traps() {
    let d = dex();
    let m = matchup(&d, "ARBOK:WRAP", "GENGAR:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let b = m.initial(&d);
    let r = dist(&eng, &b, [0, 0]);
    // No damage (Normal vs Ghost), but the trap sticks whenever the 85% accuracy roll hits.
    assert!(r.iter().all(|x| x.0.sides[1].hp == m.mons[1].max_hp));
    let acc = (85 * 255 / 100) as f64;
    close(prob(&r, |b, _| b.sides[0].bs1 & USING_TRAPPING_MOVE != 0), acc / 256.0);
    let trapped = r.iter().find(|x| x.0.sides[0].bs1 & USING_TRAPPING_MOVE != 0).unwrap().0.clone();
    assert!(!eng.needs_choice(&trapped, 1));
}

#[test]
fn thunder_wave_goes_through_substitute_but_body_slam_paralysis_does_not() {
    let d = dex();
    let m = matchup(&d, "JOLTEON:THUNDER_WAVE,BODY_SLAM", "CHANSEY:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[1].bs2 |= HAS_SUBSTITUTE_UP;
    b.sides[1].sub_hp = 255;
    let r = dist(&eng, &b, [0, 0]);
    close(prob(&r, |b, _| b.sides[1].status & PAR != 0), 255.0 / 256.0);
    let r = dist(&eng, &b, [1, 0]);
    close(prob(&r, |b, _| b.sides[1].status & PAR != 0), 0.0);
}

#[test]
fn sleep_move_always_hits_a_recharging_target() {
    let d = dex();
    let m = matchup(&d, "GENGAR:HYPNOSIS", "SNORLAX:HYPER_BEAM", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[1].bs2 |= NEEDS_TO_RECHARGE;
    b.sides[1].selected = d.move_id("HYPER_BEAM").unwrap();
    let r = dist(&eng, &b, [0, 0]);
    // Hypnosis is only 60% accurate, but the recharging Snorlax always falls asleep. Its own turn
    // then ticks the counter (uniform 1..=7) down once, so a counter of 1 wakes it immediately.
    close(prob(&r, |b, _| b.sides[1].status & SLP_MASK != 0), 6.0 / 7.0);
    close(prob(&r, |b, _| b.sides[1].status & SLP_MASK == 6), 1.0 / 7.0);
}

#[test]
fn raising_a_stat_reapplies_paralysis_to_the_opponent() {
    let d = dex();
    let m = matchup(&d, "CHANSEY:SPLASH", "TAUROS:SWORDS_DANCE", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[0].status = PAR;
    b.sides[0].stats[SPE] = m.mons[0].stats[SPE] / 4; // 101 -> 25
    let r = dist(&eng, &b, [0, 0]);
    // Every outcome where Tauros got to act quarters Chansey's speed again: 25 -> 6.
    for (nb, _, _) in &r {
        if nb.sides[1].mods[ATK] == 9 {
            assert_eq!(nb.sides[0].stats[SPE], 6);
        }
    }
    close(prob(&r, |b, _| b.sides[1].mods[ATK] == 9), 1.0);
}

#[test]
fn hyper_beam_needs_no_recharge_after_breaking_a_substitute() {
    let d = dex();
    let m = matchup(&d, "TAUROS:HYPER_BEAM", "CHANSEY:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[1].bs2 |= HAS_SUBSTITUTE_UP;
    b.sides[1].sub_hp = 89;
    let r = dist(&eng, &b, [0, 0]);
    for (nb, _, p) in &r {
        if *p > 0.0 && nb.sides[1].bs2 & HAS_SUBSTITUTE_UP == 0 {
            assert_eq!(nb.sides[0].bs2 & NEEDS_TO_RECHARGE, 0, "recharge after breaking the substitute");
        }
    }
    // Without a substitute, a hit that does not KO requires recharging (a critical hit KOs Chansey).
    let b = m.initial(&d);
    let r = dist(&eng, &b, [0, 0]);
    // Hyper Beam is 90% accurate: 90 * 255 / 100 = 229.
    close(prob(&r, |b, _| b.sides[0].bs2 & NEEDS_TO_RECHARGE != 0), 229.0 / 256.0 * 201.0 / 256.0);
    close(prob(&r, |_, o| o == Outcome::Win(0)), 229.0 / 256.0 * 55.0 / 256.0);
}

#[test]
fn recover_fails_when_missing_exactly_255_hp() {
    let d = dex();
    let m = matchup(&d, "CHANSEY:SOFTBOILED", "TAUROS:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    for (missing, heals) in [(255u16, false), (254, true), (256, true), (0, false)] {
        let mut b = m.initial(&d);
        b.sides[0].hp = 356 - missing;
        let r = dist(&eng, &b, [0, 0]);
        assert_eq!(r.iter().all(|x| x.0.sides[0].hp == 356 - missing), !heals, "missing {missing}");
    }
}

#[test]
fn substitute_at_exactly_a_quarter_hp_faints_the_user() {
    let d = dex();
    let m = matchup(&d, "TAUROS:SUBSTITUTE", "CHANSEY:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[0].hp = 181 / 4;
    let r = dist(&eng, &b, [0, 0]);
    close(prob(&r, |_, o| o == Outcome::Win(1)), 1.0);
    b.sides[0].hp = 181 / 4 - 1;
    let r = dist(&eng, &b, [0, 0]);
    close(prob(&r, |b, o| o == Outcome::Ongoing && b.sides[0].bs2 & HAS_SUBSTITUTE_UP == 0), 1.0);
}

#[test]
fn leech_seed_advances_the_toxic_counter_too() {
    let d = dex();
    let m = matchup(&d, "VENUSAUR:SPLASH", "SNORLAX:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let mut b = m.initial(&d);
    b.sides[1].status = PSN;
    b.sides[1].bs3 |= BADLY_POISONED;
    b.sides[1].bs2 |= SEEDED;
    let r = dist(&eng, &b, [0, 0]);
    let max = m.mons[1].max_hp; // 1/16 = n; poison takes n*1, leech seed n*2
    let n = max / 16;
    for (nb, _, _) in &r {
        assert_eq!(nb.sides[1].toxic, 2);
        assert_eq!(nb.sides[1].hp, max - 3 * n);
    }
}

#[test]
fn quick_attack_and_counter_priorities() {
    let d = dex();
    // Slow Snorlax's Quick Attack goes before fast Jolteon's Splash.
    let m = matchup(&d, "SNORLAX:QUICK_ATTACK,COUNTER", "JOLTEON:SPLASH,BODY_SLAM", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let b = m.initial(&d);
    let mut e = Enumerator::new();
    let (_, _, log) = eng.turn_logged(&b, [0, 0], &mut e);
    assert!(log[0].contains("SNORLAX") && log[0].contains("QUICK_ATTACK"), "{log:?}");
    // Counter goes last and returns double the Body Slam damage.
    let r = dist(&eng, &b, [1, 1]);
    for (nb, _, p) in &r {
        let taken = m.mons[0].max_hp - nb.sides[0].hp;
        let dealt = m.mons[1].max_hp - nb.sides[1].hp;
        if *p > 0.0 && taken > 0 && dealt > 0 {
            assert_eq!(dealt, 2 * taken);
        }
    }
}

#[test]
fn ohko_moves_fail_against_faster_targets() {
    let d = dex();
    let m = matchup(&d, "RHYDON:HORN_DRILL", "JOLTEON:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let r = dist(&eng, &m.initial(&d), [0, 0]);
    close(prob(&r, |_, o| o == Outcome::Win(0)), 0.0);
    let m2 = matchup(&d, "AERODACTYL:HORN_DRILL", "RHYDON:SPLASH", rules()).unwrap();
    let eng2 = Engine::new(&d, &m2);
    let r = dist(&eng2, &m2.initial(&d), [0, 0]);
    // 30% accuracy -> 76/256 (Rhydon is Ground/Rock, Horn Drill is Normal: resisted but still a KO)
    close(prob(&r, |_, o| o == Outcome::Win(0)), 76.0 / 256.0);
}

#[test]
fn sleep_clause_flag_makes_sleep_moves_fail() {
    let d = dex();
    let m = matchup(&d, "GENGAR:HYPNOSIS", "CHANSEY:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let free = m.initial(&d);
    let slept = |b: &Battle| prob(&dist(&eng, b, [0, 0]), |nb, _| nb.sides[1].status & SLP_MASK != 0);
    assert!(slept(&free) > 0.5);
    let blocked = m.entry(&d, [m.mons[0].max_hp, m.mons[1].max_hp], [0, 0], [SLEEP_BLOCKED, 0]);
    close(slept(&blocked), 0.0);
}

#[test]
fn freeze_clause_flag_stops_freezing() {
    let d = dex();
    let m = matchup(&d, "LAPRAS:BLIZZARD", "SNORLAX:SPLASH", rules()).unwrap();
    let eng = Engine::new(&d, &m);
    let frozen = |b: &Battle| prob(&dist(&eng, b, [0, 0]), |nb, _| nb.sides[1].status & FRZ != 0);
    assert!(frozen(&m.initial(&d)) > 0.05);
    let blocked = m.entry(&d, [m.mons[0].max_hp, m.mons[1].max_hp], [0, 0], [FREEZE_BLOCKED, 0]);
    close(frozen(&blocked), 0.0);
}

#[test]
fn entry_position_applies_paralysis_and_burn_penalties() {
    let d = dex();
    let m = matchup(&d, "JOLTEON:THUNDERBOLT", "TAUROS:BODY_SLAM", rules()).unwrap();
    let b = m.entry(&d, [100, 120], [PAR, BRN], [0, 0]);
    assert_eq!(b.sides[0].hp, 100);
    assert_eq!(b.sides[0].status, PAR);
    // LoadBattleMonFromParty applies the status penalties: Speed quartered, Attack halved.
    assert_eq!(b.sides[0].stats[SPE], m.mons[0].stats[SPE] / 4);
    assert_eq!(b.sides[0].stats[ATK], m.mons[0].stats[ATK]);
    assert_eq!(b.sides[1].stats[ATK], m.mons[1].stats[ATK] / 2);
    assert_eq!(b.sides[1].stats[SPE], m.mons[1].stats[SPE]);
    let s = m.entry(&d, [100, 120], [0, 3], [0, 0]);
    assert_eq!(s.sides[1].status & SLP_MASK, 3);
    assert_eq!(s.sides[1].bs3 & BADLY_POISONED, 0);
}
