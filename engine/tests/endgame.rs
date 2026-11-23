//! 2v2 endgame solver: reductions to the 1v1 solver, symmetry, dominance, and agreement of its
//! transitions with `Battle6::step`.

use gen1::battle6::{max_pp, Action, Battle6};
use gen1::chance::{Rng, Sampler};
use gen1::cli::parse_mon;
use gen1::data::Dex;
use gen1::endgame::{EndgameSolver, Next, Teams, PASS, SWITCH};
use gen1::engine::{Engine, STRUGGLE_SLOT};
use gen1::solver::Solver;
use gen1::state::*;

const TAUROS: &str = "TAUROS:BODY_SLAM,HYPER_BEAM,EARTHQUAKE,BLIZZARD";
const STARMIE: &str = "STARMIE:RECOVER,THUNDER_WAVE,THUNDERBOLT,BLIZZARD";

fn mon(d: &Dex, spec: &str) -> Mon {
    parse_mon(d, spec, &Rules::smogon()).unwrap()
}

fn teams(d: &Dex, a: [&str; 2], b: [&str; 2]) -> Teams {
    Teams::new([a.map(|s| mon(d, s)), b.map(|s| mon(d, s))], Rules::smogon())
}

/// 1v1 value of a position at the given precision, and the number of states.
fn solve_1v1_states(d: &Dex, m: &Matchup, b: &Battle, buckets: u8, levels: u16) -> (f64, usize) {
    let mut eng = Engine::new(d, m);
    eng.roll_buckets = buckets;
    let mut s = Solver::new(eng);
    s.set_hp_levels(levels);
    (s.solve(b), s.states.len())
}

fn solve_1v1(d: &Dex, m: &Matchup, b: &Battle, buckets: u8, levels: u16) -> f64 {
    solve_1v1_states(d, m, b, buckets, levels).0
}

#[test]
fn fainted_benches_reduce_to_the_1v1_solver() {
    let d = Dex::load_default();
    let t = teams(&d, [TAUROS, STARMIE], [STARMIE, TAUROS]);
    let m = t.matchup([0, 0]).clone();
    let (v1, n1) = solve_1v1_states(&d, &m, &m.initial(&d), 3, 24);
    let mut s = EndgameSolver::new(&d, &t);
    s.set_roll_buckets(3);
    s.set_hp_levels(24);
    let root = s.position([0, 0], [[353, 0], [323, 0]], [[0; 2]; 2]);
    let v2 = s.solve(&root);
    assert!((v1 - v2).abs() < 1e-9, "1v1 {v1} vs 2v2 {v2}");
    assert_eq!(s.states.len(), n1, "the same states, one for one");

    // The same pair reached by a forced replacement at the root (each side's lead has fainted).
    let mut s = EndgameSolver::new(&d, &t);
    s.set_roll_buckets(3);
    s.set_hp_levels(24);
    let root = s.position([0, 0], [[0, 323], [0, 353]], [[0; 2]; 2]);
    assert_eq!(root.active, [1, 1]);
    let v2 = s.solve(&root);
    let v1 = 1.0 - v1; // Starmie vs Tauros is the same game seen from the other side
    assert!((v1 - v2).abs() < 1e-9, "1v1 {v1} vs 2v2 {v2}");
}

#[test]
fn forced_replacement_mid_battle_reduces_to_the_1v1_solver() {
    // Side 0's lead faints in a 6v6 battle; the replacement faces a survivor that keeps its
    // volatile state, and the opponent's bench is empty: from there it is a 1v1 battle.
    let d = Dex::load_default();
    let lead = "TAUROS:BODY_SLAM,HYPER_BEAM";
    let t = teams(&d, [lead, TAUROS], [STARMIE, TAUROS]);
    let mut checked = 0;
    for seed in 0..40 {
        let mut g = Battle6::new(&d, &Rules::smogon(), [vec![mon(&d, lead), mon(&d, TAUROS)], vec![mon(&d, STARMIE), mon(&d, TAUROS)]]);
        g.teams[1][1].hp = 0;
        g.b.sides[0].hp = 60; // a weakened lead, so it faints within a few turns
        let mut rng = Rng::new(seed);
        let mut ch = Sampler { rng: Rng::new(seed + 1000) };
        while g.result.is_none() && !g.need_replace[0] {
            let a = [0, 1].map(|s| {
                let v: Vec<Action> = g.legal_actions(&d, s).into_iter().filter(|a| !matches!(a, Action::Switch(_))).collect();
                v[rng.below(v.len() as u64) as usize]
            });
            g.step(&d, a, &mut ch, false);
        }
        if g.result.is_some() {
            continue;
        }
        let s0 = EndgameSolver::new(&d, &t);
        let root = s0.from_battle6(&g);
        assert_eq!(root.active, [1, 0]);
        let mut s = EndgameSolver::new(&d, &t);
        s.set_roll_buckets(1);
        s.set_hp_levels(12);
        let v2 = s.solve(&root);
        let v1 = solve_1v1(&d, t.matchup([1, 0]), &root.b, 1, 12);
        assert!((v1 - v2).abs() < 1e-8, "seed {seed}: 1v1 {v1} vs 2v2 {v2}");
        checked += 1;
        if checked == 4 {
            break;
        }
    }
    assert!(checked >= 2, "too few replacements reached");
}

#[test]
fn mirror_is_even() {
    let d = Dex::load_default();
    let a = "TAUROS:BODY_SLAM,EARTHQUAKE,BLIZZARD";
    let b = "STARMIE:RECOVER,THUNDERBOLT,THUNDER_WAVE";
    let t = teams(&d, [a, b], [a, b]);
    let mut s = EndgameSolver::new(&d, &t);
    s.set_roll_buckets(1);
    s.set_hp_levels(4);
    let v = s.solve(&s.initial());
    assert!((v - 0.5).abs() < 1e-9, "mirror value {v}");
}

#[test]
fn a_useless_bench_never_hurts() {
    let d = Dex::load_default();
    let karp = "MAGIKARP:SPLASH";
    let m = Matchup { mons: [mon(&d, TAUROS), mon(&d, STARMIE)], rules: Rules::smogon() };
    let v1 = solve_1v1(&d, &m, &m.initial(&d), 1, 12);
    for (side, t) in [(0, teams(&d, [TAUROS, karp], [STARMIE, karp])), (1, teams(&d, [TAUROS, karp], [STARMIE, karp]))] {
        let mut s = EndgameSolver::new(&d, &t);
        s.set_roll_buckets(1);
        s.set_hp_levels(12);
        // Only `side` keeps its Magikarp.
        let karp_hp = t.mons[0][1].max_hp;
        let hp = if side == 0 { [[353, karp_hp], [323, 0]] } else { [[353, 0], [323, karp_hp]] };
        let v2 = s.solve(&s.position([0, 0], hp, [[0; 2]; 2]));
        if side == 0 {
            assert!(v2 >= v1 - 1e-9, "Magikarp on side 0: {v2} < 1v1 {v1}");
        } else {
            assert!(v2 <= v1 + 1e-9, "Magikarp on side 1: {v2} > 1v1 {v1}");
        }
        println!("side {side} with Magikarp: {v2:.6} (1v1 {v1:.6})");
    }
}

#[test]
fn turn_cache_and_matching_bench_grid_are_lossless() {
    let d = Dex::load_default();
    let t = teams(&d, ["TAUROS:BODY_SLAM,EARTHQUAKE,BLIZZARD", "RHYDON:EARTHQUAKE,ROCK_SLIDE"], ["STARMIE:RECOVER,THUNDERBOLT,THUNDER_WAVE", "SNORLAX:BODY_SLAM,SELFDESTRUCT"]);
    let solve = |cache: usize, bench: u16| {
        let mut s = EndgameSolver::new(&d, &t);
        s.set_roll_buckets(1);
        s.set_hp_levels(4);
        s.set_bench_levels(bench);
        s.cache_size = cache;
        let v = s.solve(&s.initial());
        (v, s.states.len())
    };
    let (v, n) = solve(4_000_000, 0);
    let (v_nocache, n_nocache) = solve(0, 0);
    let (v_bench, n_bench) = solve(4_000_000, 4);
    assert_eq!(n, n_nocache);
    assert!((v - v_nocache).abs() < 1e-9, "cache {v} vs none {v_nocache}");
    assert_eq!(n, n_bench, "a bench grid equal to the HP grid changes nothing");
    assert!((v - v_bench).abs() < 1e-9);
    let (v2, n2) = solve(4_000_000, 2);
    assert!(n2 < n, "a coarser bench grid has fewer states");
    println!("4 HP levels: {v:.6} ({n} states); bench on 2 levels: {v2:.6} ({n2} states)");
}

fn code(a: Action) -> u8 {
    match a {
        Action::Move(slot) => slot,
        Action::Pass => PASS,
        Action::Switch(_) => SWITCH,
    }
}

/// Plays random 6v6 battles with two Pokemon a side, and checks every turn against the 2v2
/// solver: the same legal actions; from the same raw position and the same random numbers, the
/// same next state (after the forced replacement) or result; and that next state among the exact
/// successors enumerated from the canonical position.
fn agrees_with_battle6(a: [&str; 2], b: [&str; 2], battles: u64) {
    let d = Dex::load_default();
    let t = teams(&d, a, b);
    let mut s = EndgameSolver::new(&d, &t);
    let (mut turns, mut switches, mut faints) = (0, 0, 0);
    for seed in 0..battles {
        let mut g = Battle6::new(&d, &Rules::smogon(), [a.map(|x| mon(&d, x)).to_vec(), b.map(|x| mon(&d, x)).to_vec()]);
        let mut rng = Rng::new(seed);
        for _ in 0..150 {
            // The 2v2 solver ignores PP.
            for team in g.teams.iter_mut() {
                for m in team.iter_mut() {
                    m.pp = m.mon.moves.map(|mv| max_pp(&d, mv));
                }
            }
            let raw = s.from_battle6_raw(&g);
            let st = s.canonical(raw.clone());
            let mut acts = [Action::Pass; 2];
            for side in 0..2 {
                let legal = g.legal_actions(&d, side);
                let mut six: Vec<u8> = legal.iter().map(|&x| code(x)).collect();
                let mut mine = s.actions(&st, side);
                six.sort();
                mine.sort();
                assert_eq!(six, mine, "seed {seed}: legal actions of side {side}");
                acts[side] = legal[rng.below(legal.len() as u64) as usize];
            }
            let codes = acts.map(code);
            switches += codes.iter().filter(|&&c| c == SWITCH).count();
            let seed2 = rng.next_u64();
            let next = s.step(&raw, codes, &mut Sampler { rng: Rng::new(seed2) });
            g.step(&d, acts, &mut Sampler { rng: Rng::new(seed2) }, false);
            if g.result.is_none() && g.need_replace.iter().any(|&r| r) {
                faints += 1;
                let r = [0, 1].map(|side| if g.need_replace[side] { Action::Switch(1 - g.active[side] as u8) } else { Action::Pass });
                g.step(&d, r, &mut Sampler { rng: Rng::new(0) }, false);
            }
            let theirs = match g.result {
                Some(o) => Next::End(o),
                None => Next::State(s.from_battle6(&g)),
            };
            assert_eq!(next, theirs, "seed {seed}: {codes:?} from {raw:?}");
            let (succ, ends) = s.transitions(&st, codes);
            let total: f64 = succ.iter().map(|x| x.1).sum::<f64>() + ends.iter().sum::<f64>();
            assert!((total - 1.0).abs() < 1e-9, "probabilities sum to {total}");
            match &theirs {
                Next::State(ns) => assert!(succ.iter().any(|(x, _)| x == ns), "seed {seed}: {ns:?} not among the successors of {st:?}"),
                Next::End(Outcome::Win(w)) => assert!(ends[*w] > 0.0),
                Next::End(_) => assert!(ends[2] > 0.0),
            }
            turns += 1;
            if g.result.is_some() {
                break;
            }
        }
    }
    println!("{turns} turns, {switches} switches, {faints} replacements");
    assert!(switches > 0 && faints > 0);
}

#[test]
fn transitions_match_battle6_sleep_wrap_counter_explosion() {
    agrees_with_battle6(
        ["EXEGGUTOR:SLEEP_POWDER,PSYCHIC_M,EXPLOSION,STUN_SPORE", "CHANSEY:SING,ICE_BEAM,SOFTBOILED,COUNTER"],
        ["SNORLAX:BODY_SLAM,REST,HYPER_BEAM,SELFDESTRUCT", "DRAGONITE:WRAP,AGILITY,HYPER_BEAM,THUNDER_WAVE"],
        60,
    );
}

#[test]
fn transitions_match_battle6_priority_seeding_freeze() {
    agrees_with_battle6(
        ["RATICATE:QUICK_ATTACK,SUPER_FANG,BODY_SLAM,BLIZZARD", "VENUSAUR:LEECH_SEED,TOXIC,RAZOR_LEAF,SLEEP_POWDER"],
        ["JYNX:LOVELY_KISS,BLIZZARD,PSYCHIC_M,REST", "STARMIE:RECOVER,THUNDER_WAVE,SURF,BLIZZARD"],
        60,
    );
}

#[test]
fn transitions_match_battle6_bide_dig_substitute_reflect() {
    agrees_with_battle6(
        ["CHANSEY:BIDE,SEISMIC_TOSS,REFLECT,THUNDER_WAVE", "DUGTRIO:DIG,SLASH,EARTHQUAKE,ROCK_SLIDE"],
        ["RHYDON:EARTHQUAKE,SUBSTITUTE,BODY_SLAM,COUNTER", "PERSIAN:QUICK_ATTACK,SLASH,BUBBLEBEAM,THUNDERBOLT"],
        60,
    );
}

#[test]
fn struggle_slot_is_not_a_switch_code() {
    // Action codes must not collide.
    assert!(STRUGGLE_SLOT < PASS && PASS < SWITCH);
}
