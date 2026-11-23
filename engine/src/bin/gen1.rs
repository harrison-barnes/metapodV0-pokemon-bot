//! gen1 solve  "TAUROS:BODY_SLAM,HYPER_BEAM,EARTHQUAKE,BLIZZARD" "CHANSEY:..."
//! gen1 sim    A B [--seed N] [--battles N]   (random legal moves; prints one log, then win rate)
//! gen1 replay A B [--seed N] [--buckets K] [--hp-levels Q] [--json FILE]
//!             one battle where both sides play their optimal (Nash) mixed strategies, with the
//!             game value before every turn

use gen1::chance::{Rng, Sampler};
use gen1::cli::{describe, matchup};
use gen1::data::Dex;
use gen1::engine::Engine;
use gen1::solver::Solver;
use gen1::state::{Outcome, Rules};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: gen1 solve|sim SPECIES:MOVES SPECIES:MOVES [--seed N] [--battles N] [--no-sleep-clause] [--rules video|smogon|smogon-pp]");
        std::process::exit(2);
    }
    let dex = Dex::load_default();
    let mut rules = match args.iter().position(|a| a == "--rules").and_then(|i| args.get(i + 1)) {
        Some(name) => Rules::named(name).unwrap_or_else(|| panic!("unknown ruleset {name} (video|smogon|smogon-pp)")),
        None => Rules::default(),
    };
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<u64>().ok());
    if args.iter().any(|a| a == "--no-sleep-clause") {
        rules.sleep_clause = false;
    }
    let m = matchup(&dex, &args[2], &args[3], rules).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    println!("side 0: {}", describe(&dex, &m.mons[0]));
    println!("side 1: {}", describe(&dex, &m.mons[1]));
    let mut eng = Engine::new(&dex, &m);
    eng.roll_buckets = flag("--buckets").unwrap_or(0) as u8;
    match args[1].as_str() {
        "solve" => {
            let t = Instant::now();
            let mut s = Solver::new(eng);
            s.uniform = args.iter().any(|a| a == "--uniform");
            s.split_turns = !args.iter().any(|a| a == "--no-split");
            s.set_hp_levels(flag("--hp-levels").unwrap_or(0) as u16);
            let v = s.solve(&m.initial(&dex));
            let st = &s.stats;
            println!(
                "value for side 0: {v:.6}  ({} states, {} SCCs, largest SCC {}, {} sweeps, residual {:.1e}, {} paths, {} transitions, {:.2?})",
                st.states, st.sccs, st.largest_scc, st.sweeps, st.max_residual, st.paths, st.transitions, t.elapsed()
            );
            let (x, sol) = s.policy(&m.initial(&dex));
            let name = |side: usize, c: u8| dex.mv(m.mons[side].moves[c as usize]).name.clone();
            println!("opening strategy side 0:");
            for (i, &r) in x.rows.iter().enumerate() {
                println!("  {:>14} {:6.2}%", name(0, r), sol.row[i] * 100.0);
            }
            println!("opening strategy side 1:");
            for (j, &c) in x.cols.iter().enumerate() {
                println!("  {:>14} {:6.2}%", name(1, c), sol.col[j] * 100.0);
            }
        }
        "sim" => {
            let seed = flag("--seed").unwrap_or(1);
            let n = flag("--battles").unwrap_or(1);
            let mut ch = Sampler { rng: Rng::new(seed) };
            let mut wins = [0u64; 3];
            for k in 0..n {
                let mut b = m.initial(&dex);
                let mut turn = 0;
                loop {
                    turn += 1;
                    let mut choice = [0u8; 2];
                    for s in 0..2 {
                        let c = eng.legal_choices(&b, s);
                        if !c.is_empty() {
                            choice[s] = c[ch.rng.below(c.len() as u64) as usize];
                        }
                    }
                    let (nb, o, log) = eng.turn_logged(&b, choice, &mut ch);
                    if k == 0 {
                        println!("turn {turn}:");
                        for l in log {
                            println!("{l}");
                        }
                        println!("  HP {}/{}  {}/{}", nb.sides[0].hp, m.mons[0].max_hp, nb.sides[1].hp, m.mons[1].max_hp);
                    }
                    b = nb;
                    match o {
                        Outcome::Ongoing if turn < 1000 => continue,
                        Outcome::Win(s) => wins[s] += 1,
                        _ => wins[2] += 1,
                    }
                    break;
                }
            }
            println!("random play over {n} battles: side0 {} side1 {} draw {}", wins[0], wins[1], wins[2]);
        }
        "replay" => {
            let mut s = Solver::new(eng);
            s.set_hp_levels(flag("--hp-levels").unwrap_or(0) as u16);
            let v0 = s.solve(&m.initial(&dex));
            let mut rng = Rng::new(flag("--seed").unwrap_or(1));
            let mut ch = Sampler { rng: Rng::new(flag("--seed").unwrap_or(1) ^ 0xABCDEF) };
            let mut b = m.initial(&dex);
            s.eng.canonicalize(&mut b);
            let mut turns = Vec::new();
            println!("value at start: {v0:.4}");
            for turn in 1..=500 {
                let value = s.value_of(&b).unwrap_or(f64::NAN);
                let (x, sol) = s.policy(&b);
                let pick = |p: &[f64], rng: &mut Rng| {
                    let mut r = rng.f64();
                    for (i, &q) in p.iter().enumerate() {
                        if r < q {
                            return i;
                        }
                        r -= q;
                    }
                    p.len() - 1
                };
                let (i, j) = (pick(&sol.row, &mut rng), pick(&sol.col, &mut rng));
                let choice = [x.rows[i], x.cols[j]];
                let mv = |side: usize, c: u8, need: bool| {
                    if !need { "(locked)".to_string() } else { dex.mv(m.mons[side].moves[c as usize]).name.clone() }
                };
                let need = [s.eng.needs_choice(&b, 0), s.eng.needs_choice(&b, 1)];
                let strat = |side: usize| -> Vec<(String, f64)> {
                    let (opts, p) = if side == 0 { (&x.rows, &sol.row) } else { (&x.cols, &sol.col) };
                    opts.iter().zip(p).map(|(&c, &q)| (mv(side, c, need[side]), q)).collect()
                };
                let (nb, o, log) = s.eng.turn_logged(&b, choice, &mut ch);
                println!("turn {turn}: value {value:.4}  {} vs {}", mv(0, choice[0], need[0]), mv(1, choice[1], need[1]));
                for l in &log {
                    println!("{l}");
                }
                println!("  HP {}/{}  {}/{}", nb.sides[0].hp, m.mons[0].max_hp, nb.sides[1].hp, m.mons[1].max_hp);
                turns.push(serde_json::json!({
                    "turn": turn, "value": value, "strategy": [strat(0), strat(1)],
                    "moves": [mv(0, choice[0], need[0]), mv(1, choice[1], need[1])],
                    "log": log, "hp": [nb.sides[0].hp, nb.sides[1].hp],
                }));
                if o != Outcome::Ongoing {
                    println!("result: {o:?}");
                    turns.push(serde_json::json!({"result": format!("{o:?}")}));
                    break;
                }
                let mut nb = nb;
                s.eng.canonicalize(&mut nb);
                b = s.sample_snap(nb, &mut rng);
            }
            if let Some(i) = args.iter().position(|a| a == "--json") {
                let out = serde_json::json!({
                    "sides": [describe(&dex, &m.mons[0]), describe(&dex, &m.mons[1])],
                    "max_hp": [m.mons[0].max_hp, m.mons[1].max_hp], "value": v0, "turns": turns,
                });
                std::fs::write(&args[i + 1], serde_json::to_string_pretty(&out).unwrap()).unwrap();
            }
        }
        other => eprintln!("unknown command {other}"),
    }
}
