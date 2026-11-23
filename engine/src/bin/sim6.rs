//! sim6 TEAM_A TEAM_B [--battles N] [--seed S] [--rules video|smogon] [--switch-prob P] [--log]
//!
//! Random-policy 6v6 battles (for checking the engine against Showdown). A team is
//! "SPECIES:MOVE,MOVE,...;SPECIES:...". Each turn a side switches to a random healthy team member
//! with probability P (when it may switch), and otherwise uses a random usable move, or stays in
//! when it has none to pick. Fainted Pokemon are replaced at random.
use gen1::battle6::{Action, Battle6};
use gen1::chance::{Rng, Sampler};
use gen1::cli::parse_mon;
use gen1::data::Dex;
use gen1::state::{Mon, Outcome, Rules};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: sim6 TEAM_A TEAM_B [--battles N] [--seed S] [--rules video|smogon] [--switch-prob P] [--log]");
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    let dex = Dex::load_default();
    let rules = flag("--rules").map(|r| Rules::named(r).expect("unknown ruleset")).unwrap_or_else(Rules::smogon);
    let team = |spec: &str| -> Vec<Mon> {
        spec.split(';').map(|m| parse_mon(&dex, m, &rules).unwrap_or_else(|e| panic!("{e}"))).collect()
    };
    let teams = [team(&args[1]), team(&args[2])];
    let n: u64 = flag("--battles").map(|x| x.parse().unwrap()).unwrap_or(1);
    let p_switch: f64 = flag("--switch-prob").map(|x| x.parse().unwrap()).unwrap_or(0.2);
    let log = args.iter().any(|a| a == "--log");
    let mut ch = Sampler { rng: Rng::new(flag("--seed").map(|x| x.parse().unwrap()).unwrap_or(1)) };
    let (mut wins, mut turns) = ([0u64; 3], 0u64);
    for _ in 0..n {
        let mut b = Battle6::new(&dex, &rules, teams.clone());
        while b.result.is_none() && b.turn < 1000 {
            let actions = [0, 1].map(|s| pick(&b.legal_actions(&dex, s), b.need_replace.iter().any(|&r| r), p_switch, &mut ch.rng));
            for line in b.step(&dex, actions, &mut ch, log) {
                println!("{line}");
            }
        }
        turns += b.turn as u64;
        match b.result {
            Some(Outcome::Win(s)) => wins[s] += 1,
            _ => wins[2] += 1,
        }
    }
    println!("{n} battles: side0 {} side1 {} draw {} mean turns {:.1}", wins[0], wins[1], wins[2], turns as f64 / n as f64);
}

fn pick(legal: &[Action], replacing: bool, p_switch: f64, rng: &mut Rng) -> Action {
    let switches: Vec<Action> = legal.iter().copied().filter(|a| matches!(a, Action::Switch(_))).collect();
    let rest: Vec<Action> = legal.iter().copied().filter(|a| !matches!(a, Action::Switch(_))).collect();
    let any = |v: &[Action], rng: &mut Rng| v[rng.below(v.len() as u64) as usize];
    if replacing || rest.is_empty() || (!switches.is_empty() && rng.f64() < p_switch) {
        return if switches.is_empty() { any(&rest, rng) } else { any(&switches, rng) };
    }
    any(&rest, rng)
}
