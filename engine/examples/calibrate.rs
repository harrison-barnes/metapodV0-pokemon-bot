//! Error of each approximation setting against exact solves, over random pairs from a set file.
//! Usage: calibrate SETS.json PAIRS [SEED] [RULES] > calibration.jsonl
//! Pairs whose exact game exceeds 600k states are skipped, so the sample leans towards smaller
//! games than the ones that actually need the coarse settings.
use gen1::data::Dex;
use gen1::meta::{evaluate, Precision, Set};
use gen1::state::Rules;
use rayon::prelude::*;

const SETTINGS: &[(u8, u16)] = &[(0, 96), (3, 48), (3, 24), (1, 12), (1, 6), (1, 3)];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dex = Dex::load_default();
    let rules = a.get(4).map(|r| Rules::named(r).expect("unknown ruleset")).unwrap_or_default();
    let sets: Vec<Set> = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let want: usize = a[2].parse().unwrap();
    let mut seed: u64 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let n = sets.len();
    let pairs: Vec<(usize, usize)> = (0..want * 3)
        .map(|_| (next() % n, next() % n))
        .filter(|(x, y)| x != y)
        .collect();
    let found = std::sync::atomic::AtomicUsize::new(0);
    pairs.par_iter().for_each(|&(x, y)| {
        if found.load(std::sync::atomic::Ordering::Relaxed) >= want {
            return;
        }
        let Some(exact) = evaluate(&dex, &rules, &sets[x], &sets[y], &[Precision::new(0, 0, 600_000)]) else { return };
        let approx: Vec<(String, Option<f64>)> = SETTINGS
            .iter()
            .map(|&(b, q)| {
                let e = evaluate(&dex, &rules, &sets[x], &sets[y], &[Precision::new(b, q, 600_000)]);
                (format!("{b},{q}"), e.map(|e| e.value - exact.value))
            })
            .collect();
        found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let row = serde_json::json!({"a": x, "b": y, "exact": exact.value, "states": exact.states, "err": approx});
        println!("{row}");
    });
}
