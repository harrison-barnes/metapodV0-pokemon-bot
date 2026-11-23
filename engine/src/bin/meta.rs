//! Every command takes [--rules video|smogon|smogon-pp] (default video; see `Rules::named`).
//! meta pools  [--out sets.json]                         heuristic starting sets for every fully evolved Pokemon
//! meta table  --sets sets.json --out table.jsonl [--precision screen|final|big] [--retry-unsolved] [--threads T]
//! meta search --sets field.json --out better.json [--table table.jsonl] [--gauntlet 16] [--pool 13]
//!             [--iters 8] [--only A,B] [--race] [--threads T]
//!             local search over each Pokemon's moveset (one move swapped at a time), scored by mean
//!             value against the strongest opponents' current sets
//!             solves every pair of sets (resumes from an existing output file); --retry-unsolved
//!             re-solves only the pairs recorded as unsolved, appending rows that supersede them
//! meta dataset --sets pool.json --out DIR [--mid 500] [--shard K/N] [--pairs FILE] [--limit N] [--seed S] [--threads T]
//!             the 1v1 dataset (docs/DATASET.md) for every pair of sets; resumes from DIR/done.jsonl
//! meta endgame "A1:MOVES|A2:MOVES" "B1:MOVES|B2:MOVES" [--precision B,L[,BL][;B,L[,BL]...]] [--max-states N]
//!             [--hp A1,A2/B1,B2]
//!             exact 2v2 endgame with switching (`endgame`): both sides lead with their first
//!             Pokemon (HP 0 in --hp = fainted). Tries each roll-bucket/HP-level/bench-HP-level
//!             precision in turn
//!             (default `endgame::LADDER`) and prints the value, state count, time and the opening
//!             mixed strategies

use gen1::cli::{describe, parse_mon};
use gen1::data::Dex;
use gen1::meta::{evaluate, focused_pool, heuristic_set, move_pool, to_set, Precision, Set, BIG, COARSE, FINAL, SCREEN};
use std::collections::{HashMap, HashSet};
use gen1::state::{Mon, Rules};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::sync::Mutex;
use std::time::Instant;

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

#[derive(Serialize, Deserialize)]
pub struct Row {
    pub a: usize,
    pub b: usize,
    pub value: Option<f64>,
    pub states: usize,
    pub secs: f64,
    pub precision: Option<Precision>,
    #[serde(default)]
    pub opening: Option<[Vec<(String, f64)>; 2]>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = Dex::load_default();
    let rules = match arg(&args, "--rules") {
        Some(name) => Rules::named(name).unwrap_or_else(|| panic!("unknown ruleset {name} (video|smogon|smogon-pp)")),
        None => Rules::default(),
    };
    match args.get(1).map(|s| s.as_str()) {
        Some("pools") => {
            let roster: Vec<usize> = (0..dex.species.len())
                .filter(|&i| dex.species[i].fully_evolved && !rules.banned_species.contains(&dex.species[i].name.as_str()))
                .collect();
            let field: Vec<Mon> = roster.iter().map(|&i| Mon::new(&dex, i, &[1], &rules)).collect();
            let mut sets = Vec::new();
            for &i in &roster {
                let pool = move_pool(&dex, &rules, i);
                let set = heuristic_set(&dex, &rules, i, &pool, &field);
                let names: Vec<&str> = pool.iter().map(|&m| dex.mv(m).name.as_str()).collect();
                println!("{:<11} pool {:>2}: {}", dex.species[i].name, pool.len(), names.join(" "));
                println!("{:<11} start: {}", "", to_set(&dex, i, &set).moves.join(", "));
                sets.push(to_set(&dex, i, &set));
            }
            let out = arg(&args, "--out").unwrap_or("../data/results/sets_heuristic.json");
            std::fs::write(out, serde_json::to_string_pretty(&sets).unwrap()).unwrap();
            eprintln!("wrote {} sets to {out}", sets.len());
        }
        Some("table") => {
            let sets: Vec<Set> = serde_json::from_str(&std::fs::read_to_string(arg(&args, "--sets").expect("--sets")).unwrap()).unwrap();
            let out = arg(&args, "--out").expect("--out");
            let ladder = match arg(&args, "--precision") {
                Some("final") => FINAL,
                Some("big") => BIG,
                _ => SCREEN,
            };
            let retry = args.iter().any(|a| a == "--retry-unsolved");
            if let Some(t) = arg(&args, "--threads") {
                rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().unwrap();
            }
            // Later rows supersede earlier ones for the same pair.
            let mut solved: HashMap<(usize, usize), bool> = HashMap::new();
            if let Ok(f) = std::fs::File::open(out) {
                for r in std::io::BufReader::new(f).lines().filter_map(|l| serde_json::from_str::<Row>(&l.ok()?).ok()) {
                    let ok = r.value.is_some() || solved.get(&(r.a, r.b)) == Some(&true);
                    solved.insert((r.a, r.b), ok);
                }
            }
            let mut pairs: Vec<(usize, usize)> = (0..sets.len())
                .flat_map(|a| (a + 1..sets.len()).map(move |b| (a, b)))
                .filter(|p| if retry { solved.get(p) == Some(&false) } else { !solved.contains_key(p) })
                .collect();
            let done = solved.len();
            // Interleave so progress estimates are representative.
            pairs.sort_by_key(|&(a, b)| (a * 7919 + b * 104729) % 1_000_003);
            let total = pairs.len();
            eprintln!("{} pairs to solve ({} already recorded)", total, done);
            let file = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap());
            let counter = Mutex::new(0usize);
            let t0 = Instant::now();
            pairs.par_iter().for_each(|&(a, b)| {
                let e = evaluate(&dex, &rules, &sets[a], &sets[b], ladder);
                let row = match &e {
                    Some(e) => Row {
                        a,
                        b,
                        value: Some(e.value),
                        states: e.states,
                        secs: e.secs,
                        precision: Some(e.precision),
                        opening: Some(e.opening.clone()),
                    },
                    None => Row { a, b, value: None, states: 0, secs: 0.0, precision: None, opening: None },
                };
                let line = serde_json::to_string(&row).unwrap();
                writeln!(file.lock().unwrap(), "{line}").unwrap();
                let mut c = counter.lock().unwrap();
                *c += 1;
                if *c % 50 == 0 || *c == total {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!("{}/{} done, {:.0}s elapsed, ~{:.0}s left", c, total, el, el / *c as f64 * (total - *c) as f64);
                }
            });
        }
        Some("search") => search(&args, &dex, &rules),
        Some("dataset") => dataset(&args, &dex, &rules),
        Some("endgame") => endgame(&args, &dex, &rules),
        _ => eprintln!("usage: meta pools|table|search|dataset|endgame ..."),
    }
}

/// Successive halving over opponents: every candidate plays the first few opponents, the leaders
/// play a few more, and so on, until the survivors have played everyone. Returns candidate indices,
/// best first, with every survivor ranked on the full gauntlet ahead of everything eliminated.
/// Solves are deterministic, so the only risk is a candidate that starts badly against the first
/// opponents and would have recovered; a margin keeps anything close to the cut alive.
fn race_candidates(cands: &[Vec<u8>], opps: &[usize], score: impl Fn(&[Vec<u8>], &[usize]) -> Vec<f64>) -> Vec<usize> {
    const KEEP: f64 = 0.25; // fraction surviving each round
    const MIN_KEEP: usize = 6;
    const MARGIN: f64 = 0.05; // anything this close to the cut also survives
    let rounds: [usize; 3] = [3, 6, opps.len()]; // opponents played after each round
    let mut alive: Vec<usize> = (0..cands.len()).collect();
    let mut sum = vec![0.0; cands.len()];
    let mut eliminated: Vec<(usize, f64)> = Vec::new(); // (candidate, mean when cut)
    let mut played = 0;
    for (r, &upto) in rounds.iter().enumerate() {
        let upto = upto.min(opps.len());
        if upto > played {
            let batch: Vec<Vec<u8>> = alive.iter().map(|&c| cands[c].clone()).collect();
            let means = score(&batch, &opps[played..upto]);
            for (k, &c) in alive.iter().enumerate() {
                sum[c] += means[k] * (upto - played) as f64;
            }
            played = upto;
        }
        alive.sort_by(|&a, &b| sum[b].partial_cmp(&sum[a]).unwrap());
        if r + 1 == rounds.len() || played == opps.len() {
            break;
        }
        let keep = ((alive.len() as f64 * KEEP).ceil() as usize).max(MIN_KEEP).min(alive.len());
        let cut = sum[alive[keep - 1]] / played as f64 - MARGIN;
        let (stay, go): (Vec<usize>, Vec<usize>) = alive.iter().partition(|&&c| sum[c] / played as f64 >= cut);
        eliminated.extend(go.iter().map(|&c| (c, sum[c] / played as f64)));
        alive = stay;
    }
    eliminated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    alive.into_iter().chain(eliminated.into_iter().map(|(c, _)| c)).collect()
}

/// Mean value of each set against the field, from a table file.
fn table_scores(path: &str, n: usize) -> Vec<f64> {
    let mut sum = vec![0.0; n];
    let mut cnt = vec![0usize; n];
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let r: Row = serde_json::from_str(line).unwrap();
        if let Some(v) = r.value {
            sum[r.a] += v;
            sum[r.b] += 1.0 - v;
            cnt[r.a] += 1;
            cnt[r.b] += 1;
        }
    }
    (0..n).map(|i| if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { 0.0 }).collect()
}

#[derive(Serialize, Deserialize, Clone)]
struct SearchLog {
    species: String,
    start: Vec<String>,
    start_score: f64,
    end: Vec<String>,
    end_score: f64,
    steps: Vec<(Vec<String>, f64)>,
    pool: Vec<String>,
    gauntlet: Vec<String>,
}

fn search(args: &[String], dex: &Dex, rules: &Rules) {
    let sets: Vec<Set> = serde_json::from_str(&std::fs::read_to_string(arg(args, "--sets").expect("--sets")).unwrap()).unwrap();
    let out = arg(args, "--out").expect("--out");
    let k: usize = arg(args, "--gauntlet").map(|x| x.parse().unwrap()).unwrap_or(16);
    let pool_size: usize = arg(args, "--pool").map(|x| x.parse().unwrap()).unwrap_or(13);
    let iters: usize = arg(args, "--iters").map(|x| x.parse().unwrap()).unwrap_or(8);
    let race = args.iter().any(|a| a == "--race");
    if let Some(t) = arg(args, "--threads") {
        rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().unwrap();
    }
    let n = sets.len();
    let scores = arg(args, "--table").map(|t| table_scores(t, n)).unwrap_or(vec![0.5; n]);
    let mut by_strength: Vec<usize> = (0..n).collect();
    by_strength.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let only: Option<Vec<String>> = arg(args, "--only").map(|s| s.split(',').map(|x| x.to_ascii_uppercase()).collect());
    let field: Vec<Mon> = sets.iter().map(|s| s.mon(dex, rules)).collect();
    let cache: Mutex<HashMap<(String, String), f64>> = Mutex::new(HashMap::new());
    let unsolved = Mutex::new(0usize);
    // Resume: keep species already finished by an earlier run with the same output file.
    let prior: Vec<SearchLog> = std::fs::read_to_string(format!("{out}.log.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let mut start = sets.clone();
    for l in &prior {
        if let Some(i) = sets.iter().position(|s| s.species == l.species) {
            start[i].moves = l.end.clone();
        }
    }
    let state = Mutex::new((start, prior));
    let t0 = Instant::now();
    let todo: Vec<usize> = (0..n)
        .filter(|&i| only.as_ref().is_none_or(|o| o.contains(&sets[i].species)))
        .filter(|&i| !state.lock().unwrap().1.iter().any(|l| l.species == sets[i].species))
        .collect();
    eprintln!("{} species to search", todo.len());
    // Species run concurrently (opponents' sets are fixed for the round), so one slow solve
    // never leaves the other cores idle.
    todo.par_iter().for_each(|&i| {
        let si = dex.species_index(&sets[i].species).unwrap();
        let opps: Vec<usize> = by_strength.iter().copied().filter(|&o| o != i).take(if k == 0 { n } else { k }).collect();
        let current_ids: Vec<u8> = sets[i].moves.iter().map(|m| dex.move_id(m).unwrap()).collect();
        let pool = focused_pool(dex, rules, si, &move_pool(dex, rules, si), &field, &current_ids, pool_size);
        // Mean value of each candidate set against `against`: every (candidate, opponent) solve
        // runs in parallel.
        let score_vs = |cands: &[Vec<u8>], against: &[usize], ladder: &'static [Precision]| -> Vec<f64> {
            let jobs: Vec<(usize, usize)> = (0..cands.len()).flat_map(|c| against.iter().map(move |&o| (c, o))).collect();
            let vals: Vec<(usize, f64)> = jobs
                .par_iter()
                .map(|&(c, o)| {
                    let set = to_set(dex, si, &cands[c]);
                    let key = (format!("{}|{}", set.label(), ladder.len()), sets[o].label());
                    if let Some(&v) = cache.lock().unwrap().get(&key) {
                        return (c, v);
                    }
                    let v = match evaluate(dex, rules, &set, &sets[o], ladder) {
                        Some(e) => e.value,
                        None => {
                            *unsolved.lock().unwrap() += 1;
                            eprintln!("  unsolved: {} vs {}", set.label(), sets[o].label());
                            0.5
                        }
                    };
                    cache.lock().unwrap().insert(key, v);
                    (c, v)
                })
                .collect();
            let mut sum = vec![0.0; cands.len()];
            for (c, v) in vals {
                sum[c] += v;
            }
            sum.iter().map(|s| s / against.len() as f64).collect()
        };
        let score_all = |cands: &[Vec<u8>], ladder: &'static [Precision]| score_vs(cands, &opps, ladder);
        let mut current = current_ids.clone();
        let mut best = score_all(&[current.clone()], SCREEN)[0];
        let start_score = best;
        let mut steps = Vec::new();
        for _ in 0..iters {
            let mut cands: Vec<Vec<u8>> = Vec::new();
            for slot in 0..current.len().max(1).min(4) {
                for &m in &pool {
                    if !current.contains(&m) {
                        let mut c = current.clone();
                        c[slot] = m;
                        cands.push(c);
                    }
                }
            }
            if current.len() < 4 {
                for &m in &pool {
                    if !current.contains(&m) {
                        let mut c = current.clone();
                        c.push(m);
                        cands.push(c);
                    }
                }
            }
            if cands.is_empty() {
                break;
            }
            // Rank everything coarsely, then re-score the top three at screening precision.
            let order = if race {
                race_candidates(&cands, &opps, |c, a| score_vs(c, a, COARSE))
            } else {
                let coarse = score_all(&cands, COARSE);
                let mut order: Vec<usize> = (0..cands.len()).collect();
                order.sort_by(|&a, &b| coarse[b].partial_cmp(&coarse[a]).unwrap());
                order
            };
            let top: Vec<Vec<u8>> = order.iter().take(3).map(|&i| cands[i].clone()).collect();
            let fine = score_all(&top, SCREEN);
            let (bi, bv) = fine.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, v)| (i, *v)).unwrap();
            let cands = top;
            if bv <= best + 1e-3 {
                break;
            }
            current = cands[bi].clone();
            best = bv;
            steps.push((to_set(dex, si, &current).moves, bv));
        }
        let end = to_set(dex, si, &current);
        eprintln!(
            "[{:>5.0}s] {:<11} {:.3} -> {:.3}  {}",
            t0.elapsed().as_secs_f64(),
            sets[i].species,
            start_score,
            best,
            end.moves.join(", ")
        );
        let log = SearchLog {
            species: sets[i].species.clone(),
            start: sets[i].moves.clone(),
            start_score,
            end: end.moves.clone(),
            end_score: best,
            steps,
            pool: pool.iter().map(|&m| dex.mv(m).name.clone()).collect(),
            gauntlet: opps.iter().map(|&o| sets[o].species.clone()).collect(),
        };
        let mut st = state.lock().unwrap();
        st.0[i] = end;
        st.1.push(log);
        std::fs::write(out, serde_json::to_string_pretty(&st.0).unwrap()).unwrap();
        std::fs::write(format!("{out}.log.json"), serde_json::to_string_pretty(&st.1).unwrap()).unwrap();
    });
    eprintln!("done in {:.0}s; {} solves hit every state cap (scored 0.5)", t0.elapsed().as_secs_f64(), unsolved.lock().unwrap());
}

#[derive(Serialize, Deserialize)]
struct DoneRow {
    a: usize,
    b: usize,
    /// Byte offset of this pair's rows in rows.bin, and how many there are.
    offset: u64,
    rows: usize,
    states: usize,
    secs: f64,
    precision: Option<Precision>,
    #[serde(default)]
    timing: Option<gen1::dataset::Timing>,
}

fn dataset(args: &[String], dex: &Dex, rules: &Rules) {
    use gen1::dataset::{generate_pair, schema, Config, ROW_BYTES};
    let sets_path = arg(args, "--sets").expect("--sets");
    let sets: Vec<Set> = serde_json::from_str(&std::fs::read_to_string(sets_path).unwrap()).unwrap();
    let dir = std::path::PathBuf::from(arg(args, "--out").expect("--out"));
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(t) = arg(args, "--threads") {
        rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().unwrap();
    }
    let cfg = Config {
        mid_rows: arg(args, "--mid").map(|x| x.parse().unwrap()).unwrap_or(500),
        seed: arg(args, "--seed").map(|x| x.parse().unwrap()).unwrap_or(1),
        ..Config::default()
    };
    std::fs::write(dir.join("schema.json"), serde_json::to_string_pretty(&schema()).unwrap()).unwrap();
    std::fs::copy(sets_path, dir.join("sets.json")).unwrap();

    // Resume: keep only rows that a done.jsonl line accounts for (a crash can leave a partial write).
    let done_path = dir.join("done.jsonl");
    let rows_path = dir.join("rows.bin");
    let done: Vec<DoneRow> = std::fs::read_to_string(&done_path)
        .map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
        .unwrap_or_default();
    let end = done.iter().map(|d| d.offset + (d.rows * ROW_BYTES) as u64).max().unwrap_or(0);
    let rows_file = std::fs::OpenOptions::new().create(true).read(true).write(true).truncate(false).open(&rows_path).unwrap();
    rows_file.set_len(end).unwrap();
    let finished: HashSet<(usize, usize)> = done.iter().map(|d| (d.a, d.b)).collect();
    let mut pairs: Vec<(usize, usize)> =
        (0..sets.len()).flat_map(|a| (a..sets.len()).map(move |b| (a, b))).filter(|p| !finished.contains(p)).collect();
    // --shard K/N: this machine takes every pair whose index in the full list is K mod N.
    if let Some(spec) = arg(args, "--shard") {
        let (k, n) = spec.split_once('/').expect("--shard K/N");
        let (k, n): (usize, usize) = (k.parse().unwrap(), n.parse().unwrap());
        let index = |a: usize, b: usize| a * sets.len() + b;
        pairs.retain(|&(a, b)| index(a, b) % n == k);
    }
    // --pairs FILE: only the set pairs listed in FILE, a JSON list of [a, b] (e.g. re-solving the
    // stall pairs under another ruleset).
    if let Some(path) = arg(args, "--pairs") {
        let wanted: HashSet<(usize, usize)> = serde_json::from_str::<Vec<(usize, usize)>>(&std::fs::read_to_string(path).unwrap())
            .unwrap()
            .into_iter()
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        pairs.retain(|p| wanted.contains(p));
    }
    // --pairs FILE: only the set pairs listed in FILE, a JSON list of [a, b] (e.g. re-solving the
    // stall pairs under another ruleset).
    if let Some(path) = arg(args, "--pairs") {
        let wanted: HashSet<(usize, usize)> = serde_json::from_str::<Vec<(usize, usize)>>(&std::fs::read_to_string(path).unwrap())
            .unwrap()
            .into_iter()
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        pairs.retain(|p| wanted.contains(p));
    }
    pairs.sort_by_key(|&(a, b)| (a * 7919 + b * 104729) % 1_000_003);
    if let Some(n) = arg(args, "--limit") {
        pairs.truncate(n.parse().unwrap());
    }
    let total = pairs.len();
    eprintln!("{total} set pairs to solve ({} already done)", finished.len());
    let out = Mutex::new((rows_file, std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap(), end, 0usize));
    let t0 = Instant::now();
    pairs.par_iter().for_each(|&(a, b)| {
        let t = Instant::now();
        let r = generate_pair(dex, rules, &sets, a, b, &cfg);
        let mut bytes = Vec::new();
        let (n, states, precision, timing) = match &r {
            Some((rows, states, timing)) => {
                rows.iter().for_each(|row| row.write(&mut bytes));
                (rows.len(), *states, Some(rows[0].precision), Some(*timing))
            }
            None => (0, 0, None, None),
        };
        let mut g = out.lock().unwrap();
        let offset = g.2;
        {
            use std::io::{Seek, SeekFrom};
            g.0.seek(SeekFrom::Start(offset)).unwrap();
            g.0.write_all(&bytes).unwrap();
        }
        g.2 += bytes.len() as u64;
        let line = DoneRow { a, b, offset, rows: n, states, secs: t.elapsed().as_secs_f64(), precision, timing };
        writeln!(g.1, "{}", serde_json::to_string(&line).unwrap()).unwrap();
        g.3 += 1;
        if g.3 % 20 == 0 || g.3 == total {
            let el = t0.elapsed().as_secs_f64();
            eprintln!("{}/{} pairs, {:.1} GB, {:.0}s elapsed, ~{:.0}s left", g.3, total, g.2 as f64 / 1e9, el, el / g.3 as f64 * (total - g.3) as f64);
        }
    });
}

fn endgame(args: &[String], dex: &Dex, rules: &Rules) {
    use gen1::endgame::{solve_ladder, EndgamePrecision, EndgameSolver, Teams, LADDER};
    let team = |spec: &str| -> [Mon; 2] {
        let mons: Vec<Mon> = spec.split('|').map(|m| parse_mon(dex, m, rules).unwrap_or_else(|e| panic!("{e}"))).collect();
        mons.try_into().unwrap_or_else(|_| panic!("need exactly two Pokemon per side: {spec}"))
    };
    let teams = Teams::new([team(&args[2]), team(&args[3])], rules.clone());
    for (s, t) in teams.mons.iter().enumerate() {
        for m in t {
            println!("side {s}: {}", describe(dex, m));
        }
    }
    let max_states: usize = arg(args, "--max-states").map(|x| x.parse().unwrap()).unwrap_or(4_000_000);
    let ladder: Vec<EndgamePrecision> = match arg(args, "--precision") {
        Some(spec) => spec
            .split(';')
            .map(|p| {
                let v: Vec<u16> = p.split(',').map(|x| x.trim().parse().unwrap()).collect();
                assert!(v.len() == 2 || v.len() == 3, "--precision BUCKETS,HP_LEVELS[,BENCH_LEVELS]");
                EndgamePrecision::new(v[0] as u8, v[1], v.get(2).copied().unwrap_or(0), max_states)
            })
            .collect(),
        None => LADDER.to_vec(),
    };
    let solver = EndgameSolver::new(dex, &teams);
    let root = arg(args, "--hp").map(|spec| {
        let hp: Vec<[u16; 2]> = spec
            .split('/')
            .map(|t| {
                let v: Vec<u16> = t.split(',').map(|x| x.trim().parse().unwrap()).collect();
                [v[0], v[1]]
            })
            .collect();
        solver.position([0, 0], [hp[0], hp[1]], [[0; 2]; 2])
    });
    match solve_ladder(dex, &teams, root.as_ref(), &ladder, true) {
        Some(e) => {
            let p = e.precision;
            println!(
                "value for side 0: {:.6}  ({} states, {:.2}s, {} roll buckets, {} HP levels, {} bench levels; largest SCC {}, {} sweeps, {} paths, {} transitions)",
                e.value, e.states, e.secs, p.buckets, p.hp_levels, p.bench_levels, e.largest_scc, e.sweeps, e.paths, e.transitions
            );
            for s in 0..2 {
                println!("opening strategy side {s}:");
                for (name, q) in &e.opening[s] {
                    println!("  {name:>22} {:6.2}%", q * 100.0);
                }
            }
        }
        None => println!("unsolved: every precision exceeded its state cap"),
    }
}
