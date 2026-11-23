//! Which state fields vary most across the reachable state space of a matchup?
use gen1::cli::matchup;
use gen1::data::Dex;
use gen1::engine::Engine;
use gen1::solver::Solver;
use gen1::state::Rules;
use std::collections::HashSet;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = Dex::load_default();
    let m = matchup(&d, &a[1], &a[2], Rules::default()).unwrap();
    let mut eng = Engine::new(&d, &m);
    eng.roll_buckets = a.get(3).map(|x| x.parse().unwrap()).unwrap_or(1);
    let mut s = Solver::new(eng);
    s.uniform = true;
    s.set_hp_levels(a.get(4).map(|x| x.parse().unwrap()).unwrap_or(0));
    s.max_states = a.get(5).map(|x| x.parse().unwrap()).unwrap_or(usize::MAX);
    let v = s.try_solve(&m.initial(&d));
    println!("value {v:?}, {} states", s.states.len());
    let mut sets: Vec<(&str, HashSet<String>)> = Vec::new();
    let fields: Vec<(&str, Box<dyn Fn(&gen1::state::Battle) -> String>)> = vec![
        ("hp pair", Box::new(|b| format!("{} {}", b.sides[0].hp, b.sides[1].hp))),
        ("damage", Box::new(|b| format!("{}", b.damage))),
        ("stats", Box::new(|b| format!("{:?} {:?}", b.sides[0].stats, b.sides[1].stats))),
        ("mods", Box::new(|b| format!("{:?} {:?}", b.sides[0].mods, b.sides[1].mods))),
        ("status", Box::new(|b| format!("{} {}", b.sides[0].status, b.sides[1].status))),
        ("bs", Box::new(|b| format!("{:?}", b.sides.iter().map(|x| (x.bs1, x.bs2, x.bs3)).collect::<Vec<_>>()))),
        ("counters", Box::new(|b| format!("{:?}", b.sides.iter().map(|x| (x.confused, x.toxic, x.disabled, x.attacks_left, x.sub_hp, x.bide_dmg)).collect::<Vec<_>>()))),
        ("speeds", Box::new(|b| format!("{} {}", b.sides[0].stats[2], b.sides[1].stats[2]))),
        ("attacks", Box::new(|b| format!("{} {}", b.sides[0].stats[0], b.sides[1].stats[0]))),
        ("sleep_used", Box::new(|b| format!("{} {}", b.sides[0].sleep_used, b.sides[1].sleep_used))),
        ("selected/ram", Box::new(|b| format!("{:?}", b.sides.iter().map(|x| (x.selected, x.ram)).collect::<Vec<_>>()))),
        ("all but hp", Box::new(|b| { let mut c = b.clone(); c.sides[0].hp = 0; c.sides[1].hp = 0; format!("{c:?}") })),
    ];
    for (n, _) in &fields { sets.push((n, HashSet::new())); }
    for b in &s.states { for (i, (_, f)) in fields.iter().enumerate() { sets[i].1.insert(f(b)); } }
    for (n, set) in sets { println!("{n:>14}: {} distinct", set.len()); }
}
