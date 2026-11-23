//! Helpers shared by the command-line binaries.

use crate::data::Dex;
use crate::state::{Matchup, Mon, Rules};

/// Parses `SPECIES:MOVE1,MOVE2,...` (names case-insensitive, spaces or dashes allowed).
pub fn parse_mon(dex: &Dex, spec: &str, rules: &Rules) -> Result<Mon, String> {
    let (sp, mv) = spec.split_once(':').ok_or_else(|| format!("expected SPECIES:MOVES, got {spec}"))?;
    let si = dex.species_index(sp.trim()).ok_or_else(|| format!("unknown species {sp}"))?;
    let mut moves = Vec::new();
    for m in mv.split(',') {
        let m = m.trim();
        let id = dex.move_id(m).or_else(|| dex.move_id(&format!("{m}_M"))).ok_or_else(|| format!("unknown move {m}"))?;
        moves.push(id);
    }
    if moves.is_empty() || moves.len() > 4 {
        return Err(format!("{sp}: need 1-4 moves"));
    }
    Ok(Mon::new(dex, si, &moves, rules))
}

pub fn matchup(dex: &Dex, a: &str, b: &str, rules: Rules) -> Result<Matchup, String> {
    Ok(Matchup { mons: [parse_mon(dex, a, &rules)?, parse_mon(dex, b, &rules)?], rules })
}

pub fn describe(dex: &Dex, m: &Mon) -> String {
    let moves: Vec<&str> = m.moves.iter().filter(|&&x| x != 0).map(|&x| dex.mv(x).name.as_str()).collect();
    format!(
        "{} L{} HP{} Atk{} Def{} Spe{} Spc{} [{}]",
        dex.species[m.species as usize].name,
        m.level,
        m.max_hp,
        m.stats[0],
        m.stats[1],
        m.stats[2],
        m.stats[3],
        moves.join(", ")
    )
}
