//! 6v6 battles: two teams, switching, fainting and forced replacements, on top of the 1v1 turn
//! engine (`Engine::turn_switching`), following pokered's core.asm.
//!
//! - A switch happens before moves and uses up the side's turn. The turn is still ordered by Speed
//!   (the incoming Pokemon's) and by the side's stale move selection, and the incoming Pokemon still
//!   takes poison, burn and Leech Seed damage in its slot (`ExecutePlayerMoveDone`).
//! - Sending a Pokemon out clears its battle status bytes (Substitute, confusion, Leech Seed, Reflect,
//!   Toxic's escalation...) and stat stages, applies the paralysis and burn stat drops, and ends the
//!   opponent's Wrap/Bind (`SendOutMon` resets the opponent's USING_TRAPPING_MOVE).
//! - A faint ends the turn at once. The survivor keeps Rage, Thrash and the like. Its
//!   ATTACKING_MULTIPLE_TIMES flag is cleared and its Bide damage keeps only its low byte
//!   (`FaintEnemyPokemon`, a cartridge bug). The fainted side then sends out a replacement
//!   without using a turn.
//! - Clauses (Smogon RBY OU): a side's sleep moves fail while an opposing Pokemon is asleep from a
//!   move (not Rest), and its moves can't freeze while an opposing Pokemon is frozen, as with
//!   Showdown's Sleep Clause Mod and Freeze Clause Mod.
//!
//! - PP starts at the maximum (three PP Ups, as Showdown sets it) and is spent where the cartridge
//!   calls `DecrementPP`. With no PP left the side uses Struggle.
//!
//! Not modelled yet: Metronome, Mirror Move and Mimic, and Transform's separate PP.

use crate::chance::Chance;
use crate::data::Dex;
use crate::engine::{Engine, STRUGGLE_SLOT};
use crate::state::*;

/// A team member's lasting state. Volatile state lives only on the active Pokemon (`Battle6::b`).
#[derive(Debug, Clone)]
pub struct TeamMon {
    pub mon: Mon,
    pub hp: u16,
    /// Major status, with the sleep counter in the low bits (persists while benched).
    pub status: u8,
    /// Asleep because of the opponent's move (counts for Sleep Clause), not Rest.
    pub slept_by_foe: bool,
    pub pp: [u8; 4],
    /// The opponent has seen this Pokemon (it has been sent out).
    pub revealed: bool,
    /// Bit i: the opponent has seen this Pokemon use move slot i.
    pub moves_revealed: u8,
}

/// Maximum PP: base PP with three PP Ups (+3/5 of base).
pub fn max_pp(dex: &Dex, mv: u8) -> u8 {
    if mv == 0 { 0 } else { dex.mv(mv).pp / 5 * 8 }
}

/// The menu is skipped while recharging, raging, thrashing or charging (core.asm
/// `MainInBattleLoop`), so those turns allow no switch.
pub fn can_switch(x: &Side) -> bool {
    x.bs2 & (NEEDS_TO_RECHARGE | USING_RAGE) == 0 && x.bs1 & (THRASHING_ABOUT | CHARGING_UP) == 0
}

/// Whether an active Pokemon's sleep counts for Sleep Clause when it leaves the field.
pub fn slept_by_foe(x: &Side) -> bool {
    x.status & SLP_MASK != 0 && x.clauses & SLEPT_BY_FOE != 0
}

/// `SendOutMon`: side s's active Pokemon becomes `mon` with the given lasting state. Its battle
/// status is fresh, with the paralysis and burn stat drops applied; wPlayerSelectedMove is left as
/// it was (wPlayerMoveNum is zeroed), and the opponent's Wrap/Bind ends.
pub fn send_out_side(dex: &Dex, b: &mut Battle, s: usize, mon: &Mon, hp: u16, status: u8, slept_by_foe: bool) {
    let mut x = mon.fresh_side(dex);
    x.hp = hp;
    x.status = status;
    if status & PAR != 0 {
        x.stats[SPE] = (x.stats[SPE] >> 2).max(1);
    }
    if status & BRN != 0 {
        x.stats[ATK] = (x.stats[ATK] >> 1).max(1);
    }
    if slept_by_foe {
        x.clauses |= SLEPT_BY_FOE;
    }
    x.selected = b.sides[s].selected;
    b.sides[s] = x;
    b.sides[1 - s].bs1 &= !USING_TRAPPING_MOVE;
}

/// Sleep and Freeze Clause flags (`SLEEP_BLOCKED`, `FREEZE_BLOCKED`) for a side, given each opposing
/// team member's (major status, asleep from a move) with fainted members' status cleared.
pub fn clause_flags(foes: impl Iterator<Item = (u8, bool)>) -> u8 {
    let mut c = 0;
    for (st, by_foe) in foes {
        if st & SLP_MASK != 0 && by_foe {
            c |= SLEEP_BLOCKED;
        }
        if st & FRZ != 0 {
            c |= FREEZE_BLOCKED;
        }
    }
    c
}

/// `FaintEnemyPokemon`, run by the survivor's side for its own state after the opponent faints.
pub fn survivor_after_faint(o: &mut Side) {
    o.bs1 &= !ATTACKING_MULTIPLE_TIMES;
    o.bide_dmg &= 0xFF;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// A move slot (0-3), or `engine::STRUGGLE_SLOT`.
    Move(u8),
    /// Switch to this team index.
    Switch(u8),
    /// Nothing to choose (asleep, recharging, or not replacing a fainted Pokemon).
    Pass,
}

#[derive(Debug, Clone)]
pub struct Battle6 {
    pub teams: [Vec<TeamMon>; 2],
    pub active: [usize; 2],
    /// The two active Pokemon, as the 1v1 engine sees them.
    pub b: Battle,
    pub rules: Rules,
    pub turn: u32,
    /// need_replace[s]: side s's active fainted and it must send out another before the next turn.
    pub need_replace: [bool; 2],
    pub result: Option<Outcome>,
    /// wPlayerMoveListIndex: the slot last chosen, which a turn without a choice (charging, Thrash)
    /// spends PP from.
    last_slot: [u8; 2],
    /// 0 = exact damage rolls; otherwise the 39 rolls are grouped into this many equally likely
    /// buckets (`Engine::roll_buckets`). Search sets it to keep exact enumeration small.
    pub roll_buckets: u8,
}

impl Battle6 {
    /// Everything that can differ between two outcomes of the same turn, for merging identical ones.
    pub fn outcome_key(&self) -> Vec<u64> {
        let mut k: Vec<u64> = self.b.pack().to_vec();
        for s in 0..2 {
            for t in &self.teams[s] {
                k.push(t.hp as u64 | (t.status as u64) << 16 | (t.slept_by_foe as u64) << 24 | (t.revealed as u64) << 25
                    | (t.moves_revealed as u64) << 32 | (u32::from_le_bytes(t.pp) as u64) << 40);
                k.push(t.mon.moves.iter().fold(0u64, |a, &m| a << 8 | m as u64));
            }
            k.push(self.active[s] as u64 | (self.need_replace[s] as u64) << 8 | (self.last_slot[s] as u64) << 16);
        }
        k.push(match self.result {
            None => 0,
            Some(Outcome::Ongoing) => 1,
            Some(Outcome::Win(w)) => 2 + w as u64,
            Some(Outcome::Draw) => 4,
        });
        k
    }

    /// Both sides lead with their first Pokemon.
    pub fn new(dex: &Dex, rules: &Rules, teams: [Vec<Mon>; 2]) -> Battle6 {
        let teams = teams.map(|t| {
            t.into_iter()
                .map(|mon| TeamMon {
                    hp: mon.max_hp,
                    pp: mon.moves.map(|m| max_pp(dex, m)),
                    mon,
                    status: 0,
                    slept_by_foe: false,
                    revealed: false,
                    moves_revealed: 0,
                })
                .collect::<Vec<_>>()
        });
        assert!(teams.iter().all(|t| !t.is_empty() && t.len() <= 6), "teams need 1-6 Pokemon");
        let m = Matchup { mons: [teams[0][0].mon.clone(), teams[1][0].mon.clone()], rules: rules.clone() };
        let b = m.initial(dex);
        let mut teams = teams;
        teams[0][0].revealed = true;
        teams[1][0].revealed = true;
        Battle6 { teams, active: [0, 0], b, rules: rules.clone(), turn: 0, need_replace: [false; 2], result: None, last_slot: [0; 2], roll_buckets: 0 }
    }

    /// A battle already under way, rebuilt from its parts (e.g. a position observed on Showdown).
    /// The slot last chosen is unknown and taken as 0.
    pub fn from_parts(teams: [Vec<TeamMon>; 2], active: [usize; 2], b: Battle, rules: &Rules, turn: u32,
                      need_replace: [bool; 2]) -> Battle6 {
        assert!(teams.iter().all(|t| !t.is_empty() && t.len() <= 6), "teams need 1-6 Pokemon");
        Battle6 { teams, active, b, rules: rules.clone(), turn, need_replace, result: None, last_slot: [0; 2], roll_buckets: 0 }
    }

    pub fn matchup(&self) -> Matchup {
        Matchup { mons: [0, 1].map(|s| self.teams[s][self.active[s]].mon.clone()), rules: self.rules.clone() }
    }

    fn alive_bench(&self, s: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.teams[s].len()).filter(move |&i| i != self.active[s] && self.teams[s][i].hp > 0)
    }

    fn can_switch(&self, s: usize) -> bool {
        can_switch(&self.b.sides[s])
    }

    pub fn legal_actions(&self, dex: &Dex, s: usize) -> Vec<Action> {
        if self.result.is_some() {
            return Vec::new();
        }
        if self.need_replace.iter().any(|&r| r) {
            return if self.need_replace[s] {
                self.alive_bench(s).map(|i| Action::Switch(i as u8)).collect()
            } else {
                vec![Action::Pass]
            };
        }
        let m = self.matchup();
        let eng = Engine::new(dex, &m);
        let pp = self.teams[s][self.active[s]].pp;
        let mut v: Vec<Action> = eng
            .legal_choices(&self.b, s)
            .into_iter()
            .filter(|&slot| slot == STRUGGLE_SLOT || pp[slot as usize] > 0)
            .map(Action::Move)
            .collect();
        if eng.needs_choice(&self.b, s) && v.is_empty() {
            v.push(Action::Move(STRUGGLE_SLOT)); // out of PP
        }
        if !eng.needs_choice(&self.b, s) {
            v.push(Action::Pass); // stay in: asleep, frozen, trapped, storing Bide...
        }
        if self.can_switch(s) {
            v.extend(self.alive_bench(s).map(|i| Action::Switch(i as u8)));
        }
        v
    }

    /// Copies the active Pokemon's lasting state back to its team slot.
    fn save_active(&mut self, s: usize) {
        let x = &self.b.sides[s];
        let t = &mut self.teams[s][self.active[s]];
        t.hp = x.hp;
        t.status = x.status;
        t.slept_by_foe = slept_by_foe(x);
    }

    /// `SendOutMon`: team member `i` becomes side s's active Pokemon.
    fn send_out(&mut self, dex: &Dex, s: usize, i: usize) {
        let t = &self.teams[s][i];
        assert!(t.hp > 0 && i != self.active[s], "can't send out team index {i}");
        send_out_side(dex, &mut self.b, s, &t.mon, t.hp, t.status, t.slept_by_foe);
        self.active[s] = i;
        self.teams[s][i].revealed = true;
    }

    /// Sleep and Freeze Clause flags for both active Pokemon, from the state of the opposing team.
    fn set_clauses(&mut self) {
        for s in 0..2 {
            let o = 1 - s;
            let foe = |i: usize| -> (u8, bool) {
                if i == self.active[o] {
                    let x = &self.b.sides[o];
                    if self.rules.showdown {
                        // Its sleep is judged when the move lands (a sleeping target makes the move fail anyway; one
                        // that woke up this turn can be put back to sleep): only the bench blocks for the whole turn.
                        return (x.status & !SLP_MASK, false);
                    }
                    (x.status, x.clauses & SLEPT_BY_FOE != 0)
                } else {
                    let t = &self.teams[o][i];
                    (if t.hp > 0 { t.status } else { 0 }, t.slept_by_foe)
                }
            };
            let c = (self.b.sides[s].clauses & SLEPT_BY_FOE) | clause_flags((0..self.teams[o].len()).map(foe));
            self.b.sides[s].clauses = c;
        }
    }

    fn faint(&mut self, s: usize) {
        let t = &mut self.teams[s][self.active[s]];
        t.hp = 0;
        t.status = 0;
        t.slept_by_foe = false;
        survivor_after_faint(&mut self.b.sides[1 - s]);
    }

    /// Plays one decision point: either a turn, or sending out replacements after a faint.
    /// Returns the turn log when `log` is set.
    pub fn step<C: Chance>(&mut self, dex: &Dex, actions: [Action; 2], ch: &mut C, log: bool) -> Vec<String> {
        assert!(self.result.is_none(), "battle is over");
        for s in 0..2 {
            assert!(self.legal_actions(dex, s).contains(&actions[s]), "illegal action {:?} for side {s}", actions[s]);
        }
        let mut lines = Vec::new();
        if self.need_replace.iter().any(|&r| r) {
            for s in 0..2 {
                if let Action::Switch(i) = actions[s] {
                    self.send_out(dex, s, i as usize);
                    self.need_replace[s] = false;
                    if log {
                        lines.push(format!("  side {s} sends out {}", dex.species[self.teams[s][i as usize].mon.species as usize].name));
                    }
                }
            }
            return lines;
        }
        // Who held whom in a Wrap/Clamp/Bind/Fire Spin before anyone switched, and who could have chosen a move but for it.
        let trapping = [0, 1].map(|s| self.b.sides[s].bs1 & USING_TRAPPING_MOVE != 0);
        let free = [0, 1].map(|s| {
            let x = &self.b.sides[s];
            x.status & (FRZ | SLP_MASK) == 0
                && x.bs2 & (NEEDS_TO_RECHARGE | USING_RAGE) == 0
                && x.bs1 & (STORING_ENERGY | THRASHING_ABOUT | CHARGING_UP | USING_TRAPPING_MOVE) == 0
        });
        let mut switched = [false; 2];
        for s in 0..2 {
            if let Action::Switch(i) = actions[s] {
                self.save_active(s);
                self.send_out(dex, s, i as usize);
                switched[s] = true;
                if log {
                    lines.push(format!("  side {s} switches to {}", dex.species[self.teams[s][i as usize].mon.species as usize].name));
                }
            }
        }
        self.set_clauses();
        let mut choice = actions.map(|a| match a {
            Action::Move(slot) => slot,
            _ => 0,
        });
        // skip[s]: side s executes no move this turn (it switched, or see below).
        let mut skip = switched;
        if self.rules.showdown {
            for s in 0..2 {
                let o = 1 - s;
                if actions[s] != Action::Pass || !switched[o] {
                    continue;
                }
                if trapping[s] {
                    // Our target switched out of the trap: the move is used again, afresh, on what came in.
                    let sel = self.b.sides[s].selected;
                    if let Some(slot) = self.b.sides[s].moves.iter().position(|&mv| mv != 0 && mv == sel) {
                        choice[s] = slot as u8;
                        self.last_slot[s] = slot as u8;
                    }
                } else if trapping[o] && free[s] {
                    // The trapper left: we had no move to choose this turn, and make none.
                    skip[s] = true;
                }
            }
        }
        let m = self.matchup();
        let mut eng = Engine::new(dex, &m);
        eng.roll_buckets = self.roll_buckets;
        let r = eng.turn_switching(&self.b, choice, skip, ch, log);
        lines.extend(r.log);
        self.b = r.b;
        let outcome = r.outcome;
        for s in 0..2 {
            if let Action::Move(slot) = actions[s] {
                if slot != STRUGGLE_SLOT {
                    self.last_slot[s] = slot;
                }
            }
            if r.pp_spent[s] {
                let t = &mut self.teams[s][self.active[s]];
                t.pp[self.last_slot[s] as usize] = t.pp[self.last_slot[s] as usize].saturating_sub(1);
                t.moves_revealed |= 1 << self.last_slot[s];
            }
        }
        self.turn += 1;
        for s in 0..2 {
            self.save_active(s);
        }
        let fainted = match outcome {
            Outcome::Ongoing => [false, false],
            Outcome::Win(w) => [w == 1, w == 0],
            Outcome::Draw => [true, true],
        };
        for s in 0..2 {
            if fainted[s] {
                self.faint(s);
            }
        }
        let left = [0, 1].map(|s| self.teams[s].iter().any(|t| t.hp > 0));
        self.result = match left {
            [true, true] => None,
            [true, false] => Some(Outcome::Win(0)),
            [false, true] => Some(Outcome::Win(1)),
            [false, false] => Some(Outcome::Draw),
        };
        if self.result.is_none() {
            self.need_replace = fainted;
        }
        lines
    }
}
