//! Playing against the game itself rather than a link opponent: the player's choice is the only real
//! decision, because the game follows fixed code. Ported from `SelectEnemyMove`, `TrainerAI` and
//! `AIEnemyTrainerChooseMoves` in engine/battle/{core,trainer_ai}.asm.
//!
//! Side 0 is the player, side 1 the game. Differences from link rules live in `Engine::sp` (badge
//! boosts and the 25% stat-down miss) and here (AI selection, AI actions at the game's slot in the
//! turn order, unlimited enemy PP, deterministic replacements, the bag).

use crate::battle6::{send_out_side, survivor_after_faint, TeamMon};
use crate::chance::Chance;
use crate::data::{eff, Dex};
use crate::engine::{apply_badges, Engine, SinglePlayer, SpTurn, STRUGGLE_SLOT};
use crate::state::*;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiAction {
    Switch,
    XStat(usize),
    GuardSpec,
    FullHeal,
    Heal(u16),
    FullRestore,
}

/// One rule of a trainer's special-action routine. Rules run in order; the first that passes fires.
#[derive(Debug, Clone)]
pub struct AiRule {
    /// Passes when the routine's random byte is below this.
    pub below: Option<u32>,
    /// Passes when current HP < max HP / this.
    pub hp_below: Option<u32>,
    pub needs_status: bool,
    pub action: AiAction,
}

#[derive(Debug, Clone, Default)]
pub struct AiSpec {
    pub layers: Vec<u8>,
    pub max_uses: u8,
    pub rules: Vec<AiRule>,
    /// data/types/type_matchups.asm in game order as (attacker, defender, multiplier x10).
    pub type_effects: Vec<(u8, u8, u8)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemKind {
    Heal(u16),
    FullRestore,
    /// Cures the status bits in this mask.
    Cure(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BagItem {
    pub kind: ItemKind,
    pub count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerAction {
    /// A move slot, or `STRUGGLE_SLOT`.
    Move(u8),
    Switch(u8),
    /// A bag index, used on the active Pokémon.
    Item(u8),
    /// Asleep, recharging and so on: stay in and let the turn happen.
    Pass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Finish {
    Win,
    Loss,
}

#[derive(Clone)]
pub struct TrainerBattle {
    pub teams: [Vec<TeamMon>; 2],
    pub active: [usize; 2],
    pub b: Battle,
    /// None in a wild battle.
    pub ai: Option<Arc<AiSpec>>,
    /// wAICount: 0xFF until the first check after a send-out.
    pub ai_count: u8,
    /// wAILayer2Encouragement.
    pub layer2: u8,
    pub badges: u8,
    pub bag: Vec<BagItem>,
    /// wPlayerMoveListIndex.
    pub last_slot: u8,
    pub need_replace: bool,
    pub result: Option<Finish>,
    pub rules: Rules,
}

fn sp_rules(level: u8) -> Rules {
    Rules { level, sleep_clause: false, heal_pp: false, ..Rules::default() }
}

impl TrainerBattle {
    /// A battle from its first turn, both sides leading with their first Pokémon.
    pub fn new(dex: &Dex, player: Vec<Mon>, enemy: Vec<Mon>, ai: Option<Arc<AiSpec>>, badges: u8, bag: Vec<BagItem>) -> TrainerBattle {
        let team = |t: Vec<Mon>| -> Vec<TeamMon> {
            t.into_iter()
                .map(|mon| TeamMon {
                    hp: mon.max_hp,
                    pp: mon.moves.map(|m| if m == 0 { 0 } else { dex.mv(m).pp }),
                    mon,
                    status: 0,
                    slept_by_foe: false,
                    revealed: true,
                    moves_revealed: 0,
                })
                .collect()
        };
        let teams = [team(player), team(enemy)];
        let rules = sp_rules(teams[0][0].mon.level);
        let m = Matchup { mons: [teams[0][0].mon.clone(), teams[1][0].mon.clone()], rules: rules.clone() };
        let mut b = m.initial(dex);
        apply_badges(&mut b.sides[0], badges);
        TrainerBattle { teams, active: [0, 0], b, ai, ai_count: 0xFF, layer2: 0, badges, bag, last_slot: 0, need_replace: false, result: None, rules }
    }

    fn matchup(&self) -> Matchup {
        Matchup { mons: [0, 1].map(|s| self.teams[s][self.active[s]].mon.clone()), rules: self.rules.clone() }
    }

    /// Everything that can affect the rest of the battle, for merging identical positions.
    pub fn key(&self) -> Vec<u64> {
        let mut k: Vec<u64> = self.b.pack().to_vec();
        for s in 0..2 {
            for t in &self.teams[s] {
                k.push(t.hp as u64 | (t.status as u64) << 16 | if s == 0 { (u32::from_le_bytes(t.pp) as u64) << 24 } else { 0 });
            }
            k.push(self.active[s] as u64);
        }
        for i in &self.bag {
            k.push(i.count as u64);
        }
        k.push(self.ai_count as u64 | (self.layer2.min(2) as u64) << 8 | (self.last_slot as u64) << 16 | (self.need_replace as u64) << 24);
        k.push(match self.result {
            None => 0,
            Some(Finish::Win) => 1,
            Some(Finish::Loss) => 2,
        });
        k
    }

    pub fn player_actions(&self, dex: &Dex) -> Vec<PlayerAction> {
        if self.result.is_some() {
            return Vec::new();
        }
        let alive_bench = (0..self.teams[0].len()).filter(|&i| i != self.active[0] && self.teams[0][i].hp > 0);
        if self.need_replace {
            return alive_bench.map(|i| PlayerAction::Switch(i as u8)).collect();
        }
        let m = self.matchup();
        let eng = Engine::new(dex, &m);
        let pp = self.teams[0][self.active[0]].pp;
        let mut v: Vec<PlayerAction> = eng
            .legal_choices(&self.b, 0)
            .into_iter()
            .filter(|&s| s == STRUGGLE_SLOT || pp[s as usize] > 0)
            .map(PlayerAction::Move)
            .collect();
        if eng.needs_choice(&self.b, 0) && v.is_empty() {
            v.push(PlayerAction::Move(STRUGGLE_SLOT));
        }
        if !eng.needs_choice(&self.b, 0) {
            v.push(PlayerAction::Pass);
        }
        if crate::battle6::can_switch(&self.b.sides[0]) {
            v.extend(alive_bench.map(|i| PlayerAction::Switch(i as u8)));
            let me = &self.b.sides[0];
            let max = m.mons[0].max_hp;
            for (i, it) in self.bag.iter().enumerate() {
                let useful = it.count > 0
                    && match it.kind {
                        ItemKind::Heal(_) => me.hp < max,
                        ItemKind::FullRestore => me.hp < max || me.status != 0,
                        ItemKind::Cure(mask) => me.status & mask != 0,
                    };
                if useful {
                    v.push(PlayerAction::Item(i as u8));
                }
            }
        }
        v
    }

    /// The chance of each move slot, out of 256 per re-roll. See `SelectEnemyMove`: the 25% splits are
    /// really 63, 64, 63 and 66, and an unavailable slot rolls again.
    pub fn enemy_move_weights(&self, dex: &Dex) -> [u32; 4] {
        const ROLL: [u32; 4] = [63, 64, 63, 66];
        let me = &self.b.sides[1];
        let moves = me.moves;
        let mut allowed = [false; 4];
        for i in 0..4 {
            allowed[i] = moves[i] != 0;
        }
        let disabled_slot = (me.disabled >> 4) as usize;
        if let Some(ai) = self.ai.as_deref().filter(|a| !a.layers.is_empty()) {
            let mut buf = [10i32; 4];
            if disabled_slot > 0 {
                buf[disabled_slot - 1] = 0x50;
            }
            let player = &self.b.sides[0];
            for &layer in &ai.layers {
                for i in 0..4 {
                    if moves[i] == 0 {
                        break;
                    }
                    let d = dex.mv(moves[i]);
                    match layer {
                        1 => {
                            if player.status != 0 && d.power == 0 && matches!(d.effect_id, 0x01 | eff::SLEEP | eff::POISON | eff::PARALYZE) {
                                buf[i] += 5;
                            }
                        }
                        2 => {
                            let e = d.effect_id;
                            if self.layer2 == 1 && ((eff::ATTACK_UP1..eff::BIDE).contains(&e) || (eff::ATTACK_UP2..eff::POISON).contains(&e)) {
                                buf[i] -= 1;
                            }
                        }
                        3 => {
                            // The first table entry matching either type wins, with $10 as "neutral".
                            let m = ai
                                .type_effects
                                .iter()
                                .find(|(a, def, _)| *a == d.mtype && (*def == player.types[0] || *def == player.types[1]))
                                .map_or(0x10, |x| x.2);
                            if m > 0x10 {
                                buf[i] -= 1;
                            } else if m < 0x10 {
                                let better = moves.iter().take_while(|&&x| x != 0).any(|&x| {
                                    let o = dex.mv(x);
                                    matches!(o.effect_id, eff::SUPER_FANG | eff::SPECIAL_DAMAGE | eff::FLY) || (o.mtype != d.mtype && o.power != 0)
                                });
                                if better {
                                    buf[i] += 1;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            let min = (0..4).filter(|&i| moves[i] != 0).map(|i| buf[i]).min().unwrap_or(0);
            for i in 0..4 {
                allowed[i] = moves[i] != 0 && buf[i] == min;
            }
        }
        std::array::from_fn(|i| if allowed[i] && disabled_slot != i + 1 { ROLL[i] } else { 0 })
    }

    fn pick_enemy_slot<C: Chance>(&self, dex: &Dex, ch: &mut C) -> u8 {
        let me = &self.b.sides[1];
        if me.moves[1] == 0 && me.disabled != 0 {
            return STRUGGLE_SLOT;
        }
        let w = self.enemy_move_weights(dex);
        if w.iter().all(|&x| x == 0) {
            return STRUGGLE_SLOT;
        }
        ch.pick(&w) as u8
    }

    /// Runs `TrainerAI` at the game's slot in the turn. Returns the action taken, if any.
    fn ai_decide<C: Chance>(&mut self, st: &SpTurn, ch: &mut C) -> Option<AiAction> {
        let ai = self.ai.clone()?;
        if self.ai_count == 0 {
            return None;
        }
        if self.ai_count == 0xFF {
            self.ai_count = ai.max_uses;
        }
        let me = &st.b.sides[1];
        let max = self.teams[1][self.active[1]].mon.max_hp;
        // The routine draws one random byte, so only its breakpoints matter.
        let mut cuts: Vec<u32> = ai.rules.iter().filter_map(|r| r.below).collect();
        cuts.push(256);
        cuts.sort_unstable();
        cuts.dedup();
        let mut lo = 0;
        let mut segs: Vec<(u32, Option<AiAction>)> = Vec::new();
        for &hi in &cuts {
            let fired = ai.rules.iter().find_map(|r| {
                let roll_ok = r.below.is_none_or(|b| lo < b);
                let hp_ok = r.hp_below.is_none_or(|f| (me.hp as u32) < max as u32 / f);
                let status_ok = !r.needs_status || me.status != 0;
                // A failed routine exits at the first failed `ret nc`, which is the same as trying on.
                (roll_ok && hp_ok && status_ok).then_some(r.action)
            });
            if hi > lo {
                segs.push((hi - lo, fired));
            }
            lo = hi;
        }
        let weights: Vec<u32> = segs.iter().map(|s| s.0).collect();
        let k = if weights.len() == 1 { 0 } else { ch.pick(&weights) };
        let mut act = segs[k].1;
        if act == Some(AiAction::Switch) {
            let alive = self.teams[1].iter().enumerate().filter(|(i, t)| *i == self.active[1] || t.hp > 0).count();
            if alive < 2 {
                act = None;
            }
        }
        act
    }

    fn save_active(&mut self, s: usize, b: &Battle) {
        let x = &b.sides[s];
        let t = &mut self.teams[s][self.active[s]];
        t.hp = x.hp;
        t.status = x.status;
    }

    /// `EnemySendOut`: the first party member, other than the current one, with HP left.
    fn enemy_send_out(&mut self, dex: &Dex, b: &mut Battle) {
        let cur = self.active[1];
        let i = (0..self.teams[1].len()).find(|&i| i != cur && self.teams[1][i].hp > 0).expect("a replacement exists");
        let t = &self.teams[1][i];
        send_out_side(dex, b, 1, &t.mon, t.hp, t.status, false);
        self.active[1] = i;
        self.ai_count = 0xFF;
    }

    fn player_send_out(&mut self, dex: &Dex, b: &mut Battle, i: usize) {
        let t = &self.teams[0][i];
        send_out_side(dex, b, 0, &t.mon, t.hp, t.status, false);
        apply_badges(&mut b.sides[0], self.badges);
        self.active[0] = i;
    }

    fn apply_item(&mut self, idx: usize) {
        let it = self.bag[idx];
        self.bag[idx].count -= 1;
        let a = self.active[0];
        let mon = self.teams[0][a].mon.clone();
        let x = &mut self.b.sides[0];
        let cure = |x: &mut Side| {
            // Curing resets battle stats to the party's, dropping badge boosts and stage effects.
            x.status = 0;
            x.bs3 &= !BADLY_POISONED;
            x.stats = mon.stats;
        };
        match it.kind {
            ItemKind::Heal(n) => x.hp = (x.hp + n).min(mon.max_hp),
            ItemKind::FullRestore => {
                if x.hp < mon.max_hp {
                    // Unlike a cure, this path leaves the Toxic flag and battle stats alone.
                    x.hp = mon.max_hp;
                    x.status = 0;
                } else {
                    cure(x);
                }
            }
            ItemKind::Cure(_) => cure(x),
        }
        self.teams[0][a].hp = x.hp;
        self.teams[0][a].status = x.status;
    }

    /// Plays one decision: a turn, or sending out a replacement after a faint.
    pub fn step<C: Chance>(&mut self, dex: &Dex, action: PlayerAction, ch: &mut C) {
        assert!(self.result.is_none(), "battle is over");
        if self.need_replace {
            let PlayerAction::Switch(i) = action else { panic!("must send out a replacement") };
            let mut b = self.b.clone();
            self.player_send_out(dex, &mut b, i as usize);
            if b.sides[1].hp == 0 {
                self.replace_enemy(dex, &mut b);
            }
            self.b = b;
            self.need_replace = false;
            return;
        }
        let mut skip = [false, false];
        let mut pslot = 0u8;
        match action {
            PlayerAction::Move(s) => {
                pslot = s;
                if s != STRUGGLE_SLOT {
                    self.last_slot = s;
                }
            }
            PlayerAction::Switch(i) => {
                let b0 = self.b.clone();
                self.save_active(0, &b0);
                let mut b = self.b.clone();
                self.player_send_out(dex, &mut b, i as usize);
                self.b = b;
                skip[0] = true;
            }
            PlayerAction::Item(i) => {
                self.apply_item(i as usize);
                skip[0] = true;
            }
            PlayerAction::Pass => {}
        }

        let m = self.matchup();
        let mut eng = Engine::new(dex, &m);
        eng.sp = Some(SinglePlayer { badges: self.badges });
        let eslot = if eng.needs_choice(&self.b, 1) { self.pick_enemy_slot(dex, ch) } else { 0 };
        let mut st = eng.sp_begin(&self.b, [pslot, eslot], skip, ch);
        let order = [st.first, 1 - st.first];
        let mut replaced: Option<Matchup> = None;
        let mut out = None;
        for &s in &order {
            let o = if let Some(m2) = replaced.clone() {
                let mut e2 = Engine::new(dex, &m2);
                e2.sp = eng.sp;
                self.slot(dex, &e2, &mut st, s, ch, &mut replaced)
            } else {
                self.slot(dex, &eng, &mut st, s, ch, &mut replaced)
            };
            if let Some(o) = o {
                out = Some(o);
                break;
            }
        }
        let m3 = replaced.unwrap_or_else(|| self.matchup());
        let mut e3 = Engine::new(dex, &m3);
        e3.sp = eng.sp;
        if out.is_none() {
            e3.sp_end(&mut st);
        } else {
            e3.sp_save_ram(&mut st);
        }
        if st.pp_spent[0] {
            let t = &mut self.teams[0][self.active[0]];
            let l = self.last_slot as usize;
            t.pp[l] = t.pp[l].saturating_sub(1);
        }
        let mut b = st.b;
        self.save_active(0, &b);
        self.save_active(1, &b);
        match out {
            None | Some(Outcome::Ongoing) => {}
            Some(o) => self.after_faint(dex, &mut b, o),
        }
        self.b = b;
    }

    /// One side's slot in the turn order, including the game's chance to act instead of moving.
    fn slot<C: Chance>(&mut self, dex: &Dex, eng: &Engine, st: &mut SpTurn, s: usize, ch: &mut C, replaced: &mut Option<Matchup>) -> Option<Outcome> {
        if s == 1 {
            if let Some(a) = self.ai_decide(st, ch) {
                self.apply_ai(dex, a, eng, st, ch, replaced);
                st.skip[1] = true;
                if let Some(m2) = replaced.clone() {
                    let mut e2 = Engine::new(dex, &m2);
                    e2.sp = eng.sp;
                    return e2.sp_act(st, 1, ch);
                }
            } else if st.b.sides[1].selected != CANNOT_MOVE {
                self.layer2 = self.layer2.saturating_add(1).min(2);
            }
        }
        eng.sp_act(st, s, ch)
    }

    fn apply_ai<C: Chance>(&mut self, dex: &Dex, a: AiAction, eng: &Engine, st: &mut SpTurn, ch: &mut C, replaced: &mut Option<Matchup>) {
        let max = self.teams[1][self.active[1]].mon.max_hp;
        match a {
            AiAction::Switch => {
                let b0 = st.b.clone();
                self.save_active(1, &b0);
                let mut b = st.b.clone();
                self.enemy_send_out(dex, &mut b);
                st.b = b;
                *replaced = Some(self.matchup());
                return;
            }
            AiAction::XStat(stat) => eng.sp_ai_stat_up(st, stat, ch),
            AiAction::GuardSpec => st.b.sides[1].bs2 |= PROTECTED_BY_MIST,
            AiAction::FullHeal => {
                let x = &mut st.b.sides[1];
                x.status = 0;
                x.bs3 &= !BADLY_POISONED;
            }
            AiAction::Heal(n) => {
                let x = &mut st.b.sides[1];
                x.hp = (x.hp + n).min(max);
            }
            AiAction::FullRestore => {
                let x = &mut st.b.sides[1];
                x.hp = max;
                x.status = 0;
                x.bs3 &= !BADLY_POISONED;
            }
        }
        self.ai_count -= 1;
    }

    /// `HandleEnemyMonFainted` and `HandlePlayerMonFainted`. Both check the player's party first, so a
    /// double KO on both last Pokémon is a black-out, whoever attacked. If both fainted and the battle goes
    /// on, the player chooses first and the game's replacement follows (`ChooseNextMon`, then
    /// `ReplaceFaintedEnemyMon`), so the fainted foe stays in memory while the player's menu is up.
    fn after_faint(&mut self, dex: &Dex, b: &mut Battle, o: Outcome) {
        let (mine, theirs) = match o {
            Outcome::Win(0) => (false, true),
            Outcome::Win(_) => (true, false),
            Outcome::Draw => (true, true),
            Outcome::Ongoing => return,
        };
        if theirs {
            let t = &mut self.teams[1][self.active[1]];
            t.hp = 0;
            t.status = 0;
            survivor_after_faint(&mut b.sides[0]);
        }
        if mine {
            let t = &mut self.teams[0][self.active[0]];
            t.hp = 0;
            t.status = 0;
            // `RemoveFaintedPlayerMon` zeroes the enemy's Bide damage entirely, and the battle status.
            b.sides[1].bs1 &= !ATTACKING_MULTIPLE_TIMES;
            b.sides[1].bide_dmg = 0;
            b.sides[0].status = 0;
        }
        let enemy_alive = self.teams[1].iter().any(|t| t.hp > 0);
        let player_alive = self.teams[0].iter().any(|t| t.hp > 0);
        self.result = match (enemy_alive, player_alive) {
            (_, false) => Some(Finish::Loss),
            (false, true) => Some(Finish::Win),
            (true, true) => None,
        };
        if self.result.is_some() {
            return;
        }
        if mine {
            self.need_replace = true;
        } else if theirs {
            self.replace_enemy(dex, b);
        }
    }

    /// `ReplaceFaintedEnemyMon`.
    fn replace_enemy(&mut self, dex: &Dex, b: &mut Battle) {
        self.enemy_send_out(dex, b);
        b.sides[1].ram[0] = 0;
        self.layer2 = 0;
    }
}
