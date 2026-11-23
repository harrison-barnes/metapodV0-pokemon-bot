// Ground truth for the whole Showdown-to-engine state rebuild (agent/test_adapter_truth.py): Gen 1 OU battles played
// in the Showdown simulator by scripted players. One JSON line per decision point with
//  - msgs: what the server would have sent each player since the last decision, in order (the public / private line of
//    each |split| pair, and that player's |request|), as [player, "update" | "request", text];
//  - truth: Showdown's internal state at that decision point (both sides in full: HP, status and its counters, stages,
//    stored and modified stats, volatiles with their counters, move slots and PP, last moves, types, last damage);
//  - choices: what each player then chose.
// Teams: lines of a JSON file ({"p1": export text, "p2": export text, "tag": ...}, see agent/test_adapter_truth.py
// --write-teams), or with no file "exotic" teams: random Gen 1 OU-legal Pokemon whose moves lean towards the rare
// mechanics (Substitute, Leech Seed, screens, Mist, Focus Energy, Hyper Beam, trapping, Rage, Bide, Thrash, two-turn
// moves, Transform, Disable, Mimic, Conversion, Counter, Mirror Move, Metronome, Haze, Toxic, Rest ...).
//   npm install && node gen1_truth.js --games 200 [--teams teams.jsonl] [--seed 1] [--switch 0.15] > truth.jsonl
'use strict';
const SD = process.env.SD || 'pokemon-showdown';
const {Battle} = require(`${SD}/dist/sim/battle`);
const {Teams} = require(`${SD}/dist/sim/teams`);
const {Dex} = require(`${SD}/dist/sim/dex`);
const {TeamValidator} = require(`${SD}/dist/sim/team-validator`);
const fs = require('fs');

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const N = +(args.games || 100), SEED = +(args.seed || 1), SWITCH = +(args.switch || 0.15), MAX_TURNS = +(args.turns || 250);
const FORMAT = args.format || 'gen1ou';

let a = SEED * 7919 + 12345;
const rand = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const pick = xs => xs[Math.floor(rand() * xs.length)];

// -- exotic teams -------------------------------------------------------------------------------------------------
const dex = Dex.mod('gen1');
const RARE = ['substitute', 'leechseed', 'reflect', 'lightscreen', 'mist', 'focusenergy', 'hyperbeam', 'wrap', 'bind', 'clamp',
  'firespin', 'rage', 'bide', 'thrash', 'petaldance', 'skullbash', 'skyattack', 'solarbeam', 'razorwind', 'transform', 'disable',
  'mimic', 'conversion', 'counter', 'mirrormove', 'metronome', 'haze', 'toxic', 'rest', 'confuseray', 'supersonic', 'amnesia',
  'agility', 'swordsdance', 'growl', 'screech', 'stringshot', 'sandattack', 'smokescreen', 'whirlwind', 'roar', 'teleport', 'superfang',
  'seismictoss', 'nightshade', 'psywave', 'dreameater', 'explosion', 'selfdestruct', 'highjumpkick', 'jumpkick', 'doubleedge',
  'submission', 'takedown', 'recover', 'softboiled', 'thunderwave', 'glare', 'stunspore', 'sleeppowder', 'hypnosis', 'sing',
  'lovelykiss', 'spore', 'poisonpowder', 'megadrain', 'absorb', 'leechlife', 'pinmissile', 'doublekick', 'twineedle', 'furyattack',
  'bonemerang', 'quickattack', 'swift', 'psychic', 'acid', 'bubblebeam', 'aurorabeam', 'fireblast', 'bodyslam', 'blizzard',
  'struggle', 'sonicboom', 'dragonrage', 'harden', 'withdraw', 'defensecurl', 'barrier', 'acidarmor', 'meditate', 'sharpen', 'growth'];
const validator = new TeamValidator(FORMAT);
const SPECIES = dex.species.all().filter(s => s.gen === 1 && s.num > 0 && s.num <= 151 && !s.forme && !['mew', 'mewtwo'].includes(s.id));
function learnset(id) {
  const out = new Set();
  let s = dex.species.get(id);
  while (s && s.exists) {
    const ls = dex.species.getLearnsetData(s.id).learnset || {};
    for (const m in ls) if (ls[m].some(src => src[0] === '1')) out.add(m);
    s = s.prevo ? dex.species.get(s.prevo) : null;
  }
  return [...out];
}
const LEARN = {};
function exoticTeam() {
  for (;;) {
    const team = [], used = new Set();
    while (team.length < 6) {
      const s = pick(SPECIES);
      if (used.has(s.id)) continue;
      used.add(s.id);
      const ls = LEARN[s.id] || (LEARN[s.id] = learnset(s.id).filter(m => !validator.validateSet({species: s.name, moves: [m], evs: {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252}}, {})));
      if (!ls.length) { used.delete(s.id); continue; }
      const rare = ls.filter(m => RARE.includes(m)), moves = [];
      while (moves.length < Math.min(4, ls.length)) {
        const m = (rare.length && rand() < 0.7) ? pick(rare) : pick(ls);
        if (!moves.includes(m)) moves.push(m);
      }
      team.push({name: s.name, species: s.name, moves, evs: {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252},
        ivs: {hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30}, level: 100, ability: 'No Ability', item: '', gender: '', nature: ''});
    }
    if (!validator.validateTeam(team)) return team;
  }
}

// -- scripted players ---------------------------------------------------------------------------------------------
function choice(side, sw) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  const bench = req.side.pokemon.map((p, i) => ({p, i})).filter(({p, i}) => i > 0 && !p.condition.endsWith(' fnt'));
  if (req.forceSwitch) return bench.length ? `switch ${pick(bench).i + 1}` : 'pass';
  const act = req.active[0];
  const moves = act.moves.map((m, i) => ({m, i})).filter(({m}) => !m.disabled);
  if (!act.trapped && bench.length && (rand() < sw || !moves.length)) return `switch ${pick(bench).i + 1}`;
  return moves.length ? `move ${pick(moves).i + 1}` : 'default';
}

// -- truth --------------------------------------------------------------------------------------------------------
const plain = o => { const out = {}; for (const k in o) { const v = o[k]; if (v === null || ['number', 'string', 'boolean'].includes(typeof v)) out[k] = v; } return out; };
function mon(p) {
  const vol = {};
  for (const v in p.volatiles) vol[v] = plain(p.volatiles[v]);
  const src = p.statusState && p.statusState.source;
  return {
    ident: `${p.side.id}: ${p.name}`, species: p.species.id, base_species: p.baseSpecies.id, set_species: Dex.toID(p.set.species), level: p.level,
    hp: p.hp, maxhp: p.maxhp, status: p.status, status_state: plain(p.statusState || {}), slept_by_foe: p.status === 'slp' && !!src && src.side !== p.side,
    fainted: p.fainted, active: p.isActive, position: p.position, boosts: {...p.boosts}, stored: {...p.storedStats}, base_stored: {...p.baseStoredStats},
    modified: p.modifiedStats ? {...p.modifiedStats} : null, types: p.getTypes(), transformed: !!p.transformed,
    moves: p.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled, virtual: !!m.virtual, used: !!m.used})),
    base_moves: p.baseMoveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp, used: !!m.used})),
    last_move: p.lastMove ? p.lastMove.id : null, volatiles: vol, trapped: !!p.trapped, maybe_locked: !!p.maybeLocked,
    move_this_turn: p.moveThisTurn || null,
  };
}
function truth(b) {
  return {
    turn: b.turn, last_damage: b.lastDamage, ended: b.ended, winner: b.winner || null, request_state: b.requestState,
    sides: b.sides.map(s => ({
      id: s.id, last_move: s.lastMove ? s.lastMove.id : null, last_selected_move: s.lastSelectedMove || null,
      last_selected_slot: s.lastSelectedMoveSlot === undefined ? null : s.lastSelectedMoveSlot,
      last_enemy_move: s.lastEnemyMove ? s.lastEnemyMove.id : null, last_enemy_selected_move: s.lastEnemySelectedMove || null,
      pokemon: s.pokemon.map(mon),       // in the side's current order (the active one first)
    })),
  };
}

// -- games --------------------------------------------------------------------------------------------------------
const teamLines = args.teams ? fs.readFileSync(args.teams, 'utf8').split('\n').filter(Boolean).map(JSON.parse) : null;
for (let g = 0; g < N; g++) {
  let t1, t2, tag = 'exotic';
  if (teamLines) {
    const row = teamLines[g % teamLines.length];
    t1 = Teams.import(row.p1); t2 = Teams.import(row.p2); tag = row.tag || 'file';
  } else {
    t1 = exoticTeam(); t2 = exoticTeam();
  }
  const out = [];
  const b = new Battle({formatid: FORMAT, seed: [SEED, g + 1, 8, 9], send: (type, data) => {
    if (type === 'update') {
      for (const p of ['p1', 'p2']) {
        const lines = [];
        for (let i = 0; i < data.length; i++) {
          if (data[i].startsWith('|split|')) { lines.push(data[i + (data[i].slice(7) === p ? 1 : 2)]); i += 2; } else lines.push(data[i]);
        }
        out.push([p, 'update', lines.join('\n')]);
      }
    } else if (type === 'sideupdate') {
      const [p, ...rest] = data.split('\n');
      out.push([p, 'request', rest.join('\n')]);
    }
  }});
  b.setPlayer('p1', {name: 'Truth One', team: t1});
  b.setPlayer('p2', {name: 'Truth Two', team: t2});
  const sw = [SWITCH * (0.3 + 1.4 * rand()), SWITCH * (0.3 + 1.4 * rand())];
  let step = 0;
  for (;;) {
    b.sendUpdates();
    const c = b.ended ? [null, null] : [choice(b.sides[0], sw[0]), choice(b.sides[1], sw[1])];
    console.log(JSON.stringify({game: g, tag, step: step++, msgs: out.splice(0), truth: truth(b), choices: c}));
    if (b.ended) break;
    if (b.turn > MAX_TURNS) { b.forceWin && b.tie(); b.sendUpdates(); console.log(JSON.stringify({game: g, tag, step: step++, msgs: out.splice(0), truth: truth(b), choices: [null, null]})); break; }
    if (c[0]) b.choose('p1', c[0]);
    if (c[1]) b.choose('p2', c[1]);
  }
}
