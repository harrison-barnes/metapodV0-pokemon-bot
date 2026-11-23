// Plays 6v6 Gen 1 OU battles in Pokemon Showdown with the same random policy as `sim6`:
// with probability P switch to a random healthy team member (when allowed), otherwise pick a random
// usable move; replace fainted Pokemon at random.
// Input (stdin): JSON lines {"id", "a": [{"species", "moves"}, ...], "b": [...], "n", "seed", "p"}
// Output (stdout): JSON lines {"id", "wins": [p1, p2], "draws", "turns"}
'use strict';
const readline = require('readline');
const {Battle} = require('pokemon-showdown/dist/sim/battle');
const {Dex} = require('pokemon-showdown/dist/sim/dex');

const gen1 = Dex.mod('gen1');
const MOVE_RENAMES = {psychicm: 'psychic', hijumpkick: 'highjumpkick', vicegrip: 'visegrip'};
const toId = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');

function setOf(mon) {
  const species = gen1.species.get(toId(mon.species));
  if (!species.exists) throw new Error(`unknown species ${mon.species}`);
  const moves = mon.moves.map(m => {
    const id = MOVE_RENAMES[toId(m)] || toId(m);
    if (!gen1.moves.get(id).exists) throw new Error(`unknown move ${m}`);
    return id;
  });
  return {
    name: species.name, species: species.name, moves, level: 100, ability: 'No Ability', item: '',
    evs: {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252},
    ivs: {hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30},
  };
}

// mulberry32
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function choose(side, rand, p) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  const bench = req.side.pokemon.map((m, i) => ({m, i}))
    .filter(x => !x.m.active && !x.m.condition.endsWith(' fnt'));
  const pickSwitch = () => `switch ${bench[Math.floor(rand() * bench.length)].i + 1}`;
  if (req.forceSwitch) return bench.length ? pickSwitch() : 'pass';
  const active = req.active[0];
  const moves = active.moves.map((m, i) => ({m, i})).filter(x => !x.m.disabled);
  const canSwitch = !active.trapped && !active.maybeTrapped && bench.length > 0;
  if (canSwitch && (moves.length === 0 || rand() < p)) return pickSwitch();
  if (!moves.length) return 'move 1';
  return `move ${moves[Math.floor(rand() * moves.length)].i + 1}`;
}

function playOne(job, seed, rand) {
  const s = [seed & 0xffff, (seed >>> 16) & 0xffff, 0x1234, 0x5678];
  const battle = new Battle({formatid: 'gen1ou', seed: s});
  battle.setPlayer('p1', {name: 'p1', team: job.a.map(setOf)});
  battle.setPlayer('p2', {name: 'p2', team: job.b.map(setOf)});
  while (!battle.ended && battle.turn < 1000) {
    const before = battle.turn;
    let acted = false;
    for (const side of battle.sides) {
      const c = choose(side, rand, job.p);
      if (c !== null) {
        battle.choose(side.id, c);
        acted = true;
      }
    }
    if (!acted && battle.turn === before && !battle.ended) throw new Error('battle stalled');
  }
  const winner = battle.ended ? (battle.winner === 'p1' ? 0 : battle.winner === 'p2' ? 1 : 2) : 2;
  return {winner, turns: battle.turn};
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  if (!line.trim()) return;
  const job = JSON.parse(line);
  const rand = rng(job.seed * 7919 + 1);
  const wins = [0, 0];
  let draws = 0, turns = 0;
  for (let k = 0; k < job.n; k++) {
    const r = playOne(job, job.seed * 100003 + k, rand);
    if (r.winner === 2) draws++; else wins[r.winner]++;
    turns += r.turns;
  }
  process.stdout.write(JSON.stringify({id: job.id, wins, draws, turns: turns / job.n}) + '\n');
});
