// Plays 1v1 Gen 1 battles in Pokemon Showdown with uniformly random move choices.
// Input (stdin): JSON lines {"id", "a": {"species", "moves"}, "b": {...}, "n", "seed"}
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
    name: species.name, species: species.name, moves, level: 50, ability: 'No Ability', item: '',
    evs: {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252},
    ivs: {hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30},
  };
}

// Small deterministic PRNG for move choices (mulberry32).
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

function playOne(a, b, seed, rand) {
  const s = [seed & 0xffff, (seed >>> 16) & 0xffff, 0x1234, 0x5678];
  const battle = new Battle({formatid: 'gen1customgame', seed: s});
  battle.setPlayer('p1', {name: 'p1', team: [setOf(a)]});
  battle.setPlayer('p2', {name: 'p2', team: [setOf(b)]});
  battle.choose('p1', 'team 1');
  battle.choose('p2', 'team 1');
  while (!battle.ended && battle.turn < 1000) {
    const turn = battle.turn;
    const picks = [];
    for (const side of battle.sides) {
      const req = side.activeRequest;
      if (!req || req.wait) continue;
      const moves = req.active[0].moves;
      const legal = moves.map((m, i) => ({m, i})).filter(x => !x.m.disabled);
      const pick = legal.length ? legal[Math.floor(rand() * legal.length)].i + 1 : 1;
      picks.push([side.id, `move ${pick}`]);
    }
    for (const [id, c] of picks) battle.choose(id, c);
    if (battle.turn === turn && !battle.ended) throw new Error('turn did not advance');
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
    const r = playOne(job.a, job.b, job.seed * 100003 + k, rand);
    if (r.winner === 2) draws++; else wins[r.winner]++;
    turns += r.turns;
  }
  process.stdout.write(JSON.stringify({id: job.id, wins, draws, turns: turns / job.n}) + '\n');
});
