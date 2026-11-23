// Ground truth for the Showdown adapter's Gen 1 stat-penalty tracking (agent/test_stat_drops.py): random Gen 1 random
// battles in Showdown; after every turn, the protocol lines of that turn and each active Pokemon's true Attack and
// Speed (Showdown's modifiedStats), with what they derive from (stored stat, stage, status).
//   npm install && node gen1_stat_drops.js [games=200] > drops.jsonl   (SD=path: another Showdown checkout, built)
'use strict';
const SD = process.env.SD || 'pokemon-showdown';
const {Battle} = require(`${SD}/dist/sim/battle`);
const {Teams} = require(`${SD}/dist/sim/teams`);
let a = 12345;
const rand = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
function choice(side) {
  const req = side.activeRequest;
  if (!req || req.wait) return '';
  const bench = req.side.pokemon.map((p, i) => ({p, i})).filter(({p, i}) => i > 0 && !p.condition.endsWith(' fnt'));
  if (req.forceSwitch) return bench.length ? `switch ${bench[Math.floor(rand() * bench.length)].i + 1}` : 'pass';
  const opts = [];
  req.active[0].moves.forEach((m, i) => { if (!m.disabled) for (let k = 0; k < 4; k++) opts.push(`move ${i + 1}`); });
  if (!req.active[0].trapped) for (const {i} of bench) opts.push(`switch ${i + 1}`);
  return opts.length ? opts[Math.floor(rand() * opts.length)] : 'default';
}
// the public line of each |split| pair
function lines(log) {
  const out = [];
  for (let i = 0; i < log.length; i++) {
    if (log[i].startsWith('|split|')) { out.push(log[i + 2]); i += 2; } else out.push(log[i]);
  }
  return out;
}
const N = +(process.argv[2] || 200);
for (let g = 0; g < N; g++) {
  const b = new Battle({formatid: 'gen1customgame', seed: [g + 1, 7, 8, 9]});
  b.setPlayer('p1', {team: Teams.generate('gen1randombattle', {seed: [g + 1, 1, 2, 3]})});
  b.setPlayer('p2', {team: Teams.generate('gen1randombattle', {seed: [g + 1, 4, 5, 6]})});
  let pos = 0, guard = 0;
  const emit = () => {
    const actives = b.sides.map(s => s.active[0]).filter(p => p && !p.fainted).map(p => ({
      key: `${p.side.id}: ${p.name}`, stored: [p.storedStats.atk, p.storedStats.spe], boosts: [p.boosts.atk, p.boosts.spe],
      mod: [p.modifiedStats.atk, p.modifiedStats.spe], status: p.status, transformed: !!p.transformed}));
    console.log(JSON.stringify({game: g, turn: b.turn, lines: lines(b.log.slice(pos)), actives}));
    pos = b.log.length;
  };
  if (b.requestState === 'teampreview') b.makeChoices('default', 'default');
  emit();
  while (!b.ended && guard++ < 400) {
    b.makeChoices(choice(b.sides[0]), choice(b.sides[1]));
    emit();
  }
}
