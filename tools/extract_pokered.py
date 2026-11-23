"""Extract Gen 1 battle data from the pret/pokered disassembly into data/gen1.json.

Everything the engine needs (species, moves, learnsets, type chart) comes from
the cartridge source so the data matches the ROM exactly.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# a clone of https://github.com/pret/pokered (not included here): POKERED=/path/to/pokered, default vendor/pokered
POKERED = Path(os.environ.get("POKERED", ROOT / "vendor" / "pokered"))


def consts(path, stop=None):
    """Return the `const NAME` entries of a const_def block, in order (value = index)."""
    names = []
    for line in (POKERED / path).read_text().splitlines():
        line = line.split(";")[0].strip()
        if stop and line.startswith(stop):
            break
        if line.startswith("const_skip"):
            names.append(None)  # unused index, keeps later values aligned
            continue
        m = re.match(r"const\s+(\w+)", line)
        if m:
            names.append(m.group(1))
    return names


def type_values():
    """Type constants have gaps (special types start at $14), so track const_next."""
    vals, v = {}, 0
    for line in (POKERED / "constants/type_constants.asm").read_text().splitlines():
        line = line.split(";")[0].strip()
        if line.startswith("const_next"):
            v = int(line.split()[1])
        m = re.match(r"const\s+(\w+)", line)
        if m:
            vals[m.group(1)] = v
            v += 1
    return vals


def main():
    moves_const = consts("constants/move_constants.asm", stop="DEF NUM_ATTACKS")
    assert moves_const[0] == "NO_MOVE" and moves_const[165] == "STRUGGLE", moves_const[160:170]
    moves_const = moves_const[:166]
    effects = consts("constants/move_effect_constants.asm")
    types = type_values()
    type_names = sorted(types, key=types.get)
    type_index = {t: i for i, t in enumerate(type_names)}  # dense 0..N indices for the engine

    moves = [None]
    for line in (POKERED / "data/moves/moves.asm").read_text().splitlines():
        m = re.match(r"\s*move\s+(\w+),\s*(\w+),\s*(\d+),\s*(\w+),\s*(\d+),\s*(\d+)", line)
        if m:
            name, eff, power, typ, acc, pp = m.groups()
            moves.append({
                "id": len(moves),
                "name": name,
                "effect": eff,
                "effect_id": effects.index(eff),
                "power": int(power),
                "type": type_index[typ],
                # The cartridge stores accuracy as `percent` = n * 255 / 100 (rounded down).
                "accuracy": int(acc) * 255 // 100,
                "accuracy_pct": int(acc),
                "pp": int(pp),
            })
    assert [m["name"] for m in moves[1:]] == moves_const[1:], "move table order mismatch"

    species_const = consts("constants/pokemon_constants.asm")
    # Evolutions + learnsets, keyed by internal species constant.
    evo_text = (POKERED / "data/pokemon/evos_moves.asm").read_text()
    ptr_labels = re.findall(r"dw (\w+)EvosMoves", evo_text)
    label_to_species = {lab: species_const[i + 1] for i, lab in enumerate(ptr_labels)}
    evos, levelup = {}, {}
    for block in re.split(r"\n(?=\w+EvosMoves:)", evo_text):
        m = re.match(r"(\w+)EvosMoves:", block)
        if not m or m.group(1) not in label_to_species:
            continue
        sp = label_to_species[m.group(1)]
        evo_part, learn_part = block.split("; Learnset")
        evos[sp] = [l.split(",")[-1].strip() for l in re.findall(r"db EVOLVE_\w+.*", evo_part)]
        levelup[sp] = [(int(lv), mv) for lv, mv in re.findall(r"db (\d+), (\w+)", learn_part)]
    prevo = {child: parent for parent, kids in evos.items() for child in kids}

    species = []
    for f in sorted((POKERED / "data/pokemon/base_stats").glob("*.asm")):
        t = f.read_text()
        dex_name = re.search(r"db DEX_(\w+)", t).group(1)
        stats = [int(x) for x in re.search(r"db\s+(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\n\s*;\s*hp", t).groups()]
        t1, t2 = re.search(r"db (\w+), (\w+) ; type", t).groups()
        lv1 = [x.strip() for x in re.search(r"db (.*) ; level 1 learnset", t).group(1).split(",")]
        tm = re.search(r"tmhm(.*?)\n\s*; end", t, re.S).group(1)
        tmhm = [x for x in re.split(r"[\s,\\]+", tm) if x and x != "UNUSED"]
        species.append({
            "name": dex_name, "file": f.stem,
            "base": dict(zip(["hp", "atk", "def", "spe", "spc"], stats)),
            "types": [type_index[t1], type_index[t2]],
            "lv1": [m for m in lv1 if m != "NO_MOVE"], "tmhm": tmhm,
        })
    by_name = {s["name"]: s for s in species}
    names_text = (POKERED / "data/pokemon/names.asm").read_text()

    for s in species:
        own = set(s["lv1"]) | {mv for _, mv in levelup[s["name"]]} | set(s["tmhm"])
        # Moves learnable by any earlier stage are also legal for the evolved form.
        chain, cur = [], s["name"]
        while cur in prevo:
            cur = prevo[cur]
            chain.append(cur)
        inherited = set()
        for p in chain:
            inherited |= set(by_name[p]["lv1"]) | {mv for _, mv in levelup[p]} | set(by_name[p]["tmhm"])
        learn = sorted(own | inherited, key=moves_const.index)
        s["learnset"] = [moves_const.index(m) for m in learn]
        s["prevos"] = chain
        s["fully_evolved"] = not evos[s["name"]]
        s["levelup"] = levelup[s["name"]]
    for s in species:
        del s["lv1"], s["tmhm"]
    species.sort(key=lambda s: int(re.search(r"const DEX_%s\s*;\s*(\d+)" % s["name"],
                                              (POKERED / "constants/pokedex_constants.asm").read_text()).group(1)))
    for i, s in enumerate(species):
        s["dex"] = i + 1

    chart = []
    for a, d, eff in re.findall(r"db (\w+),\s+(\w+),\s+(\w+)", (POKERED / "data/types/type_matchups.asm").read_text()):
        chart.append([type_index[a], type_index[d], {"SUPER_EFFECTIVE": 20, "NOT_VERY_EFFECTIVE": 5, "NO_EFFECT": 0}[eff]])

    out = {
        "source": "pret/pokered",
        "types": type_names,
        "special_threshold": type_index["FIRE"],  # types >= this index use Special
        "type_chart": chart,  # ordered: the cartridge applies matches in table order
        "effects": effects,
        "moves": moves,
        "species": species,
    }
    dest = ROOT / "data" / "gen1.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}: {len(species)} species, {len(moves) - 1} moves, {len(chart)} type matchups")


if __name__ == "__main__":
    sys.exit(main())
