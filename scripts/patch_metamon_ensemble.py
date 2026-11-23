"""Make Metamon's TaurosEnsemble loadable: one of its members (V2AGroupedV2ISFilter) has a checkpoint saved before
Metamon renamed its perceiver feed-forward layers (nn.Sequential "x_ff.0/.2" -> "x_ff1/x_ff2", same weights), so loading
fails with 56 missing keys. This edits YOUR metamon clone's metamon/rl/pretrained.py to rename those keys on load, only
where the new name exists in the model and the old one doesn't; the loader's strict checks still run afterwards.

    python scripts/patch_metamon_ensemble.py PATH/TO/metamon
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "metamon/rl/pretrained.py"
s = p.read_text()
old = '            ckpt_state = torch.load(ckpt_path, map_location="cpu")\n            model_state = experiment.policy.state_dict()\n'
if "_ren = lambda" in s:
    sys.exit("already patched")
if old not in s:
    sys.exit("patch target not found: this metamon version differs from the one we used (commit 0a00a759)")
new = old + (
    '            import re as _re\n'
    '            _ren = lambda k: _re.sub(r"\\.(cross|self)_ff\\.([02])\\.", lambda m: f".{m.group(1)}_ff{1 if m.group(2) == \'0\' else 2}.", k)\n'
    '            ckpt_state = {(_ren(k) if _ren(k) in model_state and k not in model_state else k): v for k, v in ckpt_state.items()}\n')
p.write_text(s.replace(old, new))
print("patched", p)
