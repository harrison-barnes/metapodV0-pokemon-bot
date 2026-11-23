import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.chdir(ROOT)


def run_module(*args, timeout=1800):
    """Run `python -m ...` from the repository root, CPU only; return stdout+stderr, fail on a non-zero exit."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-4000:]
    return out


@pytest.fixture
def run():
    return run_module
