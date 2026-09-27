import os
import subprocess
import sys

import pytest

from distfuzz.repro import extract
from distfuzz.repro.report import parse

pytestmark = pytest.mark.multirank

PROGRAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "programs")


@pytest.mark.parametrize(
    "program,title",
    [
        ("collectives/gloo-strided-allreduce.json", "c10d/gloo: wrong result in all_reduce"),
        ("dtensor/dtensor-fill-partial-fuzzprog.json", "DTensor: wrong result in fill_"),
    ],
)
def test_generated_repro_reproduces(tmp_path, program, title):
    f = extract.load(os.path.join(PROGRAMS, program))
    path = tmp_path / "repro.py"
    path.write_text(f.script)
    p = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=180, cwd=tmp_path)
    rep = parse(p.stdout + p.stderr, p.returncode, False, world=f.world, script=f.script)
    assert rep.title == title, (p.stdout + p.stderr)[-2000:]
