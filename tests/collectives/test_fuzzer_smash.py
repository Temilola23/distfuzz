"""Scheduling of guided and random picks: smashing new programs, divergence in generated ones."""

import pytest

from distfuzz.collectives.fuzzer import Fuzzer
from distfuzz.collectives.prog import has_divergence


def T():
    return {"dtype": "float32", "shape": [2], "layout": "contig", "seed": 1, "kind": "int"}


def C(op, args, rets=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": {}}


TENSOR = C("tensor", {"spec": T()}, {"out": "t1"})
AR = {"world": 4, "calls": [TENSOR, C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False})]}
BC = {
    "world": 4,
    "calls": [
        TENSOR,
        C("broadcast", {"t": "t1", "root": 0, "root_mode": "global", "group": "world", "async_op": False}),
    ],
}
EDGES = {"all_reduce": {100}, "broadcast": {200}}
GENERATED = {"world": 4, "calls": [TENSOR]}


class FakeSession:
    restarts = 0

    def run(self, prog):
        cov = set().union(*(EDGES.get(c["op"], set()) for c in prog["calls"]))
        rank = {"outputs": {}, "lists": {}, "exc": None, "wait_exc": []}
        return {"kind": "ok", "results": [dict(rank, cov=cov)] + [dict(rank) for _ in range(3)]}


def fuzzer(tmp_path, smash, mode="guided"):
    """Ordinary picks always generate (gen_prob=1), so any mutation must come from the smash queue."""
    fz = Fuzzer(str(tmp_path), mode=mode, signal="cxx", smash=smash, gen_prob=1.0, gen_diverge=0.0)
    fz.sess, fz.t0 = FakeSession(), 0.0
    fz.mutated = []
    fz.mut.mutate = lambda p, corpus=(): fz.mutated.append(p) or p
    fz.gen.generate = lambda: GENERATED
    return fz


def admit(fz, prog):
    pick, fz.pick = fz.pick, lambda: prog
    fz.step()
    fz.pick = pick


def test_new_program_is_mutated_smash_times_then_ordinary_picks_resume(tmp_path):
    fz = fuzzer(tmp_path, smash=3)
    admit(fz, AR)
    assert len(fz.corpus) == 1
    picks = [fz.pick() for _ in range(4)]
    assert fz.mutated == [fz.corpus[0]] * 3
    assert picks[3] is GENERATED and fz.stats["smashed"] == 3


def test_programs_admitted_while_smashing_wait_their_turn(tmp_path):
    fz = fuzzer(tmp_path, smash=3)
    admit(fz, AR)
    fz.pick()
    admit(fz, BC)
    for _ in range(5):
        fz.pick()
    assert fz.mutated == [fz.corpus[0]] * 3 + [fz.corpus[1]] * 3


def test_no_smash_by_default(tmp_path):
    fz = fuzzer(tmp_path, smash=0)
    admit(fz, AR)
    assert fz.pick() is GENERATED and fz.mutated == []


def test_random_mode_never_smashes(tmp_path):
    fz = fuzzer(tmp_path, smash=5, mode="random")
    admit(fz, AR)
    assert fz.corpus == [] and fz.pick() is GENERATED and fz.mutated == []


@pytest.mark.parametrize("chance,share", [(0.0, 0.0), (1.0, 1.0)])
def test_gen_diverge_sets_the_share_of_divergent_generated_programs(tmp_path, chance, share):
    fz = Fuzzer(str(tmp_path), mode="random", gen_diverge=chance)
    progs = [fz.pick() for _ in range(200)]
    assert sum(has_divergence(p) for p in progs) / len(progs) == share


def test_gen_diverge_half(tmp_path):
    fz = Fuzzer(str(tmp_path), mode="random", gen_diverge=0.5, seed=3)
    n = sum(has_divergence(fz.pick()) for _ in range(400))
    assert 150 < n < 250


def test_diverge_gives_up_on_a_program_that_cannot_diverge(tmp_path):
    fz = Fuzzer(str(tmp_path))
    assert fz.mut.diverge({"world": 4, "calls": []}) == {"world": 4, "calls": []}
