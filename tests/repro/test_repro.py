import ast
import json
import os

from distfuzz.repro import extract, gen_distfuzz, gen_dtensor, pipeline
from distfuzz.repro.report import normalize_text, op_of, parse

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "..", "examples")
FILL = os.path.join(EXAMPLES, "programs", "dtensor", "dtensor-fill-partial-fuzzprog.json")
NORM = os.path.join(EXAMPLES, "programs", "dtensor", "dtensor-inf-norm-fuzzprog.json")
GLOO = os.path.join(EXAMPLES, "programs", "collectives", "gloo-strided-allreduce.json")


def fx(name):
    return open(os.path.join(HERE, "fixtures", name)).read()


def test_op_of():
    assert op_of("v7.fill_(3)") == "fill_"
    assert op_of("torch.linalg.vector_norm(v1, ord=inf)") == "vector_norm"
    assert op_of("sharded[6] = 100") == "__setitem__"
    assert op_of("SETITEM(v10, 0, 5)") == "__setitem__"
    assert op_of("mse_loss [Replicate()]x[Shard(2)]") == "mse_loss"
    assert op_of("scatter(out)") == "scatter"
    assert op_of("partial_sum.fill_(3)") == "fill_"
    assert op_of("F.mse_loss(v1, v2)") == "mse_loss"


def test_normalize_text():
    assert normalize_text("[rank 3] at 0xdeadbeef /usr/x/y.py line 42") == "[rank] at ADDR FILE line NUM"


def test_setitem_wrong_result():
    r = parse(fx("setitem_case.txt"), 0, False, 4, "DTensor")
    assert r.title == "DTensor: wrong result in __setitem__"
    assert any("ref" in e for e in r.evidence)


def test_hang_title_from_faulthandler():
    r = parse(fx("flatten_case.txt"), 1, False, 4, "DTensor")
    assert r.title == "DTensor: hang in view_groups" and r.kind == "HANG"


def test_rank_divergent():
    r = parse(fx("max_case.txt"), 1, False, 4, "DTensor")
    assert r.title == "DTensor: rank-divergent RuntimeError in max"
    r = parse(fx("min_json.txt"), 1, False, 4, "DTensor")
    assert r.title == "DTensor: rank-divergent RuntimeError in min"


def test_exception_title_same_across_fuzzers():
    # hand-reduced case and dtensor-fuzz program both land on the same title -> dedup
    a = parse(fx("emb_case.txt"), 1, False, 4, "DTensor").title
    assert a.startswith("DTensor: AssertionError in ")


def test_gloo_titles_cross_fuzzer():
    a = parse(fx("sweep.txt"), 1, False, 4, "gloo")
    b = parse(fx("gloo_json.txt"), 0, False, 4, "gloo")
    assert a.title == b.title == "c10d/gloo: wrong result in all_reduce"
    assert "c10d/gloo: out-of-bounds write in all_reduce" in a.alt_titles
    assert "c10d/gloo: wrong result in scatter" in a.alt_titles
    assert "c10d/gloo: wrong result in all_gather" not in a.titles()  # all_gather is OK (fixed)


def test_other_failure_is_not_target():
    # torch 2.4.1: setitem fails differently (no sharding strategy) -> a different title, not the bug
    r = parse(fx("setitem_case_241.txt"), 1, False, 4, "DTensor")
    assert r.title and "wrong result" not in r.title and r.setup_error is None


def test_setup_error():
    out = (
        "Traceback (most recent call last):\n"
        '  File "r.py", line 1, in <module>\n'
        "ImportError: cannot import name 'Partial'\n"
    )
    r = parse(out, 1, False, 4, "DTensor")
    assert r.setup_error and r.title is None


def _no_fuzzer_imports(src):
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    return mods <= {"torch", "datetime", "faulthandler", "math", "os", "socket", "sys", "time", "traceback"}


def test_dtensor_codegen_standalone():
    f = extract.load(FILL)
    assert f.fuzzer == "dtensor" and _no_fuzzer_imports(f.script)
    compile(f.script, "repro.py", "exec")
    assert "v11 = v7.fill_(3)" in f.script


def test_dtensor_codegen_partial_inputs():
    f = extract.load(NORM)
    compile(f.script, "repro.py", "exec")
    assert _no_fuzzer_imports(f.script)


def test_distfuzz_codegen_standalone():
    f = extract.load(GLOO)
    assert f.fuzzer == "collectives" and _no_fuzzer_imports(f.script)
    compile(f.script, "repro.py", "exec")
    assert "'noncontig'" in f.script and "EXPECTED = {" in f.script


def test_distfuzz_minimize_units():
    rec = json.load(open(GLOO))
    small = gen_distfuzz.with_units(rec, {3, 4})
    assert [c["op"] for c in small["prog"]["calls"]] == ["tensor", "all_reduce"]
    compile(gen_distfuzz.generate(small), "r.py", "exec")


def test_dtensor_is_valid():
    rec = json.load(open(FILL))
    assert gen_dtensor.is_valid(rec)
    assert not gen_dtensor.is_valid(gen_dtensor.with_units(rec, {1, 2, 3}))  # v1 undefined


def test_case_extraction():
    f = extract.load(os.path.join(EXAMPLES, "repros", "dtensor", "cases.py") + "::setitem_shard")
    compile(f.script, "repro.py", "exec")
    assert "def setitem_shard" in f.script and "def fill_partial" not in f.script
    assert "except ImportError" in f.script


def test_launcher_detection():
    f = extract.load(os.path.join(EXAMPLES, "repros", "collectives", "gloo_strided_sweep.py"))
    assert f.launcher[0] == "torchrun" and "--nproc-per-node=4" in f.launcher


def test_ddmin():
    assert pipeline.ddmin(list(range(10)), lambda s: 3 in s and 7 in s) == [3, 7]


def test_decide_matches_pkg_bisect():
    assert pipeline.decide(4, 4, 0) == "bad"
    assert pipeline.decide(4, 0, 4) == "good"
    assert pipeline.decide(4, 1, 3) == "skip"  # one hit in 4 is not enough (wantBad=2)
    assert pipeline.decide(4, 0, 0) == "skip"  # only unrelated failures
    assert pipeline.decide(20, 3, 17) == "bad"


def test_summarize():
    res = [
        dict(version=v, verdict=d)
        for v, d in [("2.4.1", "untestable"), ("2.5.1", "good"), ("2.6.0", "bad"), ("2.7.1", "bad"), ("2.8.0", "good")]
    ]
    s = pipeline.summarize(res)
    assert s["first_bad"] == "2.6.0" and s["last_good_before"] == "2.5.1" and s["fixed_in"] == "2.8.0"


def test_reliability_classes():
    assert pipeline.classify_rate(20, 20) == "deterministic"
    assert pipeline.classify_rate(7, 20) == "flaky"
    assert pipeline.classify_rate(0, 20) == "not reproducible"


def test_summarize_other_failure_before_first_bad():
    res = [dict(version=v, verdict=d) for v, d in [("2.4.1", "skip"), ("2.8.0", "bad"), ("nightly", "bad")]]
    s = pipeline.summarize(res)
    assert s["first_bad"] == "2.8.0" and s["last_good_before"] is None and "different error" in s["note"]


def test_interleaved_dump_is_corrupted():
    out = (
        "Timeout (0:00:20)!\nThread 0xTimeout (0:00:20)!\n"
        "0000ffff Thread 0x (most recent call first):\n"
        'Thread 0x  File 0000ff" (most recent\n'
    )
    r = parse(out, 1, False, 4, "DTensor")
    assert r.corrupted and "corrupted" in r.title


def test_partial_input_not_broadcast():
    f = extract.load(NORM)
    assert "local = PIECES[i][idx]" in f.script and "distribute_tensor(local_full" not in f.script
