import decimal
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import job as J  # noqa: E402

D = decimal.Decimal


def anchors(*rows):
    out = {}
    for m, d, source, value, weak in rows:
        out.setdefault((m, d), []).append((source, D(value), weak))
    return out


def classes(rows):
    return {(r[0], r[1], r[2]): r[7] for r in rows}


def test_delta_is_exact_at_the_tolerance():
    # 0.7380 against 0.74300 is 0.005 exactly; against 0.74301, outside.
    assert J.delta("0.7380", D("0.74300")) == D("-0.005")
    assert abs(J.delta("0.7380", D("0.74300"))) <= J.TOL
    assert abs(J.delta("0.7380", D("0.74301"))) > J.TOL
    # A float would put 0.7380 - 0.7430 above 0.005.
    assert abs(0.7380 - 0.7430) > 0.005


def test_one_weak_deviation_alone_is_an_exception():
    a = anchors(("m", "fiqa", "mteb", "0.40", True), ("m", "nq", "mteb", "0.50", False))
    got = classes(J.classify({("m", "fiqa"): "0.4100", ("m", "nq"): "0.5000"}, a))
    assert got == {("m", "fiqa", "mteb"): "exception", ("m", "nq", "mteb"): "inside"}


def test_a_second_deviation_or_a_strong_one_is_to_investigate():
    a = anchors(("m", "fiqa", "mteb", "0.40", True), ("m", "nq", "mteb", "0.50", True))
    got = classes(J.classify({("m", "fiqa"): "0.4100", ("m", "nq"): "0.4900"}, a))
    assert set(got.values()) == {"investigate"}
    a = anchors(("m", "nq", "mteb", "0.50", False))
    assert classes(J.classify({("m", "nq"): "0.5100"}, a)) == {("m", "nq", "mteb"): "investigate"}


def test_bge_base_two_anchors_each_by_its_own_kind():
    m = "BAAI/bge-base-en-v1.5"
    a = anchors((m, "fiqa", "mteb", "0.40646", True), (m, "fiqa", "pyserini", "0.4065", False))
    # Outside the weak MTEB file only: an exception.
    got = classes(J.classify({(m, "fiqa"): "0.4000"}, anchors((m, "fiqa", "mteb", "0.39000", True), (m, "fiqa", "pyserini", "0.4000", False))))
    assert got == {(m, "fiqa", "mteb"): "exception", (m, "fiqa", "pyserini"): "inside"}
    # Outside Pyserini's: to investigate, and so is the MTEB miss beside it.
    got = classes(J.classify({(m, "fiqa"): "0.4200"}, a))
    assert got == {(m, "fiqa", "mteb"): "investigate", (m, "fiqa", "pyserini"): "investigate"}


def test_an_exception_waits_for_the_models_other_runs():
    a = anchors(("m", "fiqa", "mteb", "0.40", True), ("m", "nq", "mteb", "0.50", False))
    got = classes(J.classify({("m", "fiqa"): "0.4100"}, a, unrun={"m"}))
    assert got == {("m", "fiqa", "mteb"): "pending"}


def test_bm25_is_never_an_exception():
    a = anchors(("bm25", "nq", "pyserini", "0.3055", False))
    assert classes(J.classify({("bm25", "nq"): "0.3200"}, a)) == {("bm25", "nq", "pyserini"): "investigate"}


def test_committed_anchors_cover_every_cell():
    a = J.read_anchors()
    models = {m for m, _ in a if m != "bm25"}
    assert len(models) == 8
    for m in models:
        for d in J.DATASETS:
            assert any(s == "mteb" for s, _, _ in a[m, d]), (m, d)
    assert all(any(s == "pyserini" for s, _, _ in a["BAAI/bge-base-en-v1.5", d]) for d in J.DATASETS)
    assert [s for s, _, _ in a["bm25", "quora"]] == ["pyserini"] and ("bm25", "nq") in a
