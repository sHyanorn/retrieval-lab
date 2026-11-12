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


def bm25_result(d, ndcg, recall, sha="0" * 64):
    return {"model": "bm25", "dataset": d, "ndcg_cut_10": ndcg, "recall_100": recall, "run_sha256": sha}


# Known answers of the BM25 report against Pyserini's committed anchors: Quora nDCG@10 0.7886,
# Recall@100 0.9733; NQ 0.3055, 0.7513.
KNOWN = [
    # (dataset, our nDCG@10, our Recall@100, nDCG delta, agrees to three decimals, Recall delta)
    ("quora", "0.7891", "0.9733", "+0.0005", True, "+0.0000"),
    ("quora", "0.7892", "0.9700", "+0.0006", False, "-0.0033"),
    ("nq", "0.3050", "0.7600", "-0.0005", True, "+0.0087"),
    ("nq", "0.3049", "0.7513", "-0.0006", False, "+0.0000"),
]


def known_answers_hold(report):
    a, ra = J.read_anchors(), J.read_recall_anchors()
    for d, ndcg, recall, dl, agrees, rdl in KNOWN:
        (row,) = report([bm25_result(d, ndcg, recall)], a, ra)
        if (row["ndcg_delta"], row["agrees_to_three_decimals"], row["recall_delta"]) != (dl, agrees, rdl):
            return False
    return True


def test_bm25_report_known_answers():
    assert known_answers_hold(J.bm25_report)
    a, ra = J.read_anchors(), J.read_recall_anchors()
    assert ra == {("bm25", "quora"): D("0.9733"), ("bm25", "nq"): D("0.7513")}
    (row,) = J.bm25_report([bm25_result("nq", "0.3200", "0.7513")], a, ra)
    assert row["within_tolerance"] is False and row["ndcg_anchor"] == "0.3055" and row["recall_anchor"] == "0.7513"


def test_bm25_report_broken_variants_fail(monkeypatch):
    # The difference in floating point: 0.7891 - 0.7886 is 0.000500000000000056, so the edge would not agree.
    monkeypatch.setattr(J, "delta", lambda ours, anchor: float(ours) - float(anchor))
    monkeypatch.setattr(J, "agrees3", lambda dl: abs(dl) <= 0.0005)
    assert not known_answers_hold(J.bm25_report)
    monkeypatch.undo()
    # A strict comparison at 0.0005.
    monkeypatch.setattr(J, "agrees3", lambda dl: abs(dl) < J.THREE)
    assert not known_answers_hold(J.bm25_report)
    monkeypatch.undo()
    # The difference taken the other way round.
    monkeypatch.setattr(J, "delta", lambda ours, anchor: anchor - D(ours))
    assert not known_answers_hold(J.bm25_report)


def test_quora_run_sha256_beside_amendment_19():
    a, ra = J.read_anchors(), J.read_recall_anchors()
    same = bm25_result("quora", "0.7886", "0.9733", J.AMENDMENT_19_QUORA_RUN_SHA256)
    (row,) = J.bm25_report([same], a, ra)
    assert row["amendment_19_run_sha256"] == "064e7c4d1eefcbc31230ec7764efc83b1e42511af298f1a35d58e7c1bcb4a0b4"
    assert row["same_as_amendment_19"] is True
    (row,) = J.bm25_report([bm25_result("quora", "0.7886", "0.9733", "f" * 64)], a, ra)
    assert row["same_as_amendment_19"] is False and row["run_sha256"] == "f" * 64
    (row,) = J.bm25_report([bm25_result("nq", "0.3055", "0.7513")], a, ra)
    assert "same_as_amendment_19" not in row and "amendment_19_run_sha256" not in row


def job(tmp_path, anchors=J.ANCHORS):
    s4 = tmp_path / "step4"
    s4.mkdir()
    (s4 / "summary.json").write_text('{"status": "done", "reason": "", "steps": []}')
    (s4 / "vectors.sha256").write_text("")
    return J.Job(J.parse(["--vol", str(tmp_path / "vol"), "--step4", str(s4), "--bin", "x", "--trec-eval", "x",
                          "--anchors", str(anchors)]))


def test_recall_100_is_not_gated(tmp_path):
    # nDCG@10 inside on both, Recall@100 far from Pyserini's: done, exit 0, both inside check 3.
    j = job(tmp_path)
    res = [dict(bm25_result("quora", "0.7886", "0.5000"), perquery_sha256="a" * 64),
           dict(bm25_result("nq", "0.3055", "0.9999"), perquery_sha256="b" * 64)]
    assert j.finish("done", "", res) == 0
    out = tmp_path / "vol" / "step5" / "out"
    rows = [ln.split("\t") for ln in (out / "check3.tsv").read_text().splitlines()[1:]]
    assert [(r[0], r[1], r[7]) for r in rows] == [("bm25", "nq", "inside"), ("bm25", "quora", "inside")]
    s = __import__("json").loads((out / "summary.json").read_text())
    assert {r["dataset"]: r["recall_delta"] for r in s["bm25"]} == {"nq": "+0.2486", "quora": "-0.4733"}
    assert s["check3"]["investigate"] == []


def test_a_six_column_anchors_file_stops_the_job_before_any_work(tmp_path):
    six = tmp_path / "anchors6.tsv"
    six.write_text("".join("\t".join(ln.split("\t")[:6]) + "\n" for ln in open(J.ANCHORS, encoding="utf-8").read().splitlines()))
    try:
        job(tmp_path, six)
    except ValueError:
        return
    raise AssertionError("a six-column anchors file was accepted")
