import json
import os
import struct

import numpy as np
import pytest

import encode as E

HERE = os.path.dirname(os.path.abspath(__file__))
BEIR = os.path.join(HERE, "testdata", "beir")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def test_models_as_registered():
    # PREREGISTRATION.md, Section 5: dimension, pooling, prompts, length cap.
    R = "Represent this sentence for searching relevant passages: "
    want = {
        "BAAI/bge-small-en-v1.5": (384, "cls", R, "", 512),
        "BAAI/bge-base-en-v1.5": (768, "cls", R, "", 512),
        "BAAI/bge-large-en-v1.5": (1024, "cls", R, "", 512),
        "intfloat/e5-base-v2": (768, "mean", "query: ", "passage: ", 512),
        "thenlper/gte-base": (768, "mean", "", "", 512),
        MINILM: (384, "mean", "", "", 256),
        "Snowflake/snowflake-arctic-embed-m-v1.5": (768, "cls", R, "", 512),
        "mixedbread-ai/mxbai-embed-large-v1": (1024, "cls", R, "", 512),
    }
    got = {k: (m.dim, m.pooling, m.query_prompt, m.doc_prompt, m.max_len) for k, m in E.MODELS.items()}
    assert got == want
    assert R.endswith(": ")
    assert E.MODELS["BAAI/bge-base-en-v1.5"].revision == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def test_doc_text_as_go():
    # The same cases are checked against DocText in internal/dense.
    with open(os.path.join(HERE, "testdata", "doctext.json"), encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        assert E.doc_text(c["title"], c["text"]) == c["want"], c


def test_read_corpus_order_and_limit():
    ids, texts = E.read_corpus(BEIR)
    assert ids == ["d1", "d2", "d3", "d4", "d5"]
    assert texts[1] == " Exact search scores every document."
    assert texts[2].endswith("rank).")
    assert E.read_corpus(BEIR, limit=2)[0] == ["d1", "d2"]


def test_test_queries_in_file_order():
    ids, texts = E.read_test_queries(BEIR)
    assert ids == ["q1", "q3", "q4"]
    assert texts[1] == "what does fusion add"


def test_test_query_missing(tmp_path):
    for f in ("corpus.jsonl", "queries.jsonl"):
        (tmp_path / f).write_bytes(open(os.path.join(BEIR, f), "rb").read())
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\r\nq9\td1\t1\r\n")
    with pytest.raises(ValueError):
        E.read_test_queries(str(tmp_path))


def test_faiss_flat_layout(tmp_path):
    v = np.arange(6, dtype=np.float32).reshape(2, 3)
    E.write_faiss_flat(str(tmp_path), ["a", "b"], v)
    raw = (tmp_path / "index").read_bytes()
    assert len(raw) == 45 + 4 * 6 and raw[:4] == b"IxFI"
    dim, n = struct.unpack_from("<iq", raw, 4)
    count = struct.unpack_from("<Q", raw, 37)[0]
    assert (dim, n, count, raw[32], struct.unpack_from("<I", raw, 33)[0]) == (3, 2, 6, 1, 0)
    assert np.array_equal(np.frombuffer(raw[45:], dtype="<f4").reshape(2, 3), v)
    assert (tmp_path / "docid").read_text() == "a\nb\n"


def fake(dim):
    # A deterministic encoder whose output depends on a text and its batch.
    def encode(texts):
        return np.array([[len(t) + i + len(texts) * 0.5] * dim for i, t in enumerate(texts)], dtype=np.float32)

    return encode


def test_resume_matches_uninterrupted(tmp_path):
    texts = [f"text {i}" * (i + 1) for i in range(10)]
    whole, kept = E.encode_resumable(str(tmp_path / "a.part"), texts, fake(4), 4, 3)
    assert kept == 0

    part = str(tmp_path / "b.part")
    calls = []

    def failing(ts):
        calls.append(len(ts))
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return fake(4)(ts)

    with pytest.raises(RuntimeError):
        E.encode_resumable(part, texts, failing, 4, 3)
    # Two whole chunks, then a vector cut short.
    with open(part, "ab") as f:
        f.write(np.ones(6, dtype="<f4").tobytes())
    again, kept = E.encode_resumable(part, texts, fake(4), 4, 3)
    assert kept == 6
    assert np.array_equal(again, whole)
    assert os.path.getsize(part) == 10 * 4 * 4


def test_resume_cuts_to_whole_chunks(tmp_path):
    # A part file of 4 vectors with chunks of 3 keeps 3: the fourth was
    # encoded in a batch that the rerun's chunk would not reproduce.
    texts = [f"t{i}" for i in range(7)]
    part = str(tmp_path / "c.part")
    with open(part, "wb") as f:
        f.write(np.full((4, 2), 9, dtype="<f4").tobytes())
    out, kept = E.encode_resumable(part, texts, fake(2), 2, 3)
    assert kept == 3
    assert np.array_equal(out[3:], np.concatenate([fake(2)(texts[3:6]), fake(2)(texts[6:])]))


def test_settings_recorded(tmp_path):
    path = str(tmp_path / "s.json")
    E.settings_check(path, {"batch": 32})
    E.settings_check(path, {"batch": 32})
    with pytest.raises(ValueError):
        E.settings_check(path, {"batch": 16})


@pytest.fixture(scope="module")
def minilm():
    return E.load(MINILM, "cpu")


def test_torch_settings(minilm):
    import torch

    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    assert {p.dtype for p in minilm.parameters()} == {torch.float32}


def test_minilm_end_to_end(tmp_path, minilm):
    out = str(tmp_path / "out")
    s = E.run(MINILM, BEIR, out, batch=2, load_model=lambda n, d: minilm)
    assert (s["docs"], s["queries"], s["resumed_after"]) == (5, 3, 0)
    assert not os.path.exists(os.path.join(out, "vectors.part"))
    ids, vecs = read(out)
    qids, qvecs = read(os.path.join(out, "queries"))
    assert ids == ["d1", "d2", "d3", "d4", "d5"] and qids == ["q1", "q3", "q4"]
    assert vecs.shape == (5, 384) and qvecs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-6)
    direct = minilm.encode(E.read_corpus(BEIR)[1], batch_size=2, normalize_embeddings=True)
    assert np.array_equal(vecs, direct)


def test_minilm_resume_bit_identical(tmp_path, minilm):
    whole = str(tmp_path / "whole")
    E.run(MINILM, BEIR, whole, chunk=2, batch=2, load_model=lambda n, d: minilm)
    out = str(tmp_path / "cut")
    calls = []
    real = minilm.encode

    class Stop(Exception):
        pass

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise Stop
        return real(*a, **k)

    minilm.encode = flaky
    try:
        with pytest.raises(Stop):
            E.run(MINILM, BEIR, out, chunk=2, batch=2, load_model=lambda n, d: minilm)
    finally:
        minilm.encode = real
    assert os.path.getsize(os.path.join(out, "vectors.part")) == 2 * 384 * 4
    s = E.run(MINILM, BEIR, out, chunk=2, batch=2, load_model=lambda n, d: minilm)
    assert s["resumed_after"] == 2
    for sub in ("", "queries"):
        for f in ("index", "docid"):
            assert open(os.path.join(out, sub, f), "rb").read() == open(os.path.join(whole, sub, f), "rb").read()


def test_prompts_applied(tmp_path, minilm, monkeypatch):
    m = E.MODELS[MINILM]
    monkeypatch.setitem(E.MODELS, MINILM, E.Model(m.revision, m.dim, m.pooling, "query: ", "passage: ", m.max_len))
    out = str(tmp_path / "p")
    E.run(MINILM, BEIR, out, batch=2, load_model=lambda n, d: minilm)
    _, vecs = read(out)
    _, qvecs = read(os.path.join(out, "queries"))
    docs = E.read_corpus(BEIR)[1]
    queries = E.read_test_queries(BEIR)[1]
    assert np.array_equal(vecs, minilm.encode(["passage: " + t for t in docs], batch_size=2, normalize_embeddings=True))
    assert np.array_equal(qvecs, minilm.encode(["query: " + t for t in queries], batch_size=2, normalize_embeddings=True))
    assert not np.allclose(qvecs, minilm.encode(queries, batch_size=2, normalize_embeddings=True))


def read(d):
    raw = open(os.path.join(d, "index"), "rb").read()
    dim, n = struct.unpack_from("<iq", raw, 4)
    ids = open(os.path.join(d, "docid"), encoding="utf-8").read().split("\n")[:-1]
    return ids, np.frombuffer(raw[45:], dtype="<f4").reshape(n, dim)


def test_go_fixture_is_current(tmp_path):
    # internal/dense reads this fixture and writes it back byte for byte.
    v = np.array([[0.5, -0.25, 1e-30], [-0.0, 3.4e38, 0.1]], dtype=np.float32)
    E.write_faiss_flat(str(tmp_path), ["doc-1", "doc 2"], v)
    fixture = os.path.join(HERE, "..", "internal", "dense", "testdata", "python-flat")
    for f in ("index", "docid"):
        assert (tmp_path / f).read_bytes() == open(os.path.join(fixture, f), "rb").read()
