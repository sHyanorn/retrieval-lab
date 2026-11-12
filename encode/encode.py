"""Encode a BEIR dataset with one of the study's models (PREREGISTRATION.md, Section 5).

The document vectors go to OUT and the test queries' vectors to OUT/queries,
each as a faiss flat index directory (index and docid), the layout cmd/embed
writes and cmd/dense reads. While the documents are encoded, their vectors are
appended to OUT/vectors.part every --chunk documents; a run that finds that
file continues after the last whole chunk it holds, with the settings recorded
beside it in OUT/vectors.part.json, so that a resumed run encodes the same
batches as an uninterrupted one. A JSON summary goes to standard output.

usage: python encode.py --model ID --data DIR --out DIR [--limit N] [--device cpu|cuda]
"""

import argparse
import dataclasses
import json
import os
import platform
import struct
import sys
import time
import unicodedata

import numpy as np

R = "Represent this sentence for searching relevant passages: "


@dataclasses.dataclass(frozen=True)
class Model:
    revision: str
    dim: int
    pooling: str  # "cls" or "mean"
    query_prompt: str
    doc_prompt: str
    max_len: int


# Section 5's table: revisions from embeddings-benchmark/results at 62d67f4,
# prompts from MTEB's model metadata at tag 2.27.8 (652cacc).
MODELS = {
    "BAAI/bge-small-en-v1.5": Model("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", 384, "cls", R, "", 512),
    "BAAI/bge-base-en-v1.5": Model("a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", 768, "cls", R, "", 512),
    "BAAI/bge-large-en-v1.5": Model("d4aa6901d3a41ba39fb536a557fa166f842b0e09", 1024, "cls", R, "", 512),
    "intfloat/e5-base-v2": Model("1c644c92ad3ba1efdad3f1451a637716616a20e8", 768, "mean", "query: ", "passage: ", 512),
    "thenlper/gte-base": Model("c078288308d8dee004ab72c6191778064285ec0c", 768, "mean", "", "", 512),
    "sentence-transformers/all-MiniLM-L6-v2": Model("8b3219a92973c328a8e22fadcfa821b5dc75636a", 384, "mean", "", "", 256),
    "Snowflake/snowflake-arctic-embed-m-v1.5": Model("97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47", 768, "cls", R, "", 512),
    "mixedbread-ai/mxbai-embed-large-v1": Model("990580e27d329c7408b3741ecff85876e128e203", 1024, "cls", R, "", 512),
}

# DocText in internal/dense drops Go's unicode.C: Cc, Cf, Co, Cs and the
# unassigned code points, Cn. Python's tables may be of another Unicode
# version than Go's; they differ only on code points assigned in between.


def doc_text(title, text):
    """Title and text joined by a space, without characters of category C
    other than tab, newline and carriage return, as DocText (D010)."""
    s = title + " " + text
    return "".join(c for c in s if c in "\t\n\r" or unicodedata.category(c)[0] != "C")


def read_corpus(data, limit=None):
    """The IDs and texts of data/corpus.jsonl in file order, the first limit only if given."""
    ids, texts = [], []
    with open(os.path.join(data, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(ids) == limit:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            ids.append(d["_id"])
            texts.append(doc_text(d.get("title", ""), d.get("text", "")))
    return ids, texts


def read_test_queries(data):
    """The test queries, those of data/qrels/test.tsv, in the order of data/queries.jsonl."""
    with open(os.path.join(data, "qrels", "test.tsv"), encoding="utf-8", newline="") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    test = {line.split("\t")[0] for line in lines[1:] if line.strip()}
    ids, texts = [], []
    with open(os.path.join(data, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q["_id"] in test:
                ids.append(q["_id"])
                texts.append(q["text"])
    missing = test - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} queries of the qrels are not in queries.jsonl, e.g. {sorted(missing)[0]}")
    return ids, texts


def write_faiss_flat(out, ids, vecs):
    """Write vecs (float32, one row per ID) as a faiss IndexFlatIP and the IDs
    one per line, byte for byte as WriteFaissFlat in internal/dense."""
    vecs = np.ascontiguousarray(vecs, dtype="<f4")
    n, dim = vecs.shape
    if n != len(ids):
        raise ValueError(f"{n} vectors for {len(ids)} IDs")
    os.makedirs(out, exist_ok=True)
    header = b"IxFI" + struct.pack("<iqqqBIQ", dim, n, 1 << 20, 1 << 20, 1, 0, n * dim)
    with open(os.path.join(out, "index"), "wb") as f:
        f.write(header + vecs.tobytes())
    with open(os.path.join(out, "docid"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(ids) + "\n")


def encode_resumable(part, texts, encode, dim, chunk):
    """Encode texts chunk by chunk with encode, appending each chunk's vectors
    to part; the whole chunks that part already holds are kept, anything
    after them is cut off. Returns the vectors and the number kept."""
    size = 4 * dim
    with open(part, "a+b") as f:
        f.seek(0)
        raw = f.read()
        kept = (len(raw) // size) // chunk * chunk
        if kept > len(texts):
            raise ValueError(f"{part} holds {kept} vectors for {len(texts)} texts")
        f.truncate(kept * size)
        out = np.empty((len(texts), dim), dtype="<f4")
        out[:kept] = np.frombuffer(raw[: kept * size], dtype="<f4").reshape(kept, dim)
        f.seek(0, os.SEEK_END)
        for i in range(kept, len(texts), chunk):
            v = np.ascontiguousarray(encode(texts[i : i + chunk]), dtype="<f4")
            if v.shape != (len(texts[i : i + chunk]), dim):
                raise ValueError(f"encoder returned shape {v.shape}")
            out[i : i + len(v)] = v
            f.write(v.tobytes())
            f.flush()
            os.fsync(f.fileno())
    return out, kept


def setup_torch():
    """Float32 matrix products throughout: TF32 off, deterministic kernels."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return torch


def load(name, device):
    """The model at its revision in float32, checked against Section 5's table."""
    torch = setup_torch()
    from sentence_transformers import SentenceTransformer

    m = MODELS[name]
    st = SentenceTransformer(name, revision=m.revision, device=device, model_kwargs={"dtype": torch.float32})
    st.default_prompt_name = None
    dtypes = {p.dtype for p in st.parameters()}
    pooling = st[1].pooling_mode
    if dtypes != {torch.float32}:
        raise ValueError(f"{name}: parameters of type {dtypes}")
    if pooling != m.pooling or st.max_seq_length != m.max_len:
        raise ValueError(f"{name}: pooling {pooling}, max_seq_length {st.max_seq_length}")
    if st.get_embedding_dimension() != m.dim:
        raise ValueError(f"{name}: dimension {st.get_embedding_dimension()}")
    return st


def encoder(st, prompt, batch):
    """A function that encodes texts with prompt before each, L2-normalised."""

    def encode(texts):
        return st.encode(
            [prompt + t for t in texts],
            batch_size=batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return encode


def settings_check(path, settings):
    """Record settings at path, or require them to equal those recorded."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old != settings:
            raise ValueError(f"{path} records {old}, not {settings}; remove the partial files to start afresh")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, sort_keys=True)


def run(name, data, out, limit=None, device="cpu", batch=32, chunk=4096, load_model=load):
    m = MODELS[name]
    st = load_model(name, device)
    os.makedirs(out, exist_ok=True)
    doc_ids, docs = read_corpus(data, limit)
    part = os.path.join(out, "vectors.part")
    settings = {"model": name, "revision": m.revision, "limit": limit, "batch": batch, "chunk": chunk}
    settings_check(part + ".json", settings)
    start = time.monotonic()
    vecs, kept = encode_resumable(part, docs, encoder(st, m.doc_prompt, batch), m.dim, chunk)
    secs = time.monotonic() - start
    q_ids, queries = read_test_queries(data)
    qvecs = encoder(st, m.query_prompt, batch)(queries)
    write_faiss_flat(os.path.join(out, "queries"), q_ids, qvecs)
    write_faiss_flat(out, doc_ids, vecs)
    os.remove(part)
    os.remove(part + ".json")
    import sentence_transformers
    import torch
    import transformers

    return {
        "model": name,
        "revision": m.revision,
        "docs": len(doc_ids),
        "queries": len(q_ids),
        "resumed_after": kept,
        "seconds": round(secs, 1),
        "docs_per_second": (len(doc_ids) - kept) / secs if secs > 0 else None,
        "device": torch.cuda.get_device_name() if device.startswith("cuda") else platform.processor() or "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "batch": batch,
        "chunk": chunk,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--data", required=True, help="BEIR dataset directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--limit", type=int, help="encode only the first LIMIT documents")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--chunk", type=int, default=4096, help="documents between two appends to vectors.part")
    a = p.parse_args(argv)
    summary = run(a.model, a.data, a.out, a.limit, a.device, a.batch, a.chunk)
    json.dump(summary, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
