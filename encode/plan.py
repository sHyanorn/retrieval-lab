"""The order of step 4 and the cut rule (PREREGISTRATION.md, Section 12).

Work is counted in BGE-base tokens: a dataset's tokens (Section 4) times the
model's forward cost per token over BGE-base's (Section 12). The projection
of the session is the hours elapsed, OVERHEAD_H for what is left besides
encoding (model loads, downloads, check 2, checksums), and the BGE-base
tokens still to encode over the measured rate.
"""

import dataclasses

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE_BASE = "BAAI/bge-base-en-v1.5"

# Section 12: NQ is cut for one model after another in this order.
CUT_ORDER = (
    "mixedbread-ai/mxbai-embed-large-v1",
    "BAAI/bge-large-en-v1.5",
    "Snowflake/snowflake-arctic-embed-m-v1.5",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-small-en-v1.5",
    MINILM,
    BGE_BASE,
)
# NQ is encoded in the reverse of the cut order; so is every other dataset.
ORDER = tuple(reversed(CUT_ORDER))

# Section 12: forward cost per token, GFLOP.
GFLOP = {
    MINILM: 0.02,
    "BAAI/bge-small-en-v1.5": 0.05,
    BGE_BASE: 0.19,
    "intfloat/e5-base-v2": 0.19,
    "thenlper/gte-base": 0.19,
    "Snowflake/snowflake-arctic-embed-m-v1.5": 0.19,
    "BAAI/bge-large-en-v1.5": 0.66,
    "mixedbread-ai/mxbai-embed-large-v1": 0.66,
}

# Section 4: BGE tokens per dataset, measured on the six; Section 12 for
# Quora (about 9 million) and NQ (about 300 million).
TOKENS = {
    "scifact": 1.7e6,
    "nfcorpus": 1.3e6,
    "arguana": 1.8e6,
    "scidocs": 5.8e6,
    "fiqa": 9.6e6,
    "trec-covid": 40.4e6,
    "quora": 9e6,
    "nq": 300e6,
}
SMALL = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid")
# Section 5, check 2: BGE-base against Pyserini's prebuilt indexes.
CHECK2 = ("scifact", "nfcorpus", "arguana", "trec-covid")
RATE_STEP = (BGE_BASE, "nfcorpus")
RESERVE = 0.2  # Section 12: on the GPU line
OVERHEAD_H = 0.25


def steps(nq=ORDER):
    """Section 12's order: BGE-base on NFCorpus, all models on SciFact, all
    models on the other five small datasets and Quora, NQ for the models nq."""
    s = [RATE_STEP] + [(m, "scifact") for m in ORDER]
    s += [(m, d) for m in ORDER for d in SMALL[1:] + ("quora",) if (m, d) != RATE_STEP]
    return s + [(m, "nq") for m in ORDER if m in nq]


def base_tokens(step):
    m, d = step
    return TOKENS[d] * GFLOP[m] / GFLOP[BGE_BASE]


def hours(rate, elapsed_h, todo, overhead_h=OVERHEAD_H):
    """Projected hours of the session at rate BGE-base tokens per second."""
    return elapsed_h + overhead_h + sum(base_tokens(s) for s in todo) / rate / 3600


@dataclasses.dataclass
class Decision:
    nq: tuple  # models that keep NQ, in encoding order
    cuts: list  # {"model", "hours", "usd"}: the projection each cut avoided
    stop: bool  # tier 1 alone does not fit: stop after the measurement
    hours: float  # projection with the cuts
    usd: float


def decide(rate, elapsed_h, price, line, limit_h, nq=ORDER, done=(), steps=steps, overhead_h=OVERHEAD_H):
    """Section 12's cut rule. The projection may exceed neither the tier's GPU
    line (USD at price per hour) nor limit_h, the hours the pod may still
    run. NQ is cut in CUT_ORDER until it does not; a model whose NQ is in done
    is kept. If the six small datasets and Quora alone exceed the line with
    its reserve, or limit_h, the session stops."""
    done = set(done)
    left = lambda keep: [s for s in steps(keep) if s not in done]
    tier1 = hours(rate, elapsed_h, left(()), overhead_h)
    if tier1 * price > line * (1 + RESERVE) or tier1 > limit_h:
        return Decision((), [], True, tier1, tier1 * price)
    keep = [m for m in ORDER if m in nq]
    cuts = []
    for m in CUT_ORDER:
        h = hours(rate, elapsed_h, left(keep), overhead_h)
        if h * price <= line and h <= limit_h:
            break
        if m in keep and (m, "nq") not in done:
            keep.remove(m)
            cuts.append({"model": m, "hours": round(h, 3), "usd": round(h * price, 2)})
    h = hours(rate, elapsed_h, left(keep), overhead_h)
    return Decision(tuple(keep), cuts, False, h, h * price)
