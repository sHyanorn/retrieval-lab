import pytest

import plan as P

BASE = "BAAI/bge-base-en-v1.5"
MXBAI = "mixedbread-ai/mxbai-embed-large-v1"
TIER1 = [s for s in P.steps(()) if s[1] != "nq"]


def test_registered_constants():
    # PREREGISTRATION.md, Section 12.
    assert P.CUT_ORDER == (
        MXBAI,
        "BAAI/bge-large-en-v1.5",
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        "thenlper/gte-base",
        "intfloat/e5-base-v2",
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        BASE,
    )
    assert P.GFLOP[BASE] == 0.19 and P.GFLOP[MXBAI] == 0.66
    assert sum(P.TOKENS[d] for d in P.SMALL) == pytest.approx(61e6, rel=0.01)
    # Tier 2b in BGE-base tokens: about 4.19 billion.
    assert sum(P.base_tokens(s) for s in P.steps()) == pytest.approx(4.19e9, rel=0.005)


def test_order():
    s = P.steps()
    assert len(s) == len(set(s)) == 64
    assert s[0] == (BASE, "nfcorpus")
    assert s[1:9] == [(m, "scifact") for m in P.ORDER]
    assert {d for _, d in s[9:56]} == {"nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora"}
    # NQ last, by model in the reverse of the cut order.
    assert s[56:] == [(m, "nq") for m in reversed(P.CUT_ORDER)]
    assert P.steps((MXBAI, BASE))[-2:] == [(BASE, "nq"), (MXBAI, "nq")]


def rate_for(hours, elapsed=1.0, overhead=P.OVERHEAD_H, todo=None):
    """The rate at which the projection of todo is exactly hours."""
    todo = P.steps() if todo is None else todo
    return sum(P.base_tokens(s) for s in todo) / 3600 / (hours - elapsed - overhead)


def test_time_limit_boundary():
    r = rate_for(9.0)
    d = P.decide(r * (1 + 1e-9), 1.0, 1.19, 17.9, 9.0)
    assert d.nq == P.ORDER and d.cuts == [] and not d.stop
    d = P.decide(r * (1 - 1e-9), 1.0, 1.19, 17.9, 9.0)
    assert d.nq == tuple(m for m in P.ORDER if m != MXBAI)
    assert [c["model"] for c in d.cuts] == [MXBAI] and not d.stop
    assert d.hours <= 9.0


def test_cost_line_boundary():
    # On the dearer GPU the line binds before the time limit.
    line_h = 17.9 / 2.49
    r = rate_for(line_h)
    assert not P.decide(r * (1 + 1e-9), 1.0, 2.49, 17.9, 100).cuts
    assert len(P.decide(r * (1 - 1e-9), 1.0, 2.49, 17.9, 100).cuts) == 1


def test_cuts_follow_the_order():
    # Room for tier 1 and NQ of all but the two large models.
    keep = P.steps(P.CUT_ORDER[2:])
    d = P.decide(rate_for(9.0, todo=keep) * 1.0001, 1.0, 1.19, 17.9, 9.0)
    assert [c["model"] for c in d.cuts] == list(P.CUT_ORDER[:2])
    assert d.nq == P.ORDER[:6]


def test_all_cut():
    # Tier 1 fits, NQ of BGE-base alone does not.
    r = rate_for(9.0, todo=TIER1) * 1.0001
    d = P.decide(r, 1.0, 1.19, 17.9, 9.0)
    assert d.nq == () and [c["model"] for c in d.cuts] == list(P.CUT_ORDER) and not d.stop


def test_stop_when_tier1_does_not_fit():
    r = rate_for(9.0, todo=TIER1)
    d = P.decide(r * 0.9999, 1.0, 1.19, 17.9, 9.0)
    assert d.stop and d.nq == ()
    # By cost: above the line all NQ goes; above the line with its reserve the session stops.
    r = rate_for(17.9 / 2.49, todo=TIER1)
    d = P.decide(r * 0.99, 1.0, 2.49, 17.9, 100)
    assert not d.stop and d.nq == ()
    r = rate_for(17.9 * 1.2 / 2.49, todo=TIER1)
    assert not P.decide(r * 1.0001, 1.0, 2.49, 17.9, 100).stop
    assert P.decide(r * 0.9999, 1.0, 2.49, 17.9, 100).stop


def test_done_steps_not_counted():
    # Before NQ: tier 1 done, the elapsed time holds it.
    done = set(TIER1)
    r = rate_for(9.0, elapsed=4.0, todo=[s for s in P.steps() if s[1] == "nq"])
    assert P.decide(r * 1.0001, 4.0, 1.19, 17.9, 9.0, done=done).cuts == []
    d = P.decide(r * 0.9999, 4.0, 1.19, 17.9, 9.0, done=done)
    assert [c["model"] for c in d.cuts] == [MXBAI]
    # A model already cut stays cut; one whose NQ is done is not cut.
    nq = P.ORDER[:3]
    done |= {(BASE, "nq")}
    d = P.decide(1.0, 4.0, 1.19, 17.9, 9.0, nq=nq, done=done)
    assert d.nq == (BASE,)


def test_filtered_steps():
    # The dry run: one model, one dataset.
    keep = lambda nq: [s for s in P.steps(nq) if s == ("sentence-transformers/all-MiniLM-L6-v2", "scifact")]
    d = P.decide(1000.0, 0.0, 1.19, 17.9, 9.0, steps=keep)
    assert not d.stop and d.cuts == []
