package eval

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"
)

// The fixture covers a tie at the top, a graded qrel, a negative (unjudged)
// grade, a query missing from the run, a query whose judgments are all zero,
// a relevant document never retrieved, and a run query absent from qrels.
var (
	fixtureQrels = Qrels{
		"q1": {"d1": 2, "d2": 1, "d3": 0, "d4": -1},
		"q2": {"d1": 1},
		"q3": {"d5": 0},
		"q4": {"d9": 1},
	}
	fixtureRun = `q1 Q0 d3 1 5.0 r
q1 Q0 d1 2 5.0 r
q1 Q0 d2 3 4.0 r
q1 Q0 d4 4 3.0 r
q3 Q0 d5 1 1.0 r
q3 Q0 d6 2 0.5 r
q9 Q0 d1 1 1.0 r
`
	// trec_eval v10.0: trec_eval -c -q -m ndcg_cut.10 -m recall.100 on the fixture.
	fixtureTrecEval = "recall_100            \tq1\t1.0000\n" +
		"ndcg_cut_10           \tq1\t0.6697\n" +
		"recall_100            \tq2\t0.0000\n" +
		"ndcg_cut_10           \tq2\t0.0000\n" +
		"recall_100            \tq3\t0.0000\n" +
		"ndcg_cut_10           \tq3\t0.0000\n" +
		"recall_100            \tq4\t0.0000\n" +
		"ndcg_cut_10           \tq4\t0.0000\n" +
		"recall_100            \tall\t0.2500\n" +
		"ndcg_cut_10           \tall\t0.1674\n"
)

func readFixture(t *testing.T) Run {
	t.Helper()
	run, err := ReadRun(strings.NewReader(fixtureRun))
	if err != nil {
		t.Fatal(err)
	}
	return run
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestNDCGBreaksTiesByDocIDDescending(t *testing.T) {
	// d3 (grade 0) and d1 (grade 2) tie; trec_eval ranks d3 first.
	// DCG = 2/log2(3) + 1/log2(4); ideal = 2 + 1/log2(3).
	want := (2/math.Log2(3) + 0.5) / (2 + 1/math.Log2(3))
	got := NDCG(fixtureQrels["q1"], Ranked(readFixture(t)["q1"]), 10)
	if !near(got, want) {
		t.Fatalf("NDCG = %v, want %v", got, want)
	}
}

func TestNDCGCutsRankingAndIdealAtK(t *testing.T) {
	judged := map[string]int{"a": 1, "b": 1, "c": 1}
	ranked := []Hit{{"x", 3}, {"a", 2}, {"b", 1}}
	if got := NDCG(judged, ranked, 1); got != 0 {
		t.Fatalf("NDCG@1 = %v, want 0", got)
	}
	// DCG@2 = 1/log2(3); ideal@2 = 1 + 1/log2(3), with c beyond the cut.
	want := (1 / math.Log2(3)) / (1 + 1/math.Log2(3))
	if got := NDCG(judged, ranked, 2); !near(got, want) {
		t.Fatalf("NDCG@2 = %v, want %v", got, want)
	}
}

func TestRecallCountsPositiveGradesInTopK(t *testing.T) {
	judged := map[string]int{"a": 2, "b": 1, "c": 0, "d": 1}
	ranked := []Hit{{"a", 4}, {"c", 3}, {"b", 2}, {"d", 1}}
	if got := Recall(judged, ranked, 3); !near(got, 2.0/3) {
		t.Fatalf("Recall@3 = %v, want 2/3", got)
	}
}

func TestEvaluateMatchesTrecEval(t *testing.T) {
	var b bytes.Buffer
	if err := WriteTrecEval(&b, Evaluate(fixtureQrels, readFixture(t))); err != nil {
		t.Fatal(err)
	}
	if b.String() != fixtureTrecEval {
		t.Fatalf("got\n%s\nwant\n%s", b.String(), fixtureTrecEval)
	}
}

func TestReadRunRejectsDuplicates(t *testing.T) {
	_, err := ReadRun(strings.NewReader("q1 Q0 d1 1 2.0 r\nq1 Q0 d1 2 1.0 r\n"))
	if err == nil {
		t.Fatal("want an error for a document listed twice")
	}
}

func TestReadRunRejectsShortLines(t *testing.T) {
	if _, err := ReadRun(strings.NewReader("q1 Q0 d1 1 2.0\n")); err == nil {
		t.Fatal("want an error for a line with five fields")
	}
}

func TestWriteRunRoundTrips(t *testing.T) {
	hits := []Hit{{"d2", 12.3456}, {"d1", 0.5}}
	var b bytes.Buffer
	if err := WriteRun(&b, "q1", hits, "bm25"); err != nil {
		t.Fatal(err)
	}
	want := "q1 Q0 d2 1 12.345600 bm25\nq1 Q0 d1 2 0.500000 bm25\n"
	if b.String() != want {
		t.Fatalf("got %q, want %q", b.String(), want)
	}
	run, err := ReadRun(&b)
	if err != nil {
		t.Fatal(err)
	}
	if got := run["q1"]; len(got) != 2 || got[0] != hits[0] || got[1] != hits[1] {
		t.Fatalf("round trip gave %v", got)
	}
}

func TestNDCGDiscountsMatchCLibm(t *testing.T) {
	// Go's math.Log2(3) is one unit in the last place below C's, which
	// moves this value across a rounding boundary. trec_eval v10.0 prints
	// 0.6313 for it.
	judged := map[string]int{"x": 19219, "y": 36111454}
	ranked := []Hit{{"x", 2}, {"y", 1}}
	if got := fmt.Sprintf("%6.4f", NDCG(judged, ranked, 10)); got != "0.6313" {
		t.Fatalf("NDCG = %s, want 0.6313", got)
	}
}

func TestReadRunRejectsNonFiniteScores(t *testing.T) {
	for _, s := range []string{"nan", "inf", "-Inf"} {
		if _, err := ReadRun(strings.NewReader("q1 Q0 d1 1 " + s + " r\n")); err == nil {
			t.Errorf("want an error for score %s", s)
		}
	}
}
