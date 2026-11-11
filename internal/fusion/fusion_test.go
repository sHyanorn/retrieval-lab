package fusion

import (
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func h(id string, score float64) eval.Hit { return eval.Hit{DocID: id, Score: score} }

var runs = []eval.Run{
	{"q1": {h("d1", 3), h("d2", 2), h("d3", 1)}},
	{"q1": {h("d2", 0.9), h("d4", 0.5)}, "q2": {h("d9", 0.3)}},
}

func TestRRF(t *testing.T) {
	// d2 is second in one run and first in the other.
	got := RRF(runs, 60, 1000, 3)
	r61, r62 := 1/float64(61), 1/float64(62) // in float64, as the runs are summed
	want := eval.Run{
		"q1": {h("d2", r62+r61), h("d1", r61), h("d4", r62)},
		"q2": {h("d9", r61)},
	}
	for qid, hits := range want {
		if !slices.Equal(got[qid], hits) {
			t.Errorf("%s: %v, want %v", qid, got[qid], hits)
		}
	}
}

func TestNormalize(t *testing.T) {
	// d3 and d4 both sum to 0 and go in document ID order; q2's one score
	// has max = min and becomes 1.
	got := Normalize(runs, 1000, 1000)
	want := eval.Run{
		"q1": {h("d2", 1.5), h("d1", 1), h("d3", 0), h("d4", 0)},
		"q2": {h("d9", 1)},
	}
	for qid, hits := range want {
		if !slices.Equal(got[qid], hits) {
			t.Errorf("%s: %v, want %v", qid, got[qid], hits)
		}
	}
}
