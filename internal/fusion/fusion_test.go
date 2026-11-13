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
	got := RRF(runs, []int{1000, 1000}, 60, 3)
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
	got := Normalize(runs, []int{1000, 1000}, 1000)
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

func TestDepthPerRun(t *testing.T) {
	// The second run is cut after its first hit: d4, its second, gets
	// nothing from it, while the first run keeps all three.
	got := RRF(runs, []int{1000, 1}, 60, 1000)
	r61, r62, r63 := 1/float64(61), 1/float64(62), 1/float64(63)
	want := []eval.Hit{h("d2", r62+r61), h("d1", r61), h("d3", r63)}
	if !slices.Equal(got["q1"], want) {
		t.Errorf("RRF = %v, want %v", got["q1"], want)
	}
	// One depth for both, the first run's, keeps d4.
	if same := RRF(runs, []int{1000, 1000}, 60, 1000); slices.Equal(same["q1"], want) {
		t.Error("a depth shared by both runs gave the same fusion")
	}
	// Normalize scales what is left of a run after the cut: the first run
	// cut to two hits maps d1 to 1 and d2 to 0; d1 and d2 tie at 1.
	got = Normalize(runs, []int{2, 1000}, 1000)
	want = []eval.Hit{h("d1", 1), h("d2", 1), h("d4", 0)}
	if !slices.Equal(got["q1"], want) {
		t.Errorf("Normalize = %v, want %v", got["q1"], want)
	}
}

func TestTrecOrder(t *testing.T) {
	// d2 and d3 tie at 0.5 across a depth of 2. The run lists them by
	// document ID ascending, as exact search writes ties; trec_eval puts d3
	// first.
	lines := []eval.Hit{h("d1", 0.9), h("d2", 0.5), h("d3", 0.5)}
	run := eval.Run{"q1": slices.Clone(lines)}
	ranked := TrecOrder(run)
	want := []eval.Hit{h("d1", 0.9), h("d3", 0.5), h("d2", 0.5)}
	if !slices.Equal(ranked["q1"], want) {
		t.Errorf("TrecOrder = %v, want %v", ranked["q1"], want)
	}
	if !slices.Equal(run["q1"], lines) {
		t.Errorf("TrecOrder changed its input to %v", run["q1"])
	}
	if again := TrecOrder(ranked); !slices.Equal(again["q1"], want) {
		t.Errorf("TrecOrder of a run in trec_eval's order = %v, want it unchanged", again["q1"])
	}
	// The depth cuts trec_eval's order: d3 is kept, where the lines keep d2.
	r61, r62 := 1/float64(61), 1/float64(62)
	if got, want := RRF([]eval.Run{ranked}, []int{2}, 60, 1000)["q1"], []eval.Hit{h("d1", r61), h("d3", r62)}; !slices.Equal(got, want) {
		t.Errorf("RRF = %v, want %v", got, want)
	}
	if got, want := RRF([]eval.Run{run}, []int{2}, 60, 1000)["q1"], []eval.Hit{h("d1", r61), h("d2", r62)}; !slices.Equal(got, want) {
		t.Errorf("RRF by lines = %v, want %v", got, want)
	}
	// Normalize maps the tied document kept to 0, the cut run's minimum, in
	// either order; only which of the two is kept differs.
	if got, want := Normalize([]eval.Run{ranked}, []int{2}, 1000)["q1"], []eval.Hit{h("d1", 1), h("d3", 0)}; !slices.Equal(got, want) {
		t.Errorf("Normalize = %v, want %v", got, want)
	}
	if got, want := Normalize([]eval.Run{run}, []int{2}, 1000)["q1"], []eval.Hit{h("d1", 1), h("d2", 0)}; !slices.Equal(got, want) {
		t.Errorf("Normalize by lines = %v, want %v", got, want)
	}
}
