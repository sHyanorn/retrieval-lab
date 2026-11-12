package study

import (
	"math/big"
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func TestStudyGrid(t *testing.T) {
	efs := dense.StudyEFs
	if len(efs) != 65 || !slices.IsSorted(efs) || efs[0] != 10 || efs[50] != 60 || efs[64] != 1000 {
		t.Fatalf("grid = %v", efs)
	}
	for _, ef := range dense.EFs {
		if !slices.Contains(efs, ef) {
			t.Errorf("M3's ef %d is not in the grid", ef)
		}
	}
	for _, c := range [][2]int{{10, 10}, {90, 10}, {99, 10}, {100, 100}, {1000, 100}} {
		if got := dense.GraphDepth(c[0]); got != c[1] {
			t.Errorf("GraphDepth(%d) = %d, want %d", c[0], got, c[1])
		}
	}
}

func TestMeanRecall10IsExact(t *testing.T) {
	// 19 of 20 neighbours found: 0.95 exactly, which float64 cannot hold.
	qs := []eval.Query{{QID: "a", Found10: 10, Of10: 10}, {QID: "b", Found10: 9, Of10: 10}}
	got, err := MeanRecall10(qs)
	if err != nil || got.Cmp(big.NewRat(95, 100)) != 0 {
		t.Fatalf("MeanRecall10 = %v, %v, want 19/20", got, err)
	}
	// A query whose exact list holds fewer than 10 counts as Found/Of.
	got, err = MeanRecall10([]eval.Query{{QID: "a", Found10: 3, Of10: 4}, {QID: "b", Found10: 10, Of10: 10}})
	if err != nil || got.Cmp(big.NewRat(7, 8)) != 0 {
		t.Errorf("MeanRecall10 = %v, want 7/8", got)
	}
	if _, err := MeanRecall10([]eval.Query{{QID: "a", Found10: 1}}); err == nil {
		t.Error("a query without an exact list was accepted")
	}
}

func pts(r ...[2]int64) []Point {
	var ps []Point
	for _, x := range r {
		ps = append(ps, Point{EF: int(x[0]), Recall: big.NewRat(x[1], 10000)})
	}
	return ps
}

func TestSelect(t *testing.T) {
	r := big.NewRat(95, 100)
	for _, tc := range []struct {
		name   string
		points []Point
		ef     int
		status Status
	}{
		{"first ef reaching r", pts([2]int64{12, 9500}, [2]int64{10, 9000}, [2]int64{11, 9400}, [2]int64{13, 9700}), 12, Matched},
		// 0.96 − 0.95 is 0.01 exactly, matched; in float64 it exceeds 0.01.
		{"overshoot of exactly 0.01", pts([2]int64{10, 9300}, [2]int64{11, 9600}), 11, Matched},
		{"overshoot above 0.01", pts([2]int64{10, 9300}, [2]int64{11, 9601}), 11, Overshoot},
		{"below reach", pts([2]int64{10, 9700}, [2]int64{11, 9800}), 10, BelowReach},
		{"at the first ef within 0.01", pts([2]int64{10, 9550}, [2]int64{11, 9800}), 10, Matched},
		{"unreached", pts([2]int64{10, 9000}, [2]int64{1000, 9499}), 1000, Unreached},
	} {
		s, err := Select(tc.points, r)
		if err != nil || s.EF != tc.ef || s.Status != tc.status {
			t.Errorf("%s: Select = %+v, %v; want ef %d, %v", tc.name, s, err, tc.ef, tc.status)
		}
	}
	a, b := 0.96, 0.95
	if a-b <= 0.01 {
		t.Error("the float64 case the exact comparison guards against no longer arises")
	}
	if _, err := Select(pts([2]int64{10, 1}, [2]int64{10, 2}), r); err == nil {
		t.Error("an ef given twice was accepted")
	}
}

func TestRecallSet(t *testing.T) {
	m := Selection{Status: Matched}
	o := Selection{Status: Overshoot}
	builds := map[[2]string][]Selection{
		{"a", "d1"}: {m, m, m, m}, {"b", "d1"}: {m, m, m, m},
		{"a", "d2"}: {m, m, m, m}, {"b", "d2"}: {m, m, o, m},
		{"a", "d3"}: {m, m, m, m}, {"b", "d3"}: {m, m, m, m},
	}
	got := RecallSet([]string{"a", "b"}, []string{"d1", "d2", "d3", "d4"}, func(model, d string) bool {
		return AllMatched(builds[[2]string{model, d}])
	})
	if !slices.Equal(got, []string{"d1", "d3"}) {
		t.Errorf("RecallSet = %v, want [d1 d3]", got)
	}
}

func TestClosest(t *testing.T) {
	pts := []Point{
		{EF: 12, Recall: big.NewRat(99, 100)},
		{EF: 11, Recall: big.NewRat(94, 100)},
		{EF: 10, Recall: big.NewRat(92, 100)},
	}
	// 0.93 − 0.92 and 0.94 − 0.93 tie exactly: the smaller ef. In float64
	// the second difference is the smaller, and ef 11 would be chosen.
	lo, r, hi := 0.92, 0.93, 0.94
	if hi-r >= r-lo {
		t.Fatal("the float64 trap this test relies on is gone")
	}
	got, err := Closest(pts, big.NewRat(93, 100))
	if err != nil || got.EF != 10 || got.Recall.Cmp(big.NewRat(92, 100)) != 0 {
		t.Fatalf("Closest(0.93) = %+v, %v; want ef 10", got, err)
	}
	for _, c := range []struct {
		r  *big.Rat
		ef int
	}{{big.NewRat(98, 100), 12}, {big.NewRat(1, 2), 10}} {
		if got, err := Closest(pts, c.r); err != nil || got.EF != c.ef {
			t.Fatalf("Closest(%s) = %+v, %v; want ef %d", c.r.FloatString(2), got, err, c.ef)
		}
	}
	if _, err := Closest(nil, big.NewRat(1, 2)); err == nil {
		t.Fatal("want an error with no points")
	}
	if _, err := Closest([]Point{pts[0], pts[0]}, big.NewRat(1, 2)); err == nil {
		t.Fatal("want an error for an ef given twice")
	}
}
