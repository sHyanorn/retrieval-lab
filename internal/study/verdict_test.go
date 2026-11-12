package study

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// synth makes per-query values of len(offset) models on datasets of the
// given sizes: model m scores offset[m] plus noise shared by all models
// on a query, and noise of its own of size own.
func synth(sizes []int, offset []float64, shared, own float64, seed uint64) Strata {
	r := rand.New(rand.NewPCG(seed, 7))
	s := make(Strata, len(sizes))
	for d, n := range sizes {
		for range n {
			base := shared * r.NormFloat64()
			q := make([]float64, len(offset))
			for m, o := range offset {
				q[m] = o + base + own*r.NormFloat64()
			}
			s[d] = append(s[d], q)
		}
	}
	return s
}

var sizes = []int{60, 80, 100}

func TestCompareLabels(t *testing.T) {
	for _, tc := range []struct {
		name string
		loss Strata
		want Label
	}{
		{"one model loses 0.02 more", synth(sizes, []float64{0.01, 0.01, 0.03}, 0.05, 0.002, 1), LabelApart},
		{"differences of 0.001, sharply measured", synth(sizes, []float64{0.010, 0.011, 0.0105}, 0.05, 0.0002, 2), LabelEquivalent},
		{"no difference, widely spread", synth(sizes, []float64{0.01, 0.01, 0.01}, 0.05, 0.05, 3), LabelUnresolved},
	} {
		c := Compare(tc.loss)
		if got := c.Label(Delta); got != tc.want {
			t.Errorf("%s: label %v (pairs %+v, bound %v), want %v", tc.name, got, c.Pairs, c.RangeBound, tc.want)
		}
	}
}

func TestApartNeedsBothConditionsOnOnePair(t *testing.T) {
	// Pair (0, 1) is rejected but differs by 0.001; pair (0, 2) differs by
	// 0.01 but is not rejected. No pair is apart (amendment 5), and the
	// range bound is above δ.
	c := Comparison{
		Means:      []float64{0, -0.001, -0.01},
		Pairs:      []Pair{{0, 1, 0.001, 1e-5, true}, {0, 2, 0.01, 0.3, false}, {1, 2, 0.009, 0.4, false}},
		RangeBound: 0.02,
	}
	if got := c.Label(Delta); got != LabelUnresolved {
		t.Errorf("label %v, want unresolved", got)
	}
	c.Pairs[1].Reject = true
	if got := c.Label(Delta); got != LabelApart {
		t.Errorf("label %v, want apart once the large pair is rejected", got)
	}
	if got := (Comparison{Pairs: c.Pairs[:1], RangeBound: 0.004}).Label(Delta); got != LabelEquivalent {
		t.Errorf("label %v, want equivalent", got)
	}
}

func TestH1(t *testing.T) {
	for l, want := range map[Label]Verdict{LabelApart: Supported, LabelEquivalent: NotSupported, LabelUnresolved: Unresolved} {
		if got := H1(l); got != want {
			t.Errorf("H1(%v) = %v, want %v", l, got, want)
		}
	}
}

func comparison(label Label, means ...float64) Comparison {
	switch label {
	case LabelApart:
		return Comparison{Means: means, Pairs: []Pair{{0, 1, 0.01, 1e-4, true}}, RangeBound: 0.02}
	case LabelEquivalent:
		return Comparison{Means: means, RangeBound: 0.001}
	}
	return Comparison{Means: means, RangeBound: 0.02}
}

func TestH2(t *testing.T) {
	loss := synth(sizes, []float64{0.001, 0.002}, 0.01, 0.001, 4)
	small, large := []float64{0.001, 0.002}, []float64{0.006, 0.0065}
	for _, tc := range []struct {
		name      string
		sets      int
		at95      Comparison
		at40      Label
		verdict   Verdict
		reading   Reading
		negligent bool
	}{
		{"two datasets", 2, comparison(LabelEquivalent, small...), LabelApart, NotTestable, ReadingNone, false},
		{"apart at 0.95", 3, comparison(LabelApart, small...), LabelApart, NotSupported, ReadingNone, false},
		{"unresolved at 0.95", 3, comparison(LabelUnresolved, small...), LabelApart, Unresolved, ReadingNone, false},
		{"apart at 40, losses at 0.95 not negligible", 3, comparison(LabelEquivalent, large...), LabelApart, Supported, ReadingRemoved, false},
		{"apart at 40, losses at 0.95 negligible", 3, comparison(LabelEquivalent, small...), LabelApart, Supported, ReadingSideBySide, true},
		{"equivalent at 40", 3, comparison(LabelEquivalent, large...), LabelEquivalent, Supported, ReadingNeither, false},
		{"unresolved at 40", 3, comparison(LabelEquivalent, small...), LabelUnresolved, Supported, ReadingAt95Only, true},
	} {
		r := H2(tc.sets, tc.at95, comparison(tc.at40), loss, Delta)
		if r.Verdict != tc.verdict || r.Reading != tc.reading || r.Negligible != tc.negligent {
			t.Errorf("%s: %+v", tc.name, r)
		}
		if (r.Verdict == Supported) != (len(r.Intervals) == 2) {
			t.Errorf("%s: %d intervals", tc.name, len(r.Intervals))
		}
	}
	// A loss of exactly δ is outside (−δ, δ).
	if r := H2(3, comparison(LabelEquivalent, 0, Delta), comparison(LabelApart), loss, Delta); r.Reading != ReadingRemoved {
		t.Errorf("a loss of δ read as %v", r.Reading)
	}
}

func pairsP(ps ...float64) []Pair {
	var out []Pair
	for i, p := range ps {
		out = append(out, Pair{I: 0, J: i + 1, Diff: 0.01, P: p})
	}
	return out
}

func TestH3(t *testing.T) {
	order := []float64{0.5, 0.4, 0.3}
	for _, tc := range []struct {
		name          string
		exact, approx Strata
		verdict       Verdict
	}{
		{"same order", synth(sizes, order, 0.1, 0.01, 5), synth(sizes, order, 0.1, 0.01, 6), Supported},
		{"a reversal", synth(sizes, order, 0.1, 0.01, 5), synth(sizes, []float64{0.4, 0.5, 0.3}, 0.1, 0.01, 6), NotSupported},
		{"exact orders no pair", synth(sizes, []float64{0.4, 0.4, 0.4}, 0.1, 0.05, 5), synth(sizes, order, 0.1, 0.01, 6), NotInformative},
	} {
		r, err := H3(3, tc.exact, tc.approx)
		if err != nil || r.Verdict != tc.verdict {
			t.Errorf("%s: %v, %+v", tc.name, r.Verdict, r)
		}
		if tc.verdict == Supported && (r.Tau != 1 || len(r.TauByDataset) != 3) {
			t.Errorf("%s: τ %v by dataset %v", tc.name, r.Tau, r.TauByDataset)
		}
	}
	if r, err := H3(2, nil, nil); err != nil || r.Verdict != NotTestable {
		t.Errorf("two datasets: %v", r.Verdict)
	}
	// Exact orders two pairs, HNSW one of them: one unordered pair, no
	// reversal; two of three ordered is at least half.
	e, a := pairsP(0.001, 0.002, 0.9), pairsP(0.001, 0.6, 0.9)
	if r := h3(e, a, broken{}); r.Verdict != Supported || r.Ordered != 2 || r.Unordered != 1 || r.Reversed != 0 {
		t.Errorf("unordered pair: %+v", r)
	}
	// Each search is its own family of 3 (amendment 4): p = 0.01, 0.02,
	// 0.04 are all rejected; one family of 6 would reject none.
	e, a = pairsP(0.01, 0.02, 0.04), pairsP(0.01, 0.02, 0.04)
	if r := h3(e, a, broken{}); r.Ordered != 3 || r.Verdict != Supported {
		t.Errorf("two families: %+v", r)
	}
	e, a = pairsP(0.01, 0.02, 0.04), pairsP(0.01, 0.02, 0.04)
	if r := h3(e, a, broken{oneFamily: true}); r.Ordered == 3 {
		t.Error("one family of both searches passed")
	}
	// One ordered pair of three is fewer than half.
	e, a = pairsP(0.001, 0.5, 0.9), pairsP(0.001, 0.5, 0.9)
	if r := h3(e, a, broken{}); r.Verdict != NotInformative {
		t.Errorf("one of three ordered: %v", r.Verdict)
	}
}

func TestH4(t *testing.T) {
	cell := func(d string, matched bool, spread float64) H4Cell {
		return H4Cell{Model: "m", Dataset: d, Seed1Matched: matched, Builds: []float64{0.5, 0.5 + spread, 0.5, 0.5}, BuildsAt40: []float64{0.4, 0.41, 0.4, 0.4}}
	}
	for _, tc := range []struct {
		name  string
		cells []H4Cell
		want  Verdict
	}{
		{"all below δ", []H4Cell{cell("a", true, 0.004), cell("b", true, 0), cell("c", true, 0.001)}, Supported},
		{"a range of exactly δ", []H4Cell{cell("a", true, 0.004), cell("b", true, 0.005), cell("c", true, 0)}, NotSupported},
		{"the wide cell is not matched", []H4Cell{cell("a", true, 0), cell("b", true, 0), cell("c", true, 0), cell("d", false, 0.1)}, Supported},
		{"two datasets matched", []H4Cell{cell("a", true, 0), cell("b", true, 0), cell("c", false, 0)}, NotTestable},
	} {
		r := H4(tc.cells, Delta)
		if r.Verdict != tc.want {
			t.Errorf("%s: %v, ranges %v", tc.name, r.Verdict, r.Ranges)
		}
	}
	if r := H4([]H4Cell{cell("a", true, 0)}, Delta); math.Abs(r.Ranges40[0]-0.01) > 1e-12 {
		t.Errorf("range at ef 40 = %v", r.Ranges40)
	}
}

func TestH5(t *testing.T) {
	dense := synth(sizes, []float64{0.004, 0.003}, 0.002, 0.002, 8)
	scaled := func(s Strata, f, noise float64, seed uint64) Strata {
		r := rand.New(rand.NewPCG(seed, 9))
		out := make(Strata, len(s))
		for d := range s {
			for _, q := range s[d] {
				x := slices.Clone(q)
				for m := range x {
					x[m] = f*x[m] + noise*r.NormFloat64()
				}
				out[d] = append(out[d], x)
			}
		}
		return out
	}
	for _, tc := range []struct {
		name         string
		dense, fused Strata
		want         Verdict
	}{
		{"fusion keeps a tenth", dense, scaled(dense, 0.1, 0, 1), Supported},
		{"fusion keeps it all", dense, scaled(dense, 1, 0, 1), NotSupported},
		{"fused losses widely spread", dense, scaled(dense, 0.5, 0.05, 1), Unresolved},
		{"no dense loss", synth(sizes, []float64{0, 0}, 0.01, 0.01, 8), scaled(dense, 0.1, 0, 1), NotTestable},
	} {
		r := H5(tc.dense, tc.fused)
		if r.Verdict != tc.want {
			t.Errorf("%s: %v %+v", tc.name, r.Verdict, r)
		}
	}
	r := H5(dense, scaled(dense, 0.1, 0, 1))
	if d := r.Phi - 0.9; d > 1e-12 || d < -1e-12 {
		t.Errorf("φ = %v, want 0.9", r.Phi)
	}
}

func TestConfirmatory(t *testing.T) {
	all := []string{"m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"}
	small := ConfirmatoryDatasets[:6]
	enc := func(nq ...string) map[string][]string {
		e := map[string][]string{"nq": nq}
		for _, d := range small {
			e[d] = all
		}
		return e
	}
	for _, tc := range []struct {
		name     string
		failed   []string
		nq       []string
		tier1    bool
		models   []string
		datasets int
	}{
		{"2b, no cut", nil, all, false, all, 7},
		{"one cut", nil, all[:7], false, all[:7], 7},
		{"two cuts", nil, all[:6], false, all[:6], 7},
		{"three cuts", nil, all[:5], false, all, 6},
		{"tier 1", nil, all, true, all, 6},
		// A failed model on NQ is removed first: six left on NQ.
		{"one failed of seven on NQ", []string{"m2"}, all[:7], false, []string{"m1", "m3", "m4", "m5", "m6", "m7"}, 7},
		// Removed first, it leaves five on NQ: NQ leaves.
		{"one failed of six on NQ", []string{"m2"}, all[:6], false, []string{"m1", "m3", "m4", "m5", "m6", "m7", "m8"}, 6},
	} {
		m, d := Confirmatory(all, tc.failed, enc(tc.nq...), tc.tier1)
		if !slices.Equal(m, tc.models) || len(d) != tc.datasets || !slices.Equal(d[:6], small) {
			t.Errorf("%s: %v over %v", tc.name, m, d)
		}
	}
}
