package study

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// column makes a stratum of one variable from the given values.
func column(v ...float64) [][]float64 {
	out := make([][]float64, len(v))
	for i, x := range v {
		out[i] = []float64{x}
	}
	return out
}

func repeat(x float64, n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = x
	}
	return v
}

func first(m []float64) float64 { return m[0] }

func TestBootstrapDrawOrder(t *testing.T) {
	s := Strata{column(1, 10, 100), column(1000, 2000)}
	// The registered order, written out: a fresh PCG(20261111, 0), then
	// resample by resample, dataset by dataset, query by query.
	r := rand.New(rand.NewPCG(20261111, 0))
	var want []float64
	for range 3 {
		var a, b float64
		for range 3 {
			a += []float64{1, 10, 100}[r.IntN(3)]
		}
		for range 2 {
			b += []float64{1000, 2000}[r.IntN(2)]
		}
		want = append(want, (a/3+b/2)/2)
	}
	slices.Sort(want)
	got := Bootstrap(s, 3, first)
	if !slices.Equal(got, want) {
		t.Fatalf("Bootstrap = %v, want %v", got, want)
	}
	if again := Bootstrap(s, 3, first); !slices.Equal(again, got) {
		t.Error("a second interval did not start the generator afresh")
	}
}

func TestBootstrapResamplesWithinDatasets(t *testing.T) {
	// Every resample within datasets holds 10 zeros and 10 ones: its mean
	// is 0.5 exactly. Drawn from the pool of 20 it varies.
	s := Strata{column(repeat(0, 10)...), column(repeat(1, 10)...)}
	lo, hi := Interval90(Bootstrap(s, 1000, first))
	if lo != 0.5 || hi != 0.5 {
		t.Errorf("stratified interval = [%v, %v], want [0.5, 0.5]", lo, hi)
	}
	lo, hi = Interval90(bootstrap(s, 1000, first, broken{pooled: true}))
	if lo == 0.5 && hi == 0.5 {
		t.Error("the pooled variant passed")
	}
}

func TestMeansWeightDatasetsAlike(t *testing.T) {
	// 10 zeros and 30 ones: each dataset weighs 1/2, so the mean is 0.5,
	// not the 0.75 of the 40 queries.
	s := Strata{column(repeat(0, 10)...), column(repeat(1, 30)...)}
	if m := s.Means(); m[0] != 0.5 {
		t.Errorf("Means = %v, want 0.5", m)
	}
	lo, hi := Interval90(Bootstrap(s, 1000, first))
	if lo != 0.5 || hi != 0.5 {
		t.Errorf("interval = [%v, %v], want [0.5, 0.5]", lo, hi)
	}
	if _, hi := Interval90(bootstrap(s, 1000, first, broken{unweighted: true})); hi == 0.5 {
		t.Error("the variant weighting queries alike passed")
	}
}

func TestIntervals(t *testing.T) {
	v := make([]float64, 10000)
	for i := range v {
		v[i] = float64(i)
	}
	// The 501st and 9,500th of 10,000; the 251st and 9,750th (amendment 2).
	if lo, hi := Interval90(v); lo != 500 || hi != 9499 {
		t.Errorf("Interval90 = [%v, %v], want [500, 9499]", lo, hi)
	}
	if lo, hi := Interval95(v); lo != 250 || hi != 9749 {
		t.Errorf("Interval95 = [%v, %v], want [250, 9749]", lo, hi)
	}
	if lo, hi := percentile(v, 20, broken{offByOne: true}); lo == 500 && hi == 9499 {
		t.Error("the percentile one place inward passed")
	}
}

func TestRangeBound(t *testing.T) {
	// Model 1 loses 0.5 more than model 0 on every query of B and nothing
	// more on A, so every resample's range is (0 + 0.5) / 2 = 0.25 when the
	// datasets weigh alike, and 0.375 when the 40 queries do.
	var a, b [][]float64
	for i := range 10 {
		a = append(a, []float64{float64(i), float64(i)})
	}
	for i := range 30 {
		b = append(b, []float64{float64(i), float64(i) + 0.5})
	}
	s := Strata{a, b}
	if got := RangeBound(s); got != 0.25 {
		t.Errorf("RangeBound = %v, want 0.25", got)
	}
	_, hi := Interval90(bootstrap(s, Resamples, Range, broken{unweighted: true}))
	if hi == 0.25 {
		t.Error("the variant weighting queries alike passed")
	}
	if got := Range([]float64{0.3, -0.1, 0.2}); got != 0.4 {
		t.Errorf("Range = %v, want 0.4", got)
	}
}

// exactSignFlip enumerates all 2^n sign patterns of integer differences
// k[d][q] / scale[d], where scale[d] makes the weighted sum an integer:
// the exact p-value that SignFlip estimates, counting the observed signs.
func exactSignFlip(k [][]int64, mult []int64) float64 {
	var flat, w []int64
	for d, qs := range k {
		for _, x := range qs {
			flat = append(flat, x)
			w = append(w, mult[d])
		}
	}
	var obs int64
	for i, x := range flat {
		obs += w[i] * x
	}
	obs = max(obs, -obs)
	n := len(flat)
	count := 0
	for pat := range 1 << n {
		var t int64
		for i, x := range flat {
			if pat>>i&1 == 1 {
				t -= w[i] * x
			} else {
				t += w[i] * x
			}
		}
		if max(t, -t) >= obs {
			count++
		}
	}
	return float64(count) / float64(int(1)<<n)
}

func floats(k [][]int64, scale float64) [][]float64 {
	out := make([][]float64, len(k))
	for d, qs := range k {
		for _, x := range qs {
			out[d] = append(out[d], float64(x)/scale)
		}
	}
	return out
}

func TestSignFlipAgainstEnumeration(t *testing.T) {
	// Datasets of 5 and 7 queries weigh 1/10 and 1/14 per query: integer
	// weights 7 and 5 make the statistic exact.
	k := [][]int64{{3, -1, 4, 1, -5}, {9, 2, -6, 5, 3, -5, 8}}
	want := exactSignFlip(k, []int64{7, 5})
	got := SignFlip(floats(k, 10), Draws)
	se := math.Sqrt(want * (1 - want) / Draws)
	if math.Abs(got-want) > 4*se+1.0/Draws {
		t.Errorf("SignFlip = %v, exact %v (se %.4f)", got, want, se)
	}
}

func TestSignFlipWeightsDatasetsAlike(t *testing.T) {
	// Two queries gaining 1 and ten losing 0.2: the plain mean is 0, the
	// mean with the datasets alike 0.4.
	k := [][]int64{{10, 10}, {-2, -2, -2, -2, -2, -2, -2, -2, -2, -2}}
	want := exactSignFlip(k, []int64{5, 1})
	got := SignFlip(floats(k, 10), Draws)
	se := math.Sqrt(want * (1 - want) / Draws)
	if math.Abs(got-want) > 4*se+1.0/Draws || want > 0.6 {
		t.Errorf("SignFlip = %v, exact %v", got, want)
	}
	if p := signFlip(floats(k, 10), Draws, broken{unweighted: true}); math.Abs(p-want) <= 4*se+1.0/Draws {
		t.Errorf("the variant weighting queries alike gave %v, exact %v", p, want)
	}
}

func TestSignFlipCountsTiesWithinTolerance(t *testing.T) {
	// 0.1 + 0.2 − 0.3 + 0.6 is 0.6 and so is −0.1 − 0.2 + 0.3 + 0.6, but in
	// float64 the second is the smaller: the tolerance counts it as a tie.
	k := [][]int64{{1, 2, -3, 6}}
	want := exactSignFlip(k, []int64{1})
	got := SignFlip(floats(k, 10), Draws)
	se := math.Sqrt(want * (1 - want) / Draws)
	if math.Abs(got-want) > 4*se+1.0/Draws {
		t.Errorf("SignFlip = %v, exact %v", got, want)
	}
	if p := signFlip(floats(k, 10), Draws, broken{noTol: true}); math.Abs(p-want) <= 4*se+1.0/Draws {
		t.Errorf("the variant without tolerance gave %v, exact %v", p, want)
	}
}

func TestSignFlipPValueBounds(t *testing.T) {
	// Forty equal gains: no draw of 1,000 reaches the observed mean but the
	// observed signs themselves, p = 1 / 1001.
	if p := SignFlip([][]float64{repeat(0.01, 40)}, 1000); p != 1.0/1001 {
		t.Errorf("p = %v, want 1/1001", p)
	}
	if p := SignFlip([][]float64{repeat(0, 5)}, 1000); p != 1 {
		t.Errorf("p of no difference = %v, want 1", p)
	}
}

func TestHolm(t *testing.T) {
	for _, tc := range []struct {
		p    []float64
		want []bool
	}{
		// Each is at most 0.05/3, 0.05/2, 0.05: all rejected; Bonferroni
		// (0.05/3 for all) rejects only the first.
		{[]float64{0.04, 0.01, 0.02}, []bool{true, true, true}},
		// Step-down stops at 0.03 > 0.025, so 0.04 is kept although it is
		// below 0.05.
		{[]float64{0.01, 0.04, 0.03}, []bool{true, false, false}},
		{[]float64{0.2, 0.3}, []bool{false, false}},
	} {
		if got := Holm(tc.p, 0.05); !slices.Equal(got, tc.want) {
			t.Errorf("Holm(%v) = %v, want %v", tc.p, got, tc.want)
		}
	}
	if got := holm([]float64{0.04, 0.01, 0.02}, 0.05, broken{bonferroni: true}); slices.Equal(got, []bool{true, true, true}) {
		t.Error("Bonferroni passed for Holm")
	}
}

func TestTauB(t *testing.T) {
	for _, tc := range []struct {
		x, y []float64
		want float64
	}{
		{[]float64{1, 2, 3, 4}, []float64{1, 3, 2, 4}, 4.0 / 6},
		{[]float64{1, 2, 3}, []float64{3, 2, 1}, -1},
		// One tie in each: 4 concordant pairs of 6, over √(5 × 5).
		{[]float64{1, 1, 2, 3}, []float64{1, 2, 2, 3}, 0.8},
	} {
		if got, err := TauB(tc.x, tc.y); err != nil || math.Abs(got-tc.want) > 1e-15 {
			t.Errorf("TauB(%v, %v) = %v, want %v", tc.x, tc.y, got, tc.want)
		}
	}
	if got, err := tauB([]float64{1, 1, 2, 3}, []float64{1, 2, 2, 3}, broken{tauA: true}); err != nil || got == 0.8 {
		t.Error("τ-a passed for τ-b")
	}
	if _, err := TauB([]float64{1}, []float64{1, 2}); err == nil {
		t.Error("scorings of different lengths were accepted")
	}
}

// TestResampleSharesDraws checks that statistics read from one Resample
// are those of separate calls of Bootstrap, each from a fresh generator.
func TestResampleSharesDraws(t *testing.T) {
	s := Strata{{{0.1, 0.5}, {0.2, 0.3}, {0.9, 0.4}}, {{0.4, 0.4}, {0.7, 0.1}}}
	draws := Resample(s, 500)
	stats := []func([]float64) float64{
		func(m []float64) float64 { return m[0] },
		func(m []float64) float64 { return m[1] },
		Range,
	}
	for i, stat := range stats {
		if !slices.Equal(Sorted(draws, stat), Bootstrap(s, 500, stat)) {
			t.Errorf("statistic %d: Resample and Bootstrap differ", i)
		}
	}
}
