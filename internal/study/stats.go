package study

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
)

// The constants of Section 8 and amendments 1 to 3.
const (
	Seed      = 20261111 // first word of the PCG of every interval and test; the second is 0
	Resamples = 10000    // bootstrap resamples of an interval
	Draws     = 100000   // sign draws of a randomization test
	Alpha     = 0.05     // family-wise level of the tests
	RelTol    = 1e-12    // a draw this close to the observed |statistic| counts as at least it
)

// newRand is the generator of one interval or test, started afresh
// (amendment 1).
func newRand() *rand.Rand { return rand.New(rand.NewPCG(Seed, 0)) }

// broken selects deliberately wrong variants, for the tests that must
// catch them.
type broken struct {
	pooled     bool // resample all queries together, not within each dataset
	unweighted bool // weight every query alike, not every dataset alike
	bonferroni bool // compare every p-value with alpha / m
	tauA       bool // divide by all pairs, ignoring ties
	offByOne   bool // take the percentile one place inward
	noTol      bool // count only draws at least the observed |statistic|
	oneFamily  bool // H3: one Holm family of both searches
}

// Strata holds per-query values by dataset: Strata[d][q][j] is variable j
// (a model, say) on query q of dataset d. Every query of every dataset has
// the same variables.
type Strata [][][]float64

// Means is the average over datasets, each weighted alike, of each
// variable's mean over the dataset's queries: L̄ of Section 3 when the
// variables are per-query losses.
func (s Strata) Means() []float64 {
	return s.means(nil, broken{})
}

func (s Strata) vars() int {
	if len(s) == 0 || len(s[0]) == 0 {
		return 0
	}
	return len(s[0][0])
}

// means averages over each dataset's queries, those drawn by draw if it is
// not nil, and then over datasets.
func (s Strata) means(draw func(n int) int, b broken) []float64 {
	j := s.vars()
	out := make([]float64, j)
	if b.pooled {
		var all [][]float64
		for _, qs := range s {
			all = append(all, qs...)
		}
		for range all {
			q := all[draw(len(all))]
			for v := range j {
				out[v] += q[v]
			}
		}
		for v := range j {
			out[v] /= float64(len(all))
		}
		return out
	}
	sum := make([]float64, j)
	total := 0
	for _, qs := range s {
		clear(sum)
		for i := range qs {
			q := qs[i]
			if draw != nil {
				q = qs[draw(len(qs))]
			}
			for v := range j {
				sum[v] += q[v]
			}
		}
		for v := range j {
			if b.unweighted {
				out[v] += sum[v]
			} else {
				out[v] += sum[v] / float64(len(qs))
			}
		}
		total += len(qs)
	}
	for v := range j {
		if b.unweighted {
			out[v] /= float64(total)
		} else {
			out[v] /= float64(len(s))
		}
	}
	return out
}

// Bootstrap draws b resamples of the queries, within each dataset and as
// many as it holds, and returns stat of each resample's Means, sorted. The
// draws come from a fresh generator: resample by resample, dataset by
// dataset in the order given, query by query, r.IntN(n).
func Bootstrap(s Strata, b int, stat func(means []float64) float64) []float64 {
	return bootstrap(s, b, stat, broken{})
}

func bootstrap(s Strata, b int, stat func([]float64) float64, br broken) []float64 {
	return Sorted(resample(s, b, br), stat)
}

// Resample returns the Means of each of b resamples, in the order that
// Bootstrap draws them: Bootstrap(s, b, stat) is Sorted(Resample(s, b),
// stat). Intervals of several statistics from one Resample are the
// intervals that separate calls of Bootstrap give, each from a fresh
// generator, since the draws do not depend on the statistic.
func Resample(s Strata, b int) [][]float64 { return resample(s, b, broken{}) }

func resample(s Strata, b int, br broken) [][]float64 {
	r := newRand()
	out := make([][]float64, b)
	for i := range out {
		out[i] = s.means(r.IntN, br)
	}
	return out
}

// Sorted returns stat of each resample's Means, sorted.
func Sorted(draws [][]float64, stat func(means []float64) float64) []float64 {
	out := make([]float64, len(draws))
	for i, m := range draws {
		out[i] = stat(m)
	}
	slices.Sort(out)
	return out
}

// Interval90 is the 90% percentile interval of sorted resampled values:
// counted from 0, the ⌊B/20⌋-th and the (B − ⌊B/20⌋ − 1)-th (amendment 2).
// Its upper end is the one-sided bound at 0.05.
func Interval90(sorted []float64) (lo, hi float64) { return percentile(sorted, 20, broken{}) }

// Interval95 is the 95% interval: the ⌊B/40⌋-th and the (B − ⌊B/40⌋ − 1)-th,
// as eval.PairedInterval reads it.
func Interval95(sorted []float64) (lo, hi float64) { return percentile(sorted, 40, broken{}) }

func percentile(sorted []float64, div int, b broken) (lo, hi float64) {
	n := len(sorted)
	t := n / div
	if b.offByOne {
		t++
	}
	return sorted[t], sorted[n-t-1]
}

// Range is the largest value less the smallest: R̄ of Section 3 when v
// holds the models' averaged losses.
func Range(v []float64) float64 {
	return slices.Max(v) - slices.Min(v)
}

// RangeBound is the upper end of the 90% interval of the range of Means,
// over Resamples stratified resamples: the bound that the label
// equivalent compares with δ.
func RangeBound(s Strata) float64 {
	_, hi := Interval90(Bootstrap(s, Resamples, Range))
	return hi
}

// SignFlip is the two-sided p-value of the stratified randomization test
// of Section 3: d[k][q] is the difference on query q of dataset k, weighted
// by 1 / (number of datasets × the dataset's number of queries). Each of
// draws draws flips each difference's sign with probability 1/2, taking the
// signs, dataset by dataset and query by query, from the bits of the
// generator's 64-bit words, lowest first, a new word every 64 queries and
// at the start of each draw. p = (b + 1) / (draws + 1), b counting the draws
// whose |statistic| is at least the observed one within RelTol
// (amendment 3).
func SignFlip(d [][]float64, draws int) float64 {
	return signFlip(d, draws, broken{})
}

func signFlip(d [][]float64, draws int, br broken) float64 {
	total := 0
	for _, qs := range d {
		total += len(qs)
	}
	// w holds each query's weighted difference, in draw order.
	w := make([]float64, 0, total)
	for _, qs := range d {
		f := 1 / (float64(len(d)) * float64(len(qs)))
		if br.unweighted {
			f = 1 / float64(total)
		}
		for _, x := range qs {
			w = append(w, f*x)
		}
	}
	var obs float64
	for _, x := range w {
		obs += x
	}
	obs = math.Abs(obs)
	limit := obs - RelTol*obs
	if br.noTol {
		limit = obs
	}
	r := newRand()
	b := 0
	for range draws {
		var t float64
		var bits uint64
		for i, x := range w {
			if i%64 == 0 {
				bits = r.Uint64()
			}
			if bits&1 == 1 {
				x = -x
			}
			bits >>= 1
			t += x
		}
		if math.Abs(t) >= limit {
			b++
		}
	}
	return float64(b+1) / float64(draws+1)
}

// Holm reports which of the p-values Holm's step-down procedure rejects at
// family-wise level alpha: in ascending order, the i-th (from 0) of m is
// rejected while it is at most alpha / (m − i).
func Holm(p []float64, alpha float64) []bool {
	return holm(p, alpha, broken{})
}

func holm(p []float64, alpha float64, b broken) []bool {
	m := len(p)
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		switch {
		case p[a] < p[b]:
			return -1
		case p[a] > p[b]:
			return 1
		}
		return 0
	})
	out := make([]bool, m)
	for i, k := range idx {
		limit := alpha / float64(m-i)
		if b.bonferroni {
			limit = alpha / float64(m)
		}
		if p[k] > limit {
			break
		}
		out[k] = true
	}
	return out
}

// TauB is Kendall's τ-b between two scorings of the same items:
// (concordant − discordant) / √((n₀ − n₁)(n₀ − n₂)), n₀ the pairs, n₁ and n₂
// the pairs tied in x and in y. It is NaN when either scoring ties every
// pair.
func TauB(x, y []float64) (float64, error) {
	return tauB(x, y, broken{})
}

func tauB(x, y []float64, b broken) (float64, error) {
	if len(x) != len(y) {
		return 0, errors.New("τ-b of scorings of different lengths")
	}
	var c, d, n0, n1, n2 int
	for i := range x {
		for j := i + 1; j < len(x); j++ {
			n0++
			sx, sy := sign(x[i]-x[j]), sign(y[i]-y[j])
			if sx == 0 {
				n1++
			}
			if sy == 0 {
				n2++
			}
			switch sx * sy {
			case 1:
				c++
			case -1:
				d++
			}
		}
	}
	if b.tauA {
		return float64(c-d) / float64(n0), nil
	}
	return float64(c-d) / math.Sqrt(float64(n0-n1)*float64(n0-n2)), nil
}

func sign(v float64) int {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	}
	return 0
}
