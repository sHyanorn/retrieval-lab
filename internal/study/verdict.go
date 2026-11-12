package study

import (
	"math"
	"slices"
)

// Delta is δ of Section 8: the difference in nDCG@10 below which a change
// counts as negligible.
const Delta = 0.005

// Label is the label of a comparison of models at an operating point
// (Section 3).
type Label int

// The labels.
const (
	LabelUnresolved Label = iota
	LabelApart
	LabelEquivalent
)

func (l Label) String() string { return [...]string{"unresolved", "apart", "equivalent"}[l] }

// Verdict is the reading of a hypothesis.
type Verdict int

// The verdicts.
const (
	NotTestable Verdict = iota
	Supported
	NotSupported
	Unresolved
	NotInformative
)

func (v Verdict) String() string {
	return [...]string{"not testable", "supported", "not supported", "unresolved", "not informative"}[v]
}

// Pair is the randomization test of the difference of two variables,
// models I < J: Diff is the mean of the first less that of the second,
// over datasets weighted alike, P its two-sided p-value, and Reject
// whether Holm rejects it in its family.
type Pair struct {
	I, J   int
	Diff   float64
	P      float64
	Reject bool
}

// PairTests tests every pair of the variables of v, in the order (0, 1),
// (0, 2), ..., (1, 2), ...; Reject is left for HolmPairs.
func PairTests(v Strata) []Pair {
	m := v.Means()
	var out []Pair
	for i := range m {
		for j := i + 1; j < len(m); j++ {
			d := make([][]float64, len(v))
			for k, qs := range v {
				d[k] = make([]float64, len(qs))
				for q, x := range qs {
					d[k][q] = x[i] - x[j]
				}
			}
			out = append(out, Pair{I: i, J: j, Diff: m[i] - m[j], P: SignFlip(d, Draws)})
		}
	}
	return out
}

// HolmPairs sets Reject over one Holm family made of all the pairs given,
// at Alpha.
func HolmPairs(family ...[]Pair) {
	var p []float64
	for _, f := range family {
		for _, x := range f {
			p = append(p, x.P)
		}
	}
	r := Holm(p, Alpha)
	for _, f := range family {
		for i := range f {
			f[i].Reject, r = r[0], r[1:]
		}
	}
}

// Comparison is what the label of an operating point is read from: the
// models' averaged losses L̄, the tests of their pairs in one Holm family,
// and the upper end of the 90% interval of their range R̄.
type Comparison struct {
	Means      []float64
	Pairs      []Pair
	RangeBound float64
}

// Compare computes the comparison of per-query losses, loss[d][q][m] the
// loss of model m on query q of dataset d, averaged over the confirmatory
// seeds.
func Compare(loss Strata) Comparison {
	pairs := PairTests(loss)
	HolmPairs(pairs)
	return Comparison{Means: loss.Means(), Pairs: pairs, RangeBound: RangeBound(loss)}
}

// Label is apart if one pair is both rejected and of |Ē| above delta
// (amendment 5), else equivalent if the range bound is below delta, else
// unresolved.
func (c Comparison) Label(delta float64) Label {
	for _, p := range c.Pairs {
		if p.Reject && math.Abs(p.Diff) > delta {
			return LabelApart
		}
	}
	if c.RangeBound < delta {
		return LabelEquivalent
	}
	return LabelUnresolved
}

// H1 reads the label at ef 40 over the confirmatory datasets.
func H1(l Label) Verdict {
	switch l {
	case LabelApart:
		return Supported
	case LabelEquivalent:
		return NotSupported
	}
	return Unresolved
}

// Reading is how a supported H2 reads beside H1's label over S_0.95.
type Reading int

// The readings of a supported H2 (Section 3, H2).
const (
	ReadingNone       Reading = iota // H2 not supported
	ReadingRemoved                   // apart at ef 40, some L̄ outside (−δ, δ): matching removes the dependence
	ReadingSideBySide                // apart at ef 40, every L̄ inside: two facts side by side
	ReadingNeither                   // equivalent at ef 40: no dependence at either point
	ReadingAt95Only                  // unresolved at ef 40: agreement within δ at 0.95 only
)

// H2Result is H2's verdict and what is reported beside it.
type H2Result struct {
	Verdict Verdict
	// Negligible: every L̄(m) at 0.95 inside (−δ, δ); with a supported H2
	// every model's loss is then negligible, and nothing more is read.
	Negligible bool
	Reading    Reading
	Intervals  [][2]float64 // 90% interval of each L̄(m) at 0.95, when supported
}

// H2 reads equal recall 0.95 over S_0.95 of size sets: at95 is the
// comparison there, at40 the comparison at ef 40 over the same datasets,
// loss95 the losses of at95.
func H2(sets int, at95, at40 Comparison, loss95 Strata, delta float64) H2Result {
	if sets < 3 {
		return H2Result{Verdict: NotTestable}
	}
	var r H2Result
	switch at95.Label(delta) {
	case LabelApart:
		r.Verdict = NotSupported
		return r
	case LabelUnresolved:
		r.Verdict = Unresolved
		return r
	}
	r.Verdict = Supported
	r.Negligible = !slices.ContainsFunc(at95.Means, func(m float64) bool { return math.Abs(m) >= delta })
	draws := Resample(loss95, Resamples)
	for m := range at95.Means {
		lo, hi := Interval90(Sorted(draws, func(x []float64) float64 { return x[m] }))
		r.Intervals = append(r.Intervals, [2]float64{lo, hi})
	}
	switch at40.Label(delta) {
	case LabelApart:
		r.Reading = ReadingRemoved
		if r.Negligible {
			r.Reading = ReadingSideBySide
		}
	case LabelEquivalent:
		r.Reading = ReadingNeither
	default:
		r.Reading = ReadingAt95Only
	}
	return r
}

// H3Result is H3's verdict and what is reported beside it.
type H3Result struct {
	Verdict Verdict
	// Exact and Approx test each pair's difference in nDCG@10, each search
	// its own Holm family (amendment 4).
	Exact, Approx []Pair
	Ordered       int // pairs that exact search orders
	Reversed      int // pairs both order, in opposite directions
	Unordered     int // pairs exact search orders and HNSW does not
	Tau           float64
	TauByDataset  []float64
}

// H3 reads equal recall 0.95 over S_0.95 of size sets: exact[d][q][m] is
// the nDCG@10 of exact search, approx[d][q][m] that of HNSW averaged over
// the builds, each at its own ef.
func H3(sets int, exact, approx Strata) (H3Result, error) {
	if sets < 3 {
		return H3Result{Verdict: NotTestable}, nil
	}
	r := h3(PairTests(exact), PairTests(approx), broken{})
	var err error
	if r.Tau, err = TauB(exact.Means(), approx.Means()); err != nil {
		return r, err
	}
	for d := range exact {
		t, err := TauB(Strata{exact[d]}.Means(), Strata{approx[d]}.Means())
		if err != nil {
			return r, err
		}
		r.TauByDataset = append(r.TauByDataset, t)
	}
	return r, nil
}

func h3(e, a []Pair, b broken) H3Result {
	if b.oneFamily {
		HolmPairs(e, a)
	} else {
		HolmPairs(e)
		HolmPairs(a)
	}
	r := H3Result{Exact: e, Approx: a}
	for i := range e {
		if !e[i].Reject {
			continue
		}
		r.Ordered++
		switch {
		case !a[i].Reject:
			r.Unordered++
		case (e[i].Diff > 0) != (a[i].Diff > 0):
			r.Reversed++
		}
	}
	switch {
	case r.Reversed > 0:
		r.Verdict = NotSupported
	case 2*r.Ordered < len(e):
		r.Verdict = NotInformative
	default:
		r.Verdict = Supported
	}
	return r
}

// H4Cell is a model on a dataset: the mean nDCG@10 of each confirmatory
// build at seed 1's ef for recall 0.95, and at ef 40.
type H4Cell struct {
	Model, Dataset string
	Seed1Matched   bool // seed 1 matched at 0.95
	Builds         []float64
	BuildsAt40     []float64
}

// H4Result is H4's verdict over the cells where seed 1 is matched, with
// the range of each such cell, at seed 1's ef and at ef 40.
type H4Result struct {
	Verdict  Verdict
	Cells    []H4Cell
	Ranges   []float64
	Ranges40 []float64
}

// H4 reads the spread of the builds.
func H4(cells []H4Cell, delta float64) H4Result {
	var r H4Result
	datasets := map[string]bool{}
	for _, c := range cells {
		if !c.Seed1Matched {
			continue
		}
		datasets[c.Dataset] = true
		r.Cells = append(r.Cells, c)
		r.Ranges = append(r.Ranges, Range(c.Builds))
		r.Ranges40 = append(r.Ranges40, Range(c.BuildsAt40))
	}
	switch {
	case len(datasets) < 3:
		r.Verdict = NotTestable
	case slices.ContainsFunc(r.Ranges, func(x float64) bool { return x >= delta }):
		r.Verdict = NotSupported
	default:
		r.Verdict = Supported
	}
	return r
}

// H5Result is H5's verdict, the point estimates and their 95% intervals.
type H5Result struct {
	Verdict      Verdict
	LD, LF, Phi  float64
	LDLo, LDHi   float64
	PhiLo, PhiHi float64
}

// H5 reads fusion at ef 100: dense[d][q][m] is the loss of model m's dense
// run on query q of dataset d, fused[d][q][m] that of its fused run, each
// averaged over the confirmatory builds. L_D and L_F sum the models' losses
// averaged over datasets; φ = 1 − L_F / L_D. One set of resamples gives
// both intervals.
func H5(dense, fused Strata) H5Result {
	both := make(Strata, len(dense))
	for d := range dense {
		both[d] = make([][]float64, len(dense[d]))
		for q := range dense[d] {
			both[d][q] = append(slices.Clone(dense[d][q]), fused[d][q]...)
		}
	}
	k := len(dense[0][0])
	ld := func(m []float64) float64 { return sum(m[:k]) }
	phi := func(m []float64) float64 { return 1 - sum(m[k:])/sum(m[:k]) }
	m := both.Means()
	r := H5Result{LD: ld(m), LF: sum(m[k:]), Phi: phi(m)}
	draws := Resample(both, Resamples)
	r.LDLo, r.LDHi = Interval95(Sorted(draws, ld))
	r.PhiLo, r.PhiHi = Interval95(Sorted(draws, phi))
	switch {
	case r.LDLo <= 0:
		r.Verdict = NotTestable
	case r.PhiLo > 0.5:
		r.Verdict = Supported
	case r.PhiHi < 0.5:
		r.Verdict = NotSupported
	default:
		r.Verdict = Unresolved
	}
	return r
}

func sum(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}
	return s
}

// ConfirmatoryDatasets are the datasets of at least 300 test queries
// (Section 3), NQ last.
var ConfirmatoryDatasets = []string{"scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "quora", "nq"}

// Confirmatory returns the confirmatory models and datasets (Section 3,
// with amendment 8): models that failed a check of Section 5 are removed
// first; if at least six of the rest are encoded on NQ, and the tier is not
// tier 1, the datasets are all seven, else NQ leaves; the models are those
// of the rest encoded on every confirmatory dataset. encoded maps a dataset
// to the models encoded on it.
func Confirmatory(models, failed []string, encoded map[string][]string, tier1 bool) (cmodels, datasets []string) {
	var rest []string
	for _, m := range models {
		if !slices.Contains(failed, m) {
			rest = append(rest, m)
		}
	}
	onNQ := 0
	for _, m := range rest {
		if slices.Contains(encoded["nq"], m) {
			onNQ++
		}
	}
	datasets = ConfirmatoryDatasets
	if tier1 || onNQ < 6 {
		datasets = datasets[:len(datasets)-1]
	}
	for _, m := range rest {
		if !slices.ContainsFunc(datasets, func(d string) bool { return !slices.Contains(encoded[d], m) }) {
			cmodels = append(cmodels, m)
		}
	}
	return cmodels, slices.Clone(datasets)
}
