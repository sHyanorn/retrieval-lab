// Package study holds the readings of the registered study
// (docs/PREREGISTRATION.md and docs/PREREGISTRATION-AMENDMENTS.md): the
// recall operating points, the statistics and the verdicts.
package study

import (
	"errors"
	"fmt"
	"math/big"
	"slices"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// MeanRecall10 is the neighbour recall@10 of a run averaged over its
// queries, as an exact fraction: the mean of Found10 / Of10.
func MeanRecall10(qs []eval.Query) (*big.Rat, error) {
	if len(qs) == 0 {
		return nil, errors.New("neighbour recall of no queries")
	}
	sum := new(big.Rat)
	for _, q := range qs {
		if q.Of10 <= 0 || q.Found10 < 0 || q.Found10 > q.Of10 {
			return nil, fmt.Errorf("query %s: neighbour overlap %d of %d", q.QID, q.Found10, q.Of10)
		}
		sum.Add(sum, big.NewRat(int64(q.Found10), int64(q.Of10)))
	}
	return sum.Quo(sum, big.NewRat(int64(len(qs)), 1)), nil
}

// Point is the mean neighbour recall@10 of one build at one ef.
type Point struct {
	EF     int
	Recall *big.Rat
}

// Status says whether a build is matched at a recall target.
type Status int

// The statuses of a build at a recall target r (Section 3, Operating
// points, with amendment 6).
const (
	Matched    Status = iota // the smallest ef reaching r overshoots by at most 0.01
	BelowReach               // the grid's smallest ef already overshoots by more than 0.01
	Overshoot                // a larger ef reaches r, overshooting by more than 0.01
	Unreached                // no ef of the grid reaches r
)

func (s Status) String() string {
	return [...]string{"matched", "below reach", "overshoot", "unreached"}[s]
}

// Selection is the ef chosen for a build at a recall target and the
// recall it reaches; for an unreached target, the grid's largest ef.
type Selection struct {
	EF      int
	Reached *big.Rat
	Status  Status
}

// Tolerance is the overshoot of a matched build: a − r ≤ 0.01.
var Tolerance = big.NewRat(1, 100)

// Select chooses the smallest ef whose recall is at least r, comparing
// exactly.
func Select(points []Point, r *big.Rat) (Selection, error) {
	ps, err := sortPoints(points)
	if err != nil {
		return Selection{}, err
	}
	for i, p := range ps {
		if p.Recall.Cmp(r) < 0 {
			continue
		}
		s := Selection{EF: p.EF, Reached: p.Recall, Status: Matched}
		if new(big.Rat).Sub(p.Recall, r).Cmp(Tolerance) > 0 {
			s.Status = Overshoot
			if i == 0 {
				s.Status = BelowReach
			}
		}
		return s, nil
	}
	last := ps[len(ps)-1]
	return Selection{EF: last.EF, Reached: last.Recall, Status: Unreached}, nil
}

// Closest chooses the ef whose recall is closest to r, the smaller of two
// equally close, comparing exactly (amendment 10).
func Closest(points []Point, r *big.Rat) (Point, error) {
	ps, err := sortPoints(points)
	if err != nil {
		return Point{}, err
	}
	best, dist := ps[0], new(big.Rat).Abs(new(big.Rat).Sub(ps[0].Recall, r))
	for _, p := range ps[1:] {
		if d := new(big.Rat).Abs(new(big.Rat).Sub(p.Recall, r)); d.Cmp(dist) < 0 {
			best, dist = p, d
		}
	}
	return best, nil
}

// sortPoints returns the points in ascending ef, each ef once.
func sortPoints(points []Point) ([]Point, error) {
	if len(points) == 0 {
		return nil, errors.New("no points to select from")
	}
	ps := slices.SortedFunc(slices.Values(points), func(a, b Point) int { return a.EF - b.EF })
	for i, p := range ps {
		if i > 0 && p.EF == ps[i-1].EF {
			return nil, fmt.Errorf("ef %d given twice", p.EF)
		}
	}
	return ps, nil
}

// AllMatched reports whether a model is matched on a dataset: every one
// of its builds is.
func AllMatched(builds []Selection) bool {
	if len(builds) == 0 {
		return false
	}
	for _, b := range builds {
		if b.Status != Matched {
			return false
		}
	}
	return true
}

// RecallSet is S_r: the datasets, in the order given, on which every model
// is matched.
func RecallSet(models, datasets []string, matched func(model, dataset string) bool) []string {
	var out []string
	for _, d := range datasets {
		if !slices.ContainsFunc(models, func(m string) bool { return !matched(m, d) }) {
			out = append(out, d)
		}
	}
	return out
}
