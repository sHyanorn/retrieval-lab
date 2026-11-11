// Package fusion fuses runs as Pyserini's pyserini.fusion does (D015):
// each run is rescored, the scores of a document are summed over the top
// depth of every run, and the top keep are kept, by score and then
// document ID ascending.
package fusion

import (
	"cmp"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// RRF is reciprocal rank fusion: a document scores 1 / (k + rank) in each
// run, rank counting from 1 in the run's order.
func RRF(runs []eval.Run, k, depth, keep int) eval.Run {
	return merge(runs, depth, keep, func(hits []eval.Hit) []float64 {
		s := make([]float64, len(hits))
		for i := range hits {
			s[i] = 1 / float64(k+i+1)
		}
		return s
	})
}

// Normalize maps each run's scores for a query to (s − min) / (max − min),
// or to 1 when max = min, before the sum.
func Normalize(runs []eval.Run, depth, keep int) eval.Run {
	return merge(runs, depth, keep, func(hits []eval.Hit) []float64 {
		lo, hi := hits[0].Score, hits[0].Score
		for _, h := range hits {
			lo, hi = min(lo, h.Score), max(hi, h.Score)
		}
		s := make([]float64, len(hits))
		for i, h := range hits {
			s[i] = 1
			if hi > lo {
				s[i] = (h.Score - lo) / (hi - lo)
			}
		}
		return s
	})
}

func merge(runs []eval.Run, depth, keep int, rescore func([]eval.Hit) []float64) eval.Run {
	sums := map[string]map[string]float64{}
	for _, run := range runs {
		for qid, hits := range run {
			if sums[qid] == nil {
				sums[qid] = map[string]float64{}
			}
			for i, s := range rescore(hits)[:min(depth, len(hits))] {
				sums[qid][hits[i].DocID] += s
			}
		}
	}
	out := eval.Run{}
	for qid, docs := range sums {
		hits := make([]eval.Hit, 0, len(docs))
		for id, s := range docs {
			hits = append(hits, eval.Hit{DocID: id, Score: s})
		}
		slices.SortFunc(hits, func(a, b eval.Hit) int {
			return cmp.Or(cmp.Compare(b.Score, a.Score), strings.Compare(a.DocID, b.DocID))
		})
		out[qid] = hits[:min(keep, len(hits))]
	}
	return out
}
