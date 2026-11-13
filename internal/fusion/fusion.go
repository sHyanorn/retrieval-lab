// Package fusion fuses runs as Pyserini's pyserini.fusion does (D015):
// each run is rescored, the scores of a document are summed over every run,
// and the top keep are kept, by score and then document ID ascending. Each
// run is first cut to its own depth, its first lines per query: a depth at
// least the run's length is Pyserini's depth, and a shorter one fuses the
// dense run's top 100 with BM25's top 1,000 (docs/PREREGISTRATION.md,
// Section 6). A run's order is its lines, unless TrecOrder ranks it first.
package fusion

import (
	"cmp"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// RRF is reciprocal rank fusion: a document scores 1 / (k + rank) in each
// run, rank counting from 1 in the run's order; depths[i] cuts runs[i].
func RRF(runs []eval.Run, depths []int, k, keep int) eval.Run {
	return merge(runs, depths, keep, func(hits []eval.Hit) []float64 {
		s := make([]float64, len(hits))
		for i := range hits {
			s[i] = 1 / float64(k+i+1)
		}
		return s
	})
}

// TrecOrder returns run with each query's hits in trec_eval's order, score
// descending and ties by document ID descending, as eval.Ranked orders
// them. RRF then takes a hit's rank, and a depth its cut, from the scores
// of the run rather than from its lines, so that two runs holding the same
// documents at the same scores fuse alike however each was written. The
// study fuses its dense runs so (docs/DECISIONS.md, D015).
func TrecOrder(run eval.Run) eval.Run {
	out := make(eval.Run, len(run))
	for qid, hits := range run {
		out[qid] = eval.Ranked(hits)
	}
	return out
}

// Normalize maps each run's scores for a query, after the cut, to
// (s − min) / (max − min), or to 1 when max = min, before the sum.
func Normalize(runs []eval.Run, depths []int, keep int) eval.Run {
	return merge(runs, depths, keep, func(hits []eval.Hit) []float64 {
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

func merge(runs []eval.Run, depths []int, keep int, rescore func([]eval.Hit) []float64) eval.Run {
	if len(depths) != len(runs) {
		panic("fusion: one depth per run")
	}
	sums := map[string]map[string]float64{}
	for r, run := range runs {
		for qid, hits := range run {
			if sums[qid] == nil {
				sums[qid] = map[string]float64{}
			}
			hits = hits[:min(depths[r], len(hits))]
			if len(hits) == 0 {
				continue
			}
			for i, s := range rescore(hits) {
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
