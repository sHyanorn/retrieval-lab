package eval

import (
	"cmp"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strings"
)

// Qrels maps a query ID to the graded judgments of its documents. Grades
// below zero mean unjudged, as in trec_eval.
type Qrels map[string]map[string]int

// Scores holds one query's measures.
type Scores struct {
	NDCG10    float64
	Recall100 float64
}

// Ranked returns the hits in trec_eval's order: score descending, ties
// broken by document ID descending in byte order.
func Ranked(hits []Hit) []Hit {
	r := slices.Clone(hits)
	slices.SortFunc(r, func(a, b Hit) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		return strings.Compare(b.DocID, a.DocID)
	})
	return r
}

// NDCG is trec_eval's ndcg_cut.k for one query: the gain of a document is
// its grade, rank i (from 1) is discounted by log2(i+1), and the ideal
// ranking holds every judged document with a positive grade. A query with
// no such document scores 0.
func NDCG(judged map[string]int, ranked []Hit, k int) float64 {
	dcg := 0.0
	for i, h := range ranked[:min(k, len(ranked))] {
		if g := judged[h.DocID]; g > 0 {
			dcg += float64(g) / math.Log2(float64(i+2))
		}
	}
	var grades []int
	for _, g := range judged {
		if g > 0 {
			grades = append(grades, g)
		}
	}
	slices.SortFunc(grades, func(a, b int) int { return b - a })
	ideal := 0.0
	for i, g := range grades[:min(k, len(grades))] {
		ideal += float64(g) / math.Log2(float64(i+2))
	}
	if ideal == 0 {
		return 0
	}
	return dcg / ideal
}

// Recall is trec_eval's recall.k for one query: the share of documents with
// a positive grade that appear in the top k. A query with none scores 0.
func Recall(judged map[string]int, ranked []Hit, k int) float64 {
	rel := 0
	for _, g := range judged {
		if g > 0 {
			rel++
		}
	}
	if rel == 0 {
		return 0
	}
	found := 0
	for _, h := range ranked[:min(k, len(ranked))] {
		if judged[h.DocID] > 0 {
			found++
		}
	}
	return float64(found) / float64(rel)
}

// Evaluate scores every query in qrels, as trec_eval -c does: a query
// missing from the run scores 0 and queries missing from qrels are ignored.
func Evaluate(qrels Qrels, run Run) map[string]Scores {
	out := make(map[string]Scores, len(qrels))
	for qid, judged := range qrels {
		ranked := Ranked(run[qid])
		out[qid] = Scores{NDCG10: NDCG(judged, ranked, 10), Recall100: Recall(judged, ranked, 100)}
	}
	return out
}

// Mean averages the scores over all queries, summing in query ID order as
// trec_eval does.
func Mean(scores map[string]Scores) Scores {
	var m Scores
	for _, qid := range slices.Sorted(maps.Keys(scores)) {
		m.NDCG10 += scores[qid].NDCG10
		m.Recall100 += scores[qid].Recall100
	}
	n := float64(len(scores))
	return Scores{NDCG10: m.NDCG10 / n, Recall100: m.Recall100 / n}
}

// WriteTrecEval prints the scores in the layout of
// trec_eval -c -q -m ndcg_cut.10 -m recall.100, so the two outputs can be
// compared as text.
func WriteTrecEval(w io.Writer, scores map[string]Scores) error {
	line := func(measure, qid string, v float64) error {
		_, err := fmt.Fprintf(w, "%-22s\t%s\t%6.4f\n", measure, qid, v)
		return err
	}
	emit := func(qid string, s Scores) error {
		if err := line("recall_100", qid, s.Recall100); err != nil {
			return err
		}
		return line("ndcg_cut_10", qid, s.NDCG10)
	}
	for _, qid := range slices.Sorted(maps.Keys(scores)) {
		if err := emit(qid, scores[qid]); err != nil {
			return err
		}
	}
	return emit("all", Mean(scores))
}
