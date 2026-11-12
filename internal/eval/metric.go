package eval

import (
	"cmp"
	"fmt"
	"io"
	"maps"
	"math"
	"math/rand/v2"
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
			dcg += float64(g) / discount(i)
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
		ideal += float64(g) / discount(i)
	}
	if ideal == 0 {
		return 0
	}
	return dcg / ideal
}

// log2 holds log2(i + 2) for i = 0..9, the discounts of ranks 1 to 10, as
// C's libm computes them (correctly rounded). Go's math.Log2 is one unit in
// the last place lower for 3 and 9, which can move trec_eval's fourth
// decimal.
var log2 = [...]float64{
	0x1p+0, 0x1.95c01a39fbd68p+0, 0x1p+1, 0x1.2934f0979a371p+1, 0x1.4ae00d1cfdeb4p+1,
	0x1.675767f54042dp+1, 0x1.8p+1, 0x1.95c01a39fbd68p+1, 0x1.a934f0979a371p+1, 0x1.bacea7c065d42p+1,
}

// discount is the divisor of the gain at rank i + 1. Beyond rank 10 it uses
// math.Log2, which can differ from C's in the last place.
func discount(i int) float64 {
	if i < len(log2) {
		return log2[i]
	}
	return math.Log2(float64(i + 2))
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

// Overlap is the share of exact's top k that approx's top k holds, both
// ranked as trec_eval ranks them and cut at rank k: the recall of an
// approximate search against the exact one, as M3's tables count it.
// exact must not be empty.
func Overlap(exact, approx []Hit, k int) float64 {
	found, of := overlapCount(exact, approx, k, byRank)
	return float64(found) / float64(of)
}

// OverlapCount returns the study's counts of neighbour recall@k
// (docs/PREREGISTRATION-AMENDMENTS.md, 15), both runs ranked as trec_eval
// ranks them: of approx's top k, the documents exact holds at a score above
// its k-th, plus those it holds at exactly the k-th score, up to the number
// of these in exact's top k; and the size of exact's top k. Which of the
// documents tied at the cut a run holds does not count against it. The
// scores are compared as given; read from run files, they are compared as
// the files print them.
func OverlapCount(exact, approx []Hit, k int) (found, of int) {
	return overlapCount(exact, approx, k, tiesCapped)
}

// overlapRule is how overlapCount treats documents tied at exact's k-th
// score.
type overlapRule int

const (
	tiesCapped overlapRule = iota // OverlapCount's rule
	byRank                        // only exact's top k by rank, as M3's Overlap
	uncapped                      // every tied document, however many
)

func overlapCount(exact, approx []Hit, k int, rule overlapRule) (found, of int) {
	e, a := Ranked(exact), Ranked(approx)
	of = min(k, len(e))
	if of == 0 {
		return 0, 0
	}
	a = a[:min(k, len(a))]
	in := func(set []Hit, id string) bool {
		return slices.ContainsFunc(set, func(x Hit) bool { return x.DocID == id })
	}
	if rule == byRank {
		for _, h := range a {
			if in(e[:of], h.DocID) {
				found++
			}
		}
		return found, of
	}
	last := e[of-1].Score
	above := 0 // exact's documents above the k-th score, all in its top k
	for above < of && e[above].Score > last {
		above++
	}
	end := of // exact's documents tied at the k-th score are e[above:end]
	for end < len(e) && e[end].Score == last {
		end++
	}
	tied := 0
	for _, h := range a {
		switch {
		case in(e[:above], h.DocID):
			found++
		case in(e[above:end], h.DocID):
			tied++
		}
	}
	if rule == tiesCapped {
		tied = min(tied, of-above)
	}
	return found + tied, of
}

// Unjudged is the share of the top k of ranked that qrels do not judge,
// a missing judgment or a grade below zero counting as none, over k as
// trec_eval's P@k divides.
func Unjudged(judged map[string]int, ranked []Hit, k int) float64 {
	n := 0
	for _, h := range ranked[:min(k, len(ranked))] {
		if g, ok := judged[h.DocID]; !ok || g < 0 {
			n++
		}
	}
	return float64(n) / float64(k)
}

// PairedInterval is the 95% interval of the mean of a[i] − b[i] from a
// paired bootstrap: samples draws of len(a) indices with replacement, from
// a fixed seed; it returns the 251st and the 9,750th of 10,000 sorted means,
// and the same quantiles for other sample counts.
func PairedInterval(a, b []float64, samples int) (lo, hi float64) {
	r := rand.New(rand.NewPCG(1, 2))
	means := make([]float64, samples)
	for s := range means {
		var sum float64
		for range a {
			i := r.IntN(len(a))
			sum += a[i] - b[i]
		}
		means[s] = sum / float64(len(a))
	}
	slices.Sort(means)
	return means[samples/40], means[samples-samples/40-1]
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
