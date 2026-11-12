package eval

import (
	"bufio"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"
)

// Query holds one query's measures of the study (docs/PREREGISTRATION.md,
// Section 7): nDCG@10 and Recall@100 as Evaluate scores them, the share of
// unjudged documents in the top 10, and the neighbour overlap against the
// exact run as counts, Found10 of Of10 and Found100 of Of100, so that the
// neighbour recall Found/Of can be summed exactly.
type Query struct {
	QID        string
	NDCG10     float64
	Recall100  float64
	Unjudged10 float64
	Found10    int
	Of10       int
	Found100   int
	Of100      int
}

// PerQuery measures every query in qrels, as Evaluate does; with exact
// nil the overlap counts are zero.
func PerQuery(qrels Qrels, run, exact Run) []Query {
	out := make([]Query, 0, len(qrels))
	for _, qid := range slices.Sorted(maps.Keys(qrels)) {
		judged, ranked := qrels[qid], Ranked(run[qid])
		q := Query{
			QID:        qid,
			NDCG10:     NDCG(judged, ranked, 10),
			Recall100:  Recall(judged, ranked, 100),
			Unjudged10: Unjudged(judged, ranked, 10),
		}
		if exact != nil {
			q.Found10, q.Of10 = OverlapCount(exact[qid], run[qid], 10)
			q.Found100, q.Of100 = OverlapCount(exact[qid], run[qid], 100)
		}
		out = append(out, q)
	}
	return out
}

var queryHeader = "qid\tndcg_cut_10\trecall_100\tunjudged_10\tfound_10\tof_10\tfound_100\tof_100"

// WritePerQuery writes the measures as tab-separated lines under a header,
// the real numbers in full.
func WritePerQuery(w io.Writer, qs []Query) error {
	if _, err := fmt.Fprintln(w, queryHeader); err != nil {
		return err
	}
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, q := range qs {
		if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\n", q.QID, f(q.NDCG10), f(q.Recall100), f(q.Unjudged10), q.Found10, q.Of10, q.Found100, q.Of100); err != nil {
			return err
		}
	}
	return nil
}

// ReadPerQuery reads what WritePerQuery writes.
func ReadPerQuery(r io.Reader) ([]Query, error) {
	sc := bufio.NewScanner(r)
	if !sc.Scan() || sc.Text() != queryHeader {
		return nil, fmt.Errorf("per-query measures: header is not %q", queryHeader)
	}
	var out []Query
	for line := 2; sc.Scan(); line++ {
		f := strings.Split(sc.Text(), "\t")
		if len(f) != 8 {
			return nil, fmt.Errorf("per-query measures line %d: %d fields, want 8", line, len(f))
		}
		q := Query{QID: f[0]}
		var err error
		for i, p := range []*float64{&q.NDCG10, &q.Recall100, &q.Unjudged10} {
			if *p, err = strconv.ParseFloat(f[1+i], 64); err != nil {
				return nil, fmt.Errorf("per-query measures line %d: %w", line, err)
			}
		}
		for i, p := range []*int{&q.Found10, &q.Of10, &q.Found100, &q.Of100} {
			if *p, err = strconv.Atoi(f[4+i]); err != nil {
				return nil, fmt.Errorf("per-query measures line %d: %w", line, err)
			}
		}
		out = append(out, q)
	}
	return out, sc.Err()
}
