// Package eval reads and writes TREC run files and computes nDCG@k and
// Recall@k exactly as trec_eval does with -c.
package eval

import (
	"bufio"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Hit is one retrieved document and its score.
type Hit struct {
	DocID string
	Score float64
}

// Run maps a query ID to the documents retrieved for it, in any order.
type Run map[string][]Hit

// ReadRun parses a TREC run file: "qid Q0 docid rank score tag" per line.
// The rank column is ignored, as trec_eval ignores it. A document listed
// twice for the same query is an error, as it is in trec_eval; so is a
// score that is not finite.
func ReadRun(r io.Reader) (Run, error) {
	run := Run{}
	seen := map[[2]string]bool{}
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		f := strings.Fields(sc.Text())
		if len(f) == 0 {
			continue
		}
		if len(f) != 6 {
			return nil, fmt.Errorf("run line %d: want 6 fields, got %d", n, len(f))
		}
		score, err := strconv.ParseFloat(f[4], 64)
		if err != nil {
			return nil, fmt.Errorf("run line %d: %w", n, err)
		}
		// trec_eval compares a NaN as neither above nor below any score, so its
		// order is not well defined.
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return nil, fmt.Errorf("run line %d: score %s is not finite", n, f[4])
		}
		key := [2]string{f[0], f[2]}
		if seen[key] {
			return nil, fmt.Errorf("run line %d: document %s listed twice for query %s", n, f[2], f[0])
		}
		seen[key] = true
		run[f[0]] = append(run[f[0]], Hit{DocID: f[2], Score: score})
	}
	return run, sc.Err()
}

// WriteRun writes the hits of one query in the given order, ranked from 1,
// with scores printed to six decimals as Anserini and Pyserini print them.
func WriteRun(w io.Writer, qid string, hits []Hit, tag string) error {
	for i, h := range hits {
		if _, err := fmt.Fprintf(w, "%s Q0 %s %d %f %s\n", qid, h.DocID, i+1, h.Score, tag); err != nil {
			return err
		}
	}
	return nil
}

// Printed is score as WriteRun prints it and ReadRun reads it back.
func Printed(score float64) float64 {
	v, err := strconv.ParseFloat(fmt.Sprintf("%f", score), 64)
	if err != nil {
		panic(err) // %f of a float64 always parses
	}
	return v
}

// RunQueries runs search on every query in ID order and writes the top k
// hits of each as a TREC run. A hit whose document ID equals the query ID
// is left out after retrieval, as Pyserini's --remove-query does, so such a
// query gets k−1 lines.
func RunQueries(w io.Writer, search func(query string, k int) ([]Hit, error), queries map[string]string, k int, tag string) error {
	for _, qid := range slices.Sorted(maps.Keys(queries)) {
		hits, err := search(queries[qid], k)
		if err != nil {
			return fmt.Errorf("query %s: %w", qid, err)
		}
		hits = slices.DeleteFunc(hits, func(h Hit) bool { return h.DocID == qid })
		if err := WriteRun(w, qid, hits, tag); err != nil {
			return err
		}
	}
	return nil
}
