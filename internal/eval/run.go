// Package eval reads and writes TREC run files and computes nDCG@k and
// Recall@k exactly as trec_eval does with -c.
package eval

import (
	"bufio"
	"fmt"
	"io"
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
// twice for the same query is an error, as it is in trec_eval.
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
