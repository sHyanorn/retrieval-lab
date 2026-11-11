// Command fuse fuses a dataset's BM25 flat run and BGE run as Pyserini's
// pyserini.fusion does with the settings of its BEIR fusion page (D015)
// and writes the RRF and Normalize runs, scores in full.
package main

import (
	"bufio"
	"cmp"
	"flag"
	"fmt"
	"maps"
	"os"
	"slices"
	"strconv"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
	"github.com/sHyanorn/retrieval-lab/internal/fusion"
)

func main() {
	bm25 := flag.String("bm25", "", "BM25 flat run")
	bge := flag.String("bge", "", "BGE run")
	rrf := flag.String("rrf", "", "output RRF run")
	norm := flag.String("norm", "", "output Normalize run")
	flag.Parse()
	if err := run(*bm25, *bge, *rrf, *norm); err != nil {
		fmt.Fprintln(os.Stderr, "fuse:", err)
		os.Exit(1)
	}
}

func run(bm25, bge, rrf, norm string) error {
	var runs []eval.Run
	for _, p := range []string{bm25, bge} {
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		r, err := eval.ReadRun(bufio.NewReader(f))
		if err = cmp.Or(err, f.Close()); err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		runs = append(runs, r)
	}
	// k 1000, depth 1000, rrf_k 60, as on the page.
	return cmp.Or(write(rrf, fusion.RRF(runs, 60, 1000, 1000), "rrf"), write(norm, fusion.Normalize(runs, 1000, 1000), "normalize"))
}

func write(path string, r eval.Run, tag string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
lines:
	for _, qid := range slices.Sorted(maps.Keys(r)) {
		for i, h := range r[qid] {
			if _, err = fmt.Fprintf(w, "%s Q0 %s %d %s %s\n", qid, h.DocID, i+1, strconv.FormatFloat(h.Score, 'g', -1, 64), tag); err != nil {
				break lines
			}
		}
	}
	if err == nil {
		err = w.Flush()
	}
	return cmp.Or(err, f.Close())
}
