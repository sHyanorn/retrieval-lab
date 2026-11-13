// Command fuse fuses a dataset's BM25 flat run and BGE run as Pyserini's
// pyserini.fusion does with the settings of its BEIR fusion page (D015)
// and writes the RRF and Normalize runs, scores in full. -bm25-depth and
// -dense-depth cut each run before fusion; the study fuses the dense run's
// top 100 with BM25's top 1,000 (D015, docs/PREREGISTRATION.md, Section 6).
// -dense-rank trec ranks the dense run as trec_eval does before the cut,
// as the study does; by default each run's order is its lines.
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
	bm25Depth := flag.Int("bm25-depth", 1000, "hits per query taken from the BM25 run")
	denseDepth := flag.Int("dense-depth", 1000, "hits per query taken from the dense run; the study takes 100")
	denseRank := flag.String("dense-rank", "line", "order of the dense run before the cut: line, its lines; trec, trec_eval's, as the study fuses it")
	flag.Parse()
	if err := run(*bm25, *bge, *rrf, *norm, []int{*bm25Depth, *denseDepth}, *denseRank); err != nil {
		fmt.Fprintln(os.Stderr, "fuse:", err)
		os.Exit(1)
	}
}

func run(bm25, bge, rrf, norm string, depths []int, denseRank string) error {
	if denseRank != "line" && denseRank != "trec" {
		return fmt.Errorf("unknown -dense-rank %q", denseRank)
	}
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
	if denseRank == "trec" {
		runs[1] = fusion.TrecOrder(runs[1])
	}
	// k 1000, rrf_k 60, as on the page; depth 1000 on the page, by default.
	return cmp.Or(write(rrf, fusion.RRF(runs, depths, 60, 1000), "rrf"), write(norm, fusion.Normalize(runs, depths, 1000), "normalize"))
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
