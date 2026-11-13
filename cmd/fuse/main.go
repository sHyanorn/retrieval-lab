// Command fuse fuses a dataset's BM25 flat run and BGE run as Pyserini's
// pyserini.fusion does with the settings of its BEIR fusion page (D015)
// and writes the RRF and Normalize runs, scores in full. -bm25-depth and
// -dense-depth cut each run before fusion; -dense-rank trec ranks the dense
// run as trec_eval does before the cut; by default each run's order is its
// lines. -study is the study's fusion (docs/PREREGISTRATION.md, Section 6;
// D020): the dense run's top 100, ranked as trec_eval ranks it, with BM25's
// top 1,000, the outputs tagged rrf-study and normalize-study.
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
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
	"github.com/sHyanorn/retrieval-lab/internal/fusion"
)

// files are the two runs fused and the two runs written.
type files struct {
	bm25, dense, rrf, norm string
}

// settings are how the runs are cut and ranked and how the outputs are
// tagged.
type settings struct {
	depths    []int  // BM25's, then the dense run's
	denseRank string // line or trec
	suffix    string // after the tags rrf and normalize
}

// study is the fusion of the study (docs/PREREGISTRATION.md, Section 6).
var study = settings{depths: []int{1000, 100}, denseRank: "trec", suffix: "-study"}

func main() {
	in, s, err := parse(flag.CommandLine, os.Args[1:])
	if err == nil {
		err = run(in, s)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "fuse:", err)
		os.Exit(1)
	}
}

func parse(fs *flag.FlagSet, args []string) (files, settings, error) {
	bm25 := fs.String("bm25", "", "BM25 flat run")
	bge := fs.String("bge", "", "BGE run")
	rrf := fs.String("rrf", "", "output RRF run")
	norm := fs.String("norm", "", "output Normalize run")
	bm25Depth := fs.Int("bm25-depth", 1000, "hits per query taken from the BM25 run")
	denseDepth := fs.Int("dense-depth", 1000, "hits per query taken from the dense run")
	denseRank := fs.String("dense-rank", "line", "order of the dense run before the cut: line, its lines; trec, trec_eval's")
	studyMode := fs.Bool("study", false, "the study's fusion: -bm25-depth 1000 -dense-depth 100 -dense-rank trec, outputs tagged rrf-study and normalize-study; not with those three flags")
	if err := fs.Parse(args); err != nil {
		return files{}, settings{}, err
	}
	in := files{*bm25, *bge, *rrf, *norm}
	if !*studyMode {
		return in, settings{depths: []int{*bm25Depth, *denseDepth}, denseRank: *denseRank}, nil
	}
	var set []string
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "bm25-depth" || f.Name == "dense-depth" || f.Name == "dense-rank" {
			set = append(set, "-"+f.Name)
		}
	})
	if len(set) > 0 {
		return files{}, settings{}, fmt.Errorf("-study sets %s itself", strings.Join(set, ", "))
	}
	return in, study, nil
}

func run(in files, s settings) error {
	if s.denseRank != "line" && s.denseRank != "trec" {
		return fmt.Errorf("unknown -dense-rank %q", s.denseRank)
	}
	var runs []eval.Run
	for _, p := range []string{in.bm25, in.dense} {
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
	if s.denseRank == "trec" {
		runs[1] = fusion.TrecOrder(runs[1])
	}
	// k 1000, rrf_k 60, as on the page; depth 1000 on the page, by default.
	return cmp.Or(write(in.rrf, fusion.RRF(runs, s.depths, 60, 1000), "rrf"+s.suffix),
		write(in.norm, fusion.Normalize(runs, s.depths, 1000), "normalize"+s.suffix))
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
