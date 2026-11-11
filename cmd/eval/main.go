// Command eval scores a TREC run against a BEIR dataset's qrels and prints
// per-query and mean nDCG@10 and Recall@100 in the layout of
// trec_eval -c -q -m ndcg_cut.10 -m recall.100.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	split := flag.String("split", "test", "qrels split")
	runPath := flag.String("run", "", "TREC run file")
	flag.Parse()
	if err := run(*data, *split, *runPath); err != nil {
		fmt.Fprintln(os.Stderr, "eval:", err)
		os.Exit(1)
	}
}

func run(data, split, runPath string) error {
	qrels, err := beir.Qrels(data, split)
	if err != nil {
		return err
	}
	f, err := os.Open(runPath)
	if err != nil {
		return err
	}
	defer f.Close() //nolint:errcheck // read-only
	r, err := eval.ReadRun(bufio.NewReader(f))
	if err != nil {
		return err
	}
	w := bufio.NewWriter(os.Stdout)
	if err := eval.WriteTrecEval(w, eval.Evaluate(qrels, r)); err != nil {
		return err
	}
	return w.Flush()
}
