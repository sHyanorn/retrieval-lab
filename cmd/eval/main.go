// Command eval scores a TREC run against a BEIR dataset's qrels and prints
// per-query and mean nDCG@10 and Recall@100 in the layout of
// trec_eval -c -q -m ndcg_cut.10 -m recall.100.
//
// With -per-query it prints instead, per query, the measures of the study
// (eval.WritePerQuery): nDCG@10, Recall@100, the share of unjudged documents
// in the top 10 and, against the exact run -exact if given, the neighbour
// overlap at 10 and 100.
package main

import (
	"bufio"
	"errors"
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
	perQuery := flag.Bool("per-query", false, "print the study's per-query measures as tab-separated lines")
	exactPath := flag.String("exact", "", "exact run for the neighbour overlap, with -per-query")
	flag.Parse()
	if err := run(*data, *split, *runPath, *exactPath, *perQuery); err != nil {
		fmt.Fprintln(os.Stderr, "eval:", err)
		os.Exit(1)
	}
}

func run(data, split, runPath, exactPath string, perQuery bool) error {
	if exactPath != "" && !perQuery {
		return errors.New("-exact is read only with -per-query")
	}
	qrels, err := beir.Qrels(data, split)
	if err != nil {
		return err
	}
	r, err := readRun(runPath)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(os.Stdout)
	if perQuery {
		var exact eval.Run
		if exactPath != "" {
			if exact, err = readRun(exactPath); err != nil {
				return err
			}
		}
		err = eval.WritePerQuery(w, eval.PerQuery(qrels, r, exact))
	} else {
		err = eval.WriteTrecEval(w, eval.Evaluate(qrels, r))
	}
	if err != nil {
		return err
	}
	return w.Flush()
}

func readRun(path string) (eval.Run, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only
	return eval.ReadRun(bufio.NewReader(f))
}
