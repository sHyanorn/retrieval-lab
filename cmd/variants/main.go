// Command variants ranks the test queries of a BEIR dataset on an index
// made by cmd/bm25, with the Lucene scoring of internal/bm25 and, switched on
// by -variant, the ways in which Bleve's BM25 scores differently: the top
// 1000 per query in TREC format, as cmd/bm25 writes them. It prints the
// statistics of Lucene's scoring and of Bleve's as JSON.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/bm25"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	indexPath := flag.String("index", "", "index directory made by cmd/bm25")
	out := flag.String("run", "", "output run file")
	k := flag.Int("k", 1000, "hits per query")
	variant := flag.String("variant", "", "Bleve's differences to switch on, comma-separated: sqrt-tf, avgdl, idf-twice, coord, k1, exact-length, bleve-length, all-docs, or all; none gives cmd/bm25's run")
	flag.Parse()
	if err := run(*data, *indexPath, *out, *k, *variant); err != nil {
		fmt.Fprintln(os.Stderr, "variants:", err)
		os.Exit(1)
	}
}

func run(data, indexPath, out string, k int, variant string) error {
	var v Variant
	tag := "bm25-flat"
	if variant != "" {
		var err error
		if v, err = parseVariant(variant); err != nil {
			return err
		}
		tag = "bm25-" + strings.ReplaceAll(variant, ",", "-")
	}
	qrels, err := beir.Qrels(data, "test")
	if err != nil {
		return err
	}
	all, err := beir.Queries(data)
	if err != nil {
		return err
	}
	queries := make(map[string]string, len(qrels))
	for qid := range qrels {
		text, ok := all[qid]
		if !ok {
			return fmt.Errorf("query %s is in the qrels but not in queries.jsonl", qid)
		}
		queries[qid] = text
	}

	x, err := open(indexPath)
	if err != nil {
		return err
	}
	defer x.close() //nolint:errcheck // read-only
	s, err := x.scoringFor(v)
	if err != nil {
		return err
	}
	bs, err := x.bleveStats()
	if err != nil {
		return err
	}

	f, err := os.Create(out)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	err = eval.RunQueries(w, func(query string, k int) ([]eval.Hit, error) { return x.search(query, k, s) }, queries, k, tag)
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(struct {
		bm25.Stats
		Bleve bleveStats `json:"bleve"`
	}{x.stats, bs})
}
