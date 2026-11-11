// Command bm25 indexes a BEIR dataset (once) and writes the BM25 flat run
// for its test queries: the top 1000 per query, in TREC format. It prints
// the index statistics as JSON. With -bleve-scoring it ranks with Bleve's
// own BM25 instead, for the reference row of the alignment table.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/blevesearch/bleve/v2/search"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/bm25"
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	indexPath := flag.String("index", "", "index directory; built from -data if it does not exist")
	out := flag.String("run", "", "output run file")
	k := flag.Int("k", 1000, "hits per query")
	bleveScoring := flag.Bool("bleve-scoring", false, "rank with Bleve's BM25 (reference only)")
	flag.Parse()
	if err := run(*data, *indexPath, *out, *k, *bleveScoring); err != nil {
		fmt.Fprintln(os.Stderr, "bm25:", err)
		os.Exit(1)
	}
}

func run(data, indexPath, out string, k int, bleveScoring bool) error {
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

	var x *bm25.Index
	if _, err := os.Stat(indexPath); errors.Is(err, os.ErrNotExist) {
		x, err = bm25.Build(indexPath, func(fn func(beir.Doc) error) error { return beir.Corpus(data, fn) })
		if err != nil {
			return err
		}
	} else if x, err = bm25.Open(indexPath); err != nil {
		return err
	}
	defer x.Close() //nolint:errcheck // read-only after the build

	f, err := os.Create(out)
	if err != nil {
		return err
	}
	searchFn, tag := x.Search, "bm25-flat"
	if bleveScoring {
		// Bleve's BM25 parameters are package variables; use Anserini's.
		search.BM25_k1, search.BM25_b = 0.9, 0.4
		searchFn, tag = x.SearchBleve, "bleve-bm25"
	}
	w := bufio.NewWriter(f)
	err = bm25.WriteRun(w, searchFn, queries, k, tag)
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(x.Stats())
}
