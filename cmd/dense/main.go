// Command dense writes the BGE-base-en-v1.5 run for a BEIR dataset's test
// queries: each query, with the BGE instruction before it, is encoded and
// searched exactly against the document vectors that cmd/embed wrote; the
// top 1000 per query go out in TREC format.
package main

import (
	"bufio"
	"cmp"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	vectors := flag.String("vectors", "", "document vectors written by cmd/embed")
	model := flag.String("model", "", "directory holding bge-base-en-v1.5-optimized.onnx and tokenizer.json")
	ortLib := flag.String("ort", "", "path to libonnxruntime.so")
	out := flag.String("run", "", "output run file")
	k := flag.Int("k", 1000, "hits per query")
	flag.Parse()
	if err := run(*data, *vectors, *model, *ortLib, *out, *k); err != nil {
		fmt.Fprintln(os.Stderr, "dense:", err)
		os.Exit(1)
	}
}

func run(data, vectors, model, ortLib, out string, k int) error {
	qrels, err := beir.Qrels(data, "test")
	if err != nil {
		return err
	}
	all, err := beir.Queries(data)
	if err != nil {
		return err
	}
	queries := make(map[string]string, len(qrels))
	var texts []string
	for qid := range qrels {
		text, ok := all[qid]
		if !ok {
			return fmt.Errorf("query %s is in the qrels but not in queries.jsonl", qid)
		}
		queries[qid] = text
		texts = append(texts, dense.QueryPrefix+text)
	}

	if err := dense.Init(ortLib); err != nil {
		return err
	}
	e, err := dense.NewEncoder(filepath.Join(model, "bge-base-en-v1.5-optimized.onnx"), filepath.Join(model, "tokenizer.json"), 512)
	if err != nil {
		return err
	}
	defer e.Close() //nolint:errcheck // the run is written by then
	qvecs, err := e.EncodeAll(texts, runtime.GOMAXPROCS(0), nil)
	if err != nil {
		return err
	}
	byText := make(map[string][]float32, len(texts))
	for i, t := range texts {
		byText[t] = qvecs[i*dense.Dim : (i+1)*dense.Dim]
	}

	ids, vecs, dim, err := dense.ReadFaissFlat(vectors)
	if err != nil {
		return err
	}
	if dim != dense.Dim {
		return fmt.Errorf("%s: vectors of dimension %d, want %d", vectors, dim, dense.Dim)
	}
	x := &dense.Index{IDs: ids, Vecs: vecs}
	search := func(query string, k int) ([]eval.Hit, error) {
		return x.Search(byText[dense.QueryPrefix+query], k), nil
	}

	f, err := os.Create(out)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	err = eval.RunQueries(w, search, queries, k, "bge-base-en-v1.5")
	if err == nil {
		err = w.Flush()
	}
	return cmp.Or(err, f.Close())
}
