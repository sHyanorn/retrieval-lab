// Command dense writes the BGE-base-en-v1.5 run for a BEIR dataset's test
// queries: each query, with the BGE instruction before it, is encoded and
// searched exactly against the document vectors that cmd/embed wrote; the
// top 1000 per query go out in TREC format.
//
// With -ann it also builds an HNSW graph over the same vectors and writes,
// for each ef, the run of the same query vectors searched on the graph
// (D014), and prints the search times as JSON.
package main

import (
	"bufio"
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
	"github.com/sHyanorn/retrieval-lab/internal/hnsw"
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	vectors := flag.String("vectors", "", "document vectors written by cmd/embed")
	model := flag.String("model", "", "directory holding bge-base-en-v1.5-optimized.onnx and tokenizer.json")
	ortLib := flag.String("ort", "", "path to libonnxruntime.so")
	out := flag.String("run", "", "output run file")
	k := flag.Int("k", 1000, "hits per query")
	ann := flag.String("ann", "", "also write the HNSW run for each ef to <ann>-ef<ef>.run")
	flag.Parse()
	if err := run(*data, *vectors, *model, *ortLib, *out, *k, *ann); err != nil {
		fmt.Fprintln(os.Stderr, "dense:", err)
		os.Exit(1)
	}
}

func run(data, vectors, model, ortLib, out string, k int, ann string) error {
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
	if err = cmp.Or(err, f.Close()); err != nil || ann == "" {
		return err
	}
	byQID := make(map[string][]float32, len(queries))
	for qid, text := range queries {
		byQID[qid] = byText[dense.QueryPrefix+text]
	}
	t, err := approximate(x, byQID, k, ann)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(t)
}

type times struct {
	MeanMS   float64 `json:"mean_ms"`
	MedianMS float64 `json:"median_ms"`
}

type efTimes struct {
	EF int `json:"ef"`
	K  int `json:"k"`
	times
}

// approximate builds the graph over x and writes one run per ef of
// dense.EFs, dense.GraphDepth(ef) hits deep. A query's own document is
// skipped inside the search, so each query gets that many lines. It returns
// the time per query on one goroutine, of exact search at depth k and of
// the graph at each ef.
func approximate(x *dense.Index, qvecs map[string][]float32, k int, prefix string) (any, error) {
	qids := slices.Sorted(maps.Keys(qvecs))
	start := time.Now()
	g, err := hnsw.Build(x.Vecs, dense.Dim, dense.GraphConfig)
	if err != nil {
		return nil, err
	}
	build := time.Since(start).Seconds()
	exact := timed(qids, func(qid string) { x.Search(qvecs[qid], k) })
	var graph []efTimes
	for _, ef := range dense.EFs {
		k := dense.GraphDepth(ef)
		hits := make(map[string][]eval.Hit, len(qids))
		t := timed(qids, func(qid string) { hits[qid] = x.SearchGraph(g, qvecs[qid], k, ef, qid) })
		graph = append(graph, efTimes{ef, k, t})
		if err := writeRun(fmt.Sprintf("%s-ef%d.run", prefix, ef), qids, hits, fmt.Sprintf("bge-base-en-v1.5-hnsw-ef%d", ef)); err != nil {
			return nil, err
		}
	}
	return map[string]any{"build_seconds": build, "exact": exact, "hnsw": graph}, nil
}

func timed(qids []string, search func(qid string)) times {
	d := make([]float64, len(qids))
	for i, qid := range qids {
		start := time.Now()
		search(qid)
		d[i] = float64(time.Since(start).Microseconds()) / 1000
	}
	var sum float64
	for _, v := range d {
		sum += v
	}
	slices.Sort(d)
	return times{MeanMS: sum / float64(len(d)), MedianMS: d[len(d)/2]}
}

func writeRun(path string, qids []string, hits map[string][]eval.Hit, tag string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	for _, qid := range qids {
		if err = eval.WriteRun(w, qid, hits[qid], tag); err != nil {
			break
		}
	}
	if err == nil {
		err = w.Flush()
	}
	return cmp.Or(err, f.Close())
}
