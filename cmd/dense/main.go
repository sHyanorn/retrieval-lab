// Command dense writes the run for a BEIR dataset's test queries: each
// query vector is searched exactly against the document vectors that
// cmd/embed or encode/encode.py wrote, and the top 1000 per query go out in
// TREC format. The query vectors are read from -queries, as encode/encode.py
// writes them, or else each query, with the BGE instruction before it, is
// encoded with BGE-base-en-v1.5 on ONNX Runtime.
//
// With -ann it also builds an HNSW graph over the same vectors, with -m
// links and -efc efConstruction, inserting the documents in the order of
// build seed -seed (docs/PREREGISTRATION.md, Section 6), and writes, for each
// ef, the run of the same query vectors searched on the graph (D014), and
// prints the search times as JSON. -run may then be left out to skip the
// exact run.
package main

import (
	"bufio"
	"cmp"
	"encoding/json"
	"errors"
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
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	vectors := flag.String("vectors", "", "document vectors written by cmd/embed")
	model := flag.String("model", "", "directory holding bge-base-en-v1.5-optimized.onnx and tokenizer.json")
	ortLib := flag.String("ort", "", "path to libonnxruntime.so")
	out := flag.String("run", "", "output run file")
	k := flag.Int("k", 1000, "hits per query")
	ann := flag.String("ann", "", "also write the HNSW run for each ef to <ann>-ef<ef>.run")
	queryVecs := flag.String("queries", "", "test query vectors written by encode/encode.py, in place of encoding the queries with -model")
	tag := flag.String("tag", "bge-base-en-v1.5", "run tag")
	seed := flag.Uint64("seed", 0, "build seed of the graph: 0 inserts in corpus order")
	m := flag.Int("m", dense.GraphConfig.M, "links per node of the graph")
	efc := flag.Int("efc", dense.GraphConfig.EfConstruction, "efConstruction of the graph")
	grid := flag.String("grid", "m3", "ef values searched: m3, the eight of M3, or study, the 65 of the study")
	flag.Parse()
	g := graph{seed: *seed, m: *m, efc: *efc, efs: dense.EFs}
	switch *grid {
	case "m3":
	case "study":
		g.efs, g.study = dense.StudyEFs, true
	default:
		fmt.Fprintf(os.Stderr, "dense: unknown -grid %q\n", *grid)
		os.Exit(2)
	}
	if err := run(*data, *vectors, *queryVecs, *model, *ortLib, *out, *tag, *k, *ann, g); err != nil {
		fmt.Fprintln(os.Stderr, "dense:", err)
		os.Exit(1)
	}
}

// graph is the build of the HNSW graph.
type graph struct {
	seed   uint64
	m, efc int
	efs    []int
	study  bool // the study's grid: runs cut as trec_eval ranks them
}

func run(data, vectors, queryVecs, model, ortLib, out, tag string, k int, ann string, gr graph) error {
	if out == "" && ann == "" {
		return errors.New("neither -run nor -ann given")
	}
	qrels, err := beir.Qrels(data, "test")
	if err != nil {
		return err
	}
	ids, vecs, dim, err := dense.ReadFaissFlat(vectors)
	if err != nil {
		return err
	}
	qids := slices.Sorted(maps.Keys(qrels))
	var byQID map[string][]float32
	if queryVecs != "" {
		byQID, err = readQueryVectors(queryVecs, qids, dim)
	} else {
		byQID, err = encodeQueries(data, qids, model, ortLib, dim)
	}
	if err != nil {
		return err
	}
	x := &dense.Index{IDs: ids, Vecs: vecs}
	// RunQueries hands search each query's text; here the text is its ID.
	queries := make(map[string]string, len(qids))
	for _, qid := range qids {
		queries[qid] = qid
	}
	search := func(qid string, k int) ([]eval.Hit, error) {
		return x.Search(byQID[qid], k), nil
	}

	if out != "" {
		f, err := os.Create(out)
		if err != nil {
			return err
		}
		w := bufio.NewWriter(f)
		err = eval.RunQueries(w, search, queries, k, tag)
		if err == nil {
			err = w.Flush()
		}
		if err = cmp.Or(err, f.Close()); err != nil {
			return err
		}
	}
	if ann == "" {
		return nil
	}
	t, err := approximate(x, byQID, k, ann, tag, gr, out != "")
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

// approximate builds the graph of gr over x and writes one run per ef of
// gr.efs, dense.GraphDepth(ef) hits deep. A query's own document is
// skipped inside the search, so each query gets that many lines. On the
// study's grid a run holds the top of all ef documents the search found, as
// cut cuts them; on M3's, the top of the graph's order. It returns the time
// per query on one goroutine of the graph search at each ef and, with
// timeExact, of exact search at depth k. On the study's grid that time
// includes scoring all ef documents, not GraphDepth(ef) as on M3's.
func approximate(x *dense.Index, qvecs map[string][]float32, k int, prefix, tag string, gr graph, timeExact bool) (any, error) {
	qids := slices.Sorted(maps.Keys(qvecs))
	start := time.Now()
	g, err := x.BuildGraph(gr.m, gr.efc, gr.seed)
	if err != nil {
		return nil, err
	}
	build := time.Since(start).Seconds()
	out := map[string]any{"build_seconds": build}
	if timeExact {
		out["exact"] = timed(qids, func(qid string) { x.Search(qvecs[qid], k) })
	}
	var graph []efTimes
	for _, ef := range gr.efs {
		k := dense.GraphDepth(ef)
		found := k
		if gr.study {
			found = ef // every document of layer 0's search, ef >= k on the grid
		}
		hits := make(map[string][]eval.Hit, len(qids))
		t := timed(qids, func(qid string) { hits[qid] = x.SearchGraph(g, qvecs[qid], found, ef, qid) })
		if gr.study {
			for qid, h := range hits {
				hits[qid] = cut(h, k)
			}
		}
		graph = append(graph, efTimes{ef, k, t})
		if err := writeRun(fmt.Sprintf("%s-ef%d.run", prefix, ef), qids, hits, fmt.Sprintf("%s-hnsw-ef%d", tag, ef)); err != nil {
			return nil, err
		}
	}
	out["hnsw"] = graph
	return out, nil
}

// cut returns the top k of hits as trec_eval ranks them in the run file:
// score as eval.WriteRun prints it, descending, ties by document ID
// descending (docs/PREREGISTRATION-AMENDMENTS.md, 15). Of documents tied at
// the cut that the search found, the run then keeps those that sort first
// in the exact run, not those of the lower node.
func cut(hits []eval.Hit, k int) []eval.Hit {
	r := make([]eval.Hit, len(hits))
	for i, h := range hits {
		r[i] = eval.Hit{DocID: h.DocID, Score: eval.Printed(h.Score)}
	}
	r = eval.Ranked(r)
	return r[:min(k, len(r))]
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

// readQueryVectors reads the query vectors in dir, the layout of
// encode/encode.py, and returns those of qids, which must all be there and
// of dimension dim.
func readQueryVectors(dir string, qids []string, dim int) (map[string][]float32, error) {
	ids, vecs, d, err := dense.ReadFaissFlat(dir)
	if err != nil {
		return nil, err
	}
	if d != dim {
		return nil, fmt.Errorf("%s: query vectors of dimension %d, documents of %d", dir, d, dim)
	}
	all := make(map[string][]float32, len(ids))
	for i, id := range ids {
		all[id] = vecs[i*d : (i+1)*d]
	}
	out := make(map[string][]float32, len(qids))
	for _, qid := range qids {
		v, ok := all[qid]
		if !ok {
			return nil, fmt.Errorf("%s: no vector for test query %s", dir, qid)
		}
		out[qid] = v
	}
	return out, nil
}

// encodeQueries encodes the test queries qids with BGE-base-en-v1.5 on
// ONNX Runtime, the instruction before each (D009).
func encodeQueries(data string, qids []string, model, ortLib string, dim int) (map[string][]float32, error) {
	if dim != dense.Dim {
		return nil, fmt.Errorf("document vectors of dimension %d, BGE-base-en-v1.5's are %d", dim, dense.Dim)
	}
	all, err := beir.Queries(data)
	if err != nil {
		return nil, err
	}
	texts := make([]string, len(qids))
	for i, qid := range qids {
		text, ok := all[qid]
		if !ok {
			return nil, fmt.Errorf("query %s is in the qrels but not in queries.jsonl", qid)
		}
		texts[i] = dense.QueryPrefix + text
	}
	if err := dense.Init(ortLib); err != nil {
		return nil, err
	}
	e, err := dense.NewEncoder(filepath.Join(model, "bge-base-en-v1.5-optimized.onnx"), filepath.Join(model, "tokenizer.json"), 512)
	if err != nil {
		return nil, err
	}
	defer e.Close() //nolint:errcheck // the vectors are copied out by then
	vecs, err := e.EncodeAll(texts, runtime.GOMAXPROCS(0), nil)
	if err != nil {
		return nil, err
	}
	out := make(map[string][]float32, len(qids))
	for i, qid := range qids {
		out[qid] = vecs[i*dim : (i+1)*dim]
	}
	return out, nil
}
