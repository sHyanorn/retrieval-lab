package main

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func TestReadQueryVectors(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "queries")
	if err := dense.WriteFaissFlat(dir, []string{"q2", "q1", "q9"}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3); err != nil {
		t.Fatal(err)
	}
	got, err := readQueryVectors(dir, []string{"q1", "q2"}, 3)
	if err != nil || len(got) != 2 || !slices.Equal(got["q1"], []float32{4, 5, 6}) || !slices.Equal(got["q2"], []float32{1, 2, 3}) {
		t.Fatalf("readQueryVectors = %v, %v", got, err)
	}
	if _, err := readQueryVectors(dir, []string{"q1", "q3"}, 3); err == nil {
		t.Error("a test query without a vector was accepted")
	}
	if _, err := readQueryVectors(dir, []string{"q1"}, 4); err == nil {
		t.Error("query vectors of another dimension than the documents' were accepted")
	}
}

// TestStudyRunCutsTiesAsTrecEval searches 40 documents on the unit circle,
// at decreasing cosine to the query, on a graph that finds them all at ef
// 40. The 10th and 11th are the same vector: node 2, a, and node 39, z. The
// graph keeps the lower node, a; trec_eval ranks z 10th in the exact run.
func TestStudyRunCutsTiesAsTrecEval(t *testing.T) {
	const n, ef = 40, 40
	x := &dense.Index{IDs: make([]string, n), Vecs: make([]float32, 2*n)}
	rank := 0
	for i := range n {
		id, r := fmt.Sprintf("d%02d", i), 9
		switch i {
		case 2:
			id = "a"
		case n - 1:
			id = "z"
		default:
			if rank == 9 {
				rank++
			}
			r = rank
			rank++
		}
		theta := 0.03 * float64(r)
		x.IDs[i] = id
		x.Vecs[2*i], x.Vecs[2*i+1] = float32(math.Cos(theta)), float32(math.Sin(theta))
	}
	q := map[string][]float32{"q": {1, 0}}
	qrels := eval.Qrels{"q": {"d00": 1, "z": 1}}

	var b bytes.Buffer
	search := func(qid string, k int) ([]eval.Hit, error) { return x.Search(q[qid], k), nil }
	if err := eval.RunQueries(&b, search, map[string]string{"q": "q"}, 1000, "t"); err != nil {
		t.Fatal(err)
	}
	exact, err := eval.ReadRun(&b)
	if err != nil {
		t.Fatal(err)
	}
	want := eval.PerQuery(qrels, exact, exact)[0]

	measure := func(study bool) eval.Query {
		prefix := filepath.Join(t.TempDir(), "hnsw")
		if _, err := approximate(x, q, 1000, prefix, "t", graph{seed: 0, m: 16, efc: 200, efs: []int{ef}, study: study}, false); err != nil {
			t.Fatal(err)
		}
		f, err := os.Open(fmt.Sprintf("%s-ef%d.run", prefix, ef))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close() //nolint:errcheck // read only
		run, err := eval.ReadRun(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(run["q"]) != dense.GraphDepth(ef) {
			t.Errorf("run of %d hits, want %d", len(run["q"]), dense.GraphDepth(ef))
		}
		return eval.PerQuery(qrels, run, exact)[0]
	}
	if got := measure(true); got.NDCG10 != want.NDCG10 || got.Found10 != 10 || got.Of10 != 10 {
		t.Errorf("study run: nDCG@10 %v (exact %v), neighbour recall %d of %d, want no loss", got.NDCG10, want.NDCG10, got.Found10, got.Of10)
	}
	// M3's run keeps the graph's choice, a, and loses z's gain.
	if got := measure(false); got.NDCG10 == want.NDCG10 {
		t.Error("the run cut by the graph's order of nodes passed")
	}
}

func TestCutRanksPrintedScores(t *testing.T) {
	// a and z differ in the seventh decimal and print alike: trec_eval reads
	// a tie and ranks z first.
	hits := []eval.Hit{{DocID: "b", Score: 0.9}, {DocID: "a", Score: 0.5000004}, {DocID: "z", Score: 0.5000001}, {DocID: "c", Score: 0.1}}
	if eval.Ranked(hits)[1].DocID != "a" {
		t.Fatal("a is not above z in the raw scores")
	}
	got := cut(hits, 2)
	if len(got) != 2 || got[0].DocID != "b" || got[1].DocID != "z" {
		t.Errorf("cut = %v, want b then z", got)
	}
}
