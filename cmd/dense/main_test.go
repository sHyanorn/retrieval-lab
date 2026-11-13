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
	"github.com/sHyanorn/retrieval-lab/internal/fusion"
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

// TestWorkersWriteTheSameRun writes the exact run of a small dataset with
// one worker and with four, and wants the same file. Some documents share a
// vector, so that ties are ordered by ID, and some queries are documents, so
// that the self-hit is removed.
func TestWorkersWriteTheSameRun(t *testing.T) {
	const n, nq, dim = 300, 40, 4
	dir := t.TempDir()
	data := filepath.Join(dir, "data")
	if err := os.MkdirAll(filepath.Join(data, "qrels"), 0o755); err != nil {
		t.Fatal(err)
	}
	ids, vecs := make([]string, n), make([]float32, n*dim)
	for i := range n {
		ids[i] = fmt.Sprintf("d%03d", i)
		for j := range dim {
			vecs[i*dim+j] = float32(math.Sin(float64((i%97 + 1) * (j + 3)))) // i and i+97 share a vector
		}
	}
	qids, qvecs := make([]string, nq), make([]float32, nq*dim)
	qrels := "query-id\tcorpus-id\tscore\n"
	for i := range nq {
		qids[i] = fmt.Sprintf("q%02d", i)
		if i%5 == 0 {
			qids[i] = ids[i*7]
		}
		copy(qvecs[i*dim:], vecs[(i*7)*dim:(i*7+1)*dim])
		qrels += fmt.Sprintf("%s\t%s\t1\n", qids[i], ids[(i*11)%n])
	}
	if err := os.WriteFile(filepath.Join(data, "qrels", "test.tsv"), []byte(qrels), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := dense.WriteFaissFlat(filepath.Join(dir, "docs"), ids, vecs, dim); err != nil {
		t.Fatal(err)
	}
	if err := dense.WriteFaissFlat(filepath.Join(dir, "queries"), qids, qvecs, dim); err != nil {
		t.Fatal(err)
	}
	write := func(workers int) []byte {
		out := filepath.Join(dir, fmt.Sprintf("w%d.run", workers))
		if err := run(data, filepath.Join(dir, "docs"), filepath.Join(dir, "queries"), "", "", out, "t", 100, "", graph{workers: workers}); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	one, four := write(1), write(4)
	if len(one) == 0 || !bytes.Equal(one, four) {
		t.Fatalf("runs differ: %d bytes with one worker, %d with four", len(one), len(four))
	}
	if bytes.Contains(one, []byte("d000 Q0 d000")) || !bytes.Contains(one, []byte("\nd035 Q0 d132 1 ")) {
		t.Fatalf("self-hit kept, or tie not first by ID:\n%.400s", one)
	}
}

// An HNSW run that holds every document of the exact run, or of its top at
// the dense run's depth, must fuse to the same run (PREREGISTRATION.md,
// Section 6). It does once fusion ranks each dense run as trec_eval does;
// by their lines the two differ, since the exact run lists ties by
// document ID ascending and the study's HNSW runs by ID descending.
func TestFusedRunsOfTheSameDocumentsAgree(t *testing.T) {
	file := func(h []eval.Hit) eval.Run {
		var b bytes.Buffer
		if err := eval.WriteRun(&b, "q", h, "t"); err != nil {
			t.Fatal(err)
		}
		r, err := eval.ReadRun(&b)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	bm25 := eval.Run{"q": {{DocID: "c", Score: 3}}}
	fuse := func(d eval.Run, depth int) []eval.Hit {
		return fusion.RRF([]eval.Run{bm25, d}, []int{1000, depth}, 60, 1000)["q"]
	}
	for _, c := range []struct {
		name  string
		ids   []string
		vecs  []float32
		depth int
	}{
		// a and z tie at 1, above c.
		{"tie above the depth", []string{"a", "z", "c"}, []float32{1, 0, 1, 0, 0, 1}, 100},
		// x and y tie at 0.8 across a depth of 2: the exact run's first two
		// lines hold x, trec_eval's top two y.
		{"tie across the depth", []string{"b", "x", "y"}, []float32{1, 0, 0.6, 0.8, 0.6, 0.8}, 2},
	} {
		x := &dense.Index{IDs: c.ids, Vecs: c.vecs}
		hits := x.Search([]float32{1, 0}, 1000)
		exact := file(hits)                // as cmd/dense writes the exact run
		approx := file(cut(hits, c.depth)) // as cmd/dense -grid study writes a run that found them all
		if e, a := fuse(fusion.TrecOrder(exact), c.depth), fuse(fusion.TrecOrder(approx), c.depth); !slices.Equal(e, a) {
			t.Errorf("%s: fused %v with the exact run, %v with an HNSW run of the same documents", c.name, e, a)
		}
		if e, a := fuse(exact, c.depth), fuse(approx, c.depth); slices.Equal(e, a) {
			t.Errorf("%s: by their lines the two runs fuse alike, %v", c.name, e)
		}
	}
}
