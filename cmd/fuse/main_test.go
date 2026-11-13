package main

import (
	"os"
	"path/filepath"
	"testing"
)

// The study's two dense runs of the same documents, an exact run with ties
// by document ID ascending and an HNSW run as trec_eval ranks it, fuse to
// the same files with -dense-rank trec; by their lines, to different RRF
// files and the same Normalize file, which reads scores, not ranks.
func TestDenseRankTrec(t *testing.T) {
	dir := t.TempDir()
	write := func(name, text string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	bm25 := write("bm25.run", "q Q0 c 1 3.000000 bm25\n")
	exact := write("exact.run", "q Q0 a 1 1.000000 t\nq Q0 z 2 1.000000 t\nq Q0 c 3 0.500000 t\n")
	hnsw := write("hnsw.run", "q Q0 z 1 1.000000 t\nq Q0 a 2 1.000000 t\nq Q0 c 3 0.500000 t\n")
	fused := func(dense, rank string) (rrf, norm string) {
		r, n := filepath.Join(dir, "rrf.run"), filepath.Join(dir, "norm.run")
		if err := run(bm25, dense, r, n, []int{1000, 100}, rank); err != nil {
			t.Fatal(err)
		}
		a, err := os.ReadFile(r)
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(n)
		if err != nil {
			t.Fatal(err)
		}
		return string(a), string(b)
	}
	er, en := fused(exact, "trec")
	ar, an := fused(hnsw, "trec")
	if er != ar || en != an {
		t.Errorf("-dense-rank trec: the exact run fuses to\n%s%s, the HNSW run to\n%s%s", er, en, ar, an)
	}
	er, en = fused(exact, "line")
	ar, an = fused(hnsw, "line")
	if er == ar {
		t.Errorf("-dense-rank line: both runs fuse by RRF to\n%s", er)
	}
	if en != an {
		t.Errorf("-dense-rank line: Normalize gives\n%s and\n%s", en, an)
	}
	if err := run(bm25, exact, filepath.Join(dir, "r"), filepath.Join(dir, "n"), []int{1000, 100}, "rank"); err == nil {
		t.Error("an unknown -dense-rank was accepted")
	}
}
