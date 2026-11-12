package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
)

// write stores rows of dimension 2 under ids in a faiss flat directory.
func write(t *testing.T, ids []string, rows ...[2]float32) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "v")
	var v []float32
	for _, r := range rows {
		v = append(v, r[0], r[1])
	}
	if err := dense.WriteFaissFlat(dir, ids, v, 2); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestSubsetOfTheReference(t *testing.T) {
	// The reference holds the first documents only, in another order.
	ref := write(t, []string{"b", "a"}, [2]float32{0, 1}, [2]float32{1, 0})
	vecs := write(t, []string{"a", "b", "c"}, [2]float32{1, 0}, [2]float32{0, 1}, [2]float32{1, 1})
	g, err := agree(ref, vecs, false)
	if err != nil {
		t.Fatal(err)
	}
	if g.Docs != 2 || g.Min != 1 || g.Below9999 != 0 {
		t.Fatalf("got %+v", g)
	}
	if _, err := agree(ref, vecs, true); err == nil || !strings.Contains(err.Error(), "3 documents against 2") {
		t.Fatalf("-all with an extra document: %v", err)
	}
}

func TestBelowThreshold(t *testing.T) {
	ref := write(t, []string{"a", "b"}, [2]float32{1, 0}, [2]float32{0, 1})
	vecs := write(t, []string{"a", "b"}, [2]float32{1, 0}, [2]float32{0.02, 1})
	g, err := agree(ref, vecs, true)
	if err != nil {
		t.Fatal(err)
	}
	if g.Below9999 != 1 || g.Min >= 0.9999 {
		t.Fatalf("got %+v", g)
	}
}

func TestRefused(t *testing.T) {
	ref := write(t, []string{"a", "b"}, [2]float32{1, 0}, [2]float32{0, 1})
	for name, vecs := range map[string]string{
		"missing":   write(t, []string{"a", "c"}, [2]float32{1, 0}, [2]float32{0, 1}),
		"duplicate": write(t, []string{"a", "a", "b"}, [2]float32{1, 0}, [2]float32{1, 0}, [2]float32{0, 1}),
	} {
		if _, err := agree(ref, vecs, false); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	three := filepath.Join(t.TempDir(), "v")
	if err := dense.WriteFaissFlat(three, []string{"a", "b"}, make([]float32, 6), 3); err != nil {
		t.Fatal(err)
	}
	if _, err := agree(ref, three, false); err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("dimension 3 against 2: %v", err)
	}
}
