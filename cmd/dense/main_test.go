package main

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
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
