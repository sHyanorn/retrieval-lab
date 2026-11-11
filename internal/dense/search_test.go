package dense

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// unit returns a vector of Dim values with the given leading components.
func unit(c ...float32) []float32 {
	v := make([]float32, Dim)
	copy(v, c)
	return v
}

func TestSearchScoresAsLuceneDotProduct(t *testing.T) {
	x := &Index{IDs: []string{"c", "b", "a", "d"}}
	for _, v := range [][]float32{unit(0.6, 0.8), unit(1), unit(1), unit(-1)} {
		x.Vecs = append(x.Vecs, v...)
	}
	got := x.Search(unit(1), 3)
	// a and b tie at (1 + 1) / 2 = 1 and come in ID order; c scores
	// (1 + 0.6) / 2; d, at (1 − 1) / 2 = 0, is cut by k.
	want := []eval.Hit{{DocID: "a", Score: 1}, {DocID: "b", Score: 1}, {DocID: "c", Score: float64(float32(1.6) / 2)}}
	if !slices.Equal(got, want) {
		t.Fatalf("Search = %v, want %v", got, want)
	}
}

func TestFaissFlatRoundTrip(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "v")
	ids, vecs := []string{"d1", "d2"}, append(unit(1, 2), unit(3)...)
	if err := WriteFaissFlat(dir, ids, vecs, Dim); err != nil {
		t.Fatal(err)
	}
	gotIDs, gotVecs, dim, err := ReadFaissFlat(dir)
	if err != nil || dim != Dim || !slices.Equal(gotIDs, ids) || !slices.Equal(gotVecs, vecs) {
		t.Fatalf("read back %v, %d values, dim %d, %v", gotIDs, len(gotVecs), dim, err)
	}
}

func TestCompareAlignsByIDAndCountsLowCosines(t *testing.T) {
	// b holds the same documents in another order; d2's vectors are 60°
	// apart (cosine 0.5), d1's are parallel and d3's equal.
	a := append(append(unit(1), unit(2)...), unit(2, 2)...)
	b := append(append(unit(1, 1.7320508), unit(3, 3)...), unit(5)...)
	g, err := Compare([]string{"d1", "d2", "d3"}, a, []string{"d2", "d3", "d1"}, b, Dim)
	if err != nil {
		t.Fatal(err)
	}
	if g.Docs != 3 || math.Abs(g.Min-0.5) > 1e-6 || g.P50 < 0.9999999 || g.Below9999 != 1 || g.Below99999 != 1 {
		t.Fatalf("Compare = %+v", g)
	}
	if _, err := Compare([]string{"d1"}, unit(1), []string{"x"}, unit(1), Dim); err == nil {
		t.Fatal("want an error for different documents")
	}
}

func TestDocTextDropsFormatCharacters(t *testing.T) {
	// Hugging Face's BertNormalizer drops every character of Unicode
	// category C but tab, newline and carriage return; go-huggingface drops
	// only Cc. A soft hyphen, a zero-width space and a left-to-right mark
	// (all Cf) must go; the newline stays.
	d := beir.Doc{Title: "infor\u00admation", Text: "zero\u200bwidth\u200e\nline"}
	if got, want := DocText(d), "information zerowidth\nline"; got != want {
		t.Fatalf("DocText = %q, want %q", got, want)
	}
}
