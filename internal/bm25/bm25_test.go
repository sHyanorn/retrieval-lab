package bm25

import (
	"bytes"
	"maps"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
)

// The corpus: "a" holds cat twice (title and text), "b" and "e" hold the
// same tokens and tie, "d" holds only stop words, and "f" has 51 tokens,
// which Lucene stores as a length of 50.
var corpus = []beir.Doc{
	{ID: "a", Title: "Cat", Text: "cat dog"},
	{ID: "b", Text: "fish and the cat"},
	{ID: "c", Text: "dog bird"},
	{ID: "d", Text: "the a"},
	{ID: "e", Text: "cat fish"},
	{ID: "f", Text: "cat" + strings.Repeat(" zebra", 50)},
}

func build(t *testing.T) *Index {
	t.Helper()
	x, err := Build(filepath.Join(t.TempDir(), "idx"), func(fn func(beir.Doc) error) error {
		for _, d := range corpus {
			if err := fn(d); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { x.Close() }) //nolint:errcheck // test cleanup
	return x
}

func TestAnalyzerFollowsAnserini(t *testing.T) {
	x := build(t)
	// "The" and "be" are Lucene stop words; "would" is not (it is in Bleve's
	// longer English list). The possessive goes before stemming.
	got := x.terms("The Cat's ponies would be running")
	want := map[string]int{"cat": 1, "poni": 1, "would": 1, "run": 1}
	if !maps.Equal(got, want) {
		t.Fatalf("terms = %v, want %v", got, want)
	}
}

func TestStats(t *testing.T) {
	want := Stats{Docs: 6, NonEmpty: 5, Tokens: 3 + 2 + 2 + 0 + 2 + 51}
	if got := build(t).Stats(); got != want {
		t.Fatalf("stats = %+v, want %+v", got, want)
	}
}

func TestSearchScoresByHand(t *testing.T) {
	// Five documents have a token, 60 tokens in all: avgdl 12. cat is in a,
	// b, e and f: idf = log(1 + (5 − 4 + 0.5) / (4 + 0.5)) = log(4/3). The
	// query repeats cat, so its one clause has boost 2.
	idf := math.Log(4.0 / 3)
	score := func(tf, length float64) float64 {
		return 2 * idf * tf / (tf + 0.9*(0.6+0.4*length/12))
	}
	want := []struct {
		id    string
		score float64
	}{{"a", score(2, 3)}, {"b", score(1, 2)}, {"e", score(1, 2)}, {"f", score(1, 50)}}

	hits, err := build(t).Search("cat cat", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != len(want) {
		t.Fatalf("got %d hits, want %d: %v", len(hits), len(want), hits)
	}
	for i, w := range want {
		// Scores come rounded to four decimals.
		if hits[i].DocID != w.id || math.Abs(hits[i].Score-w.score) > 1e-4 {
			t.Fatalf("hit %d = %v, want %s with %.6f; all: %v", i, hits[i], w.id, w.score, hits)
		}
	}
	// b and e tie; e follows b by document ID and sits just below it.
	if hits[2].Score >= hits[1].Score {
		t.Fatalf("tied hits not separated: %v", hits)
	}
}

func TestSearchCutsTiesByDocID(t *testing.T) {
	hits, err := build(t).Search("fish", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].DocID != "b" {
		t.Fatalf("hits = %v, want b alone", hits)
	}
}

func TestWriteRunLeavesOutTheQueryDocument(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteRun(&buf, build(t).Search, map[string]string{"a": "cat"}, 2, "t"); err != nil {
		t.Fatal(err)
	}
	if lines := strings.Split(strings.TrimSpace(buf.String()), "\n"); len(lines) != 1 || !strings.HasPrefix(lines[0], "a Q0 b 1 ") {
		t.Fatalf("run = %q, want only document b at rank 1", buf.String())
	}
}
