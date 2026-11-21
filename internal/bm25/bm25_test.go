package bm25

import (
	"bytes"
	"cmp"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	index "github.com/blevesearch/bleve_index_api"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
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

func build(t *testing.T) *Index { t.Helper(); return buildFrom(t, corpus) }

func buildFrom(t *testing.T, docs []beir.Doc) *Index {
	t.Helper()
	x, err := Build(filepath.Join(t.TempDir(), "idx"), func(fn func(beir.Doc) error) error {
		for _, d := range docs {
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

func TestRunLeavesOutTheQueryDocument(t *testing.T) {
	var buf bytes.Buffer
	if err := eval.RunQueries(&buf, build(t).Search, map[string]string{"a": "cat"}, 2, "t"); err != nil {
		t.Fatal(err)
	}
	if lines := strings.Split(strings.TrimSpace(buf.String()), "\n"); len(lines) != 1 || !strings.HasPrefix(lines[0], "a Q0 b 1 ") {
		t.Fatalf("run = %q, want only document b at rank 1", buf.String())
	}
}

func TestRunAdjustsTiesBeforeLeavingOutTheQueryDocument(t *testing.T) {
	// b and e tie for "fish"; Pyserini adjusts the 1000 hits, b first, and
	// then drops the hit equal to the query ID, so e keeps its lowered score.
	x := build(t)
	hits, err := x.Search("fish", 2)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := eval.RunQueries(&buf, x.Search, map[string]string{"b": "fish"}, 2, "t"); err != nil {
		t.Fatal(err)
	}
	if want := fmt.Sprintf("b Q0 e 1 %f t\n", hits[1].Score); buf.String() != want || hits[1].Score >= hits[0].Score {
		t.Fatalf("run = %q, want %q below %v", buf.String(), want, hits[0].Score)
	}
}

// randomCorpus draws n documents from a vocabulary of four words and two
// stop words, a third of them copies of an earlier one, so that many
// documents tie; IDs are drawn apart from the order of indexing.
func randomCorpus(r *rand.Rand, n int) []beir.Doc {
	words := []string{"cat", "dog", "fish", "bird", "the", "and"}
	text := func(most int) string {
		w := make([]string, r.IntN(most+1))
		for i := range w {
			w[i] = words[r.IntN(len(words))]
		}
		return strings.Join(w, " ")
	}
	docs := make([]beir.Doc, n)
	for i, id := range r.Perm(n) {
		docs[i] = beir.Doc{ID: fmt.Sprintf("d%03d", id), Title: text(2), Text: text(8)}
		if i > 0 && r.IntN(3) == 0 {
			c := docs[r.IntN(i)]
			docs[i].Title, docs[i].Text = c.Title, c.Text
		}
	}
	return docs
}

// ranked is what Search must return: every document query matches, sorted
// by score and then ID, the first k, ties adjusted.
func ranked(x *Index, query string, k int) ([]eval.Hit, error) {
	acc := &accumulator{sum: make([]float64, x.stats.Docs), seen: make([]bool, x.stats.Docs)}
	counts := x.terms(query)
	for _, t := range slices.Sorted(maps.Keys(counts)) {
		if err := x.addTerm(acc, t, float32(counts[t])); err != nil {
			return nil, err
		}
	}
	hits := make([]eval.Hit, len(acc.touched))
	for i, d := range acc.touched {
		id, err := x.r.ExternalID(index.NewIndexInternalID(nil, d))
		if err != nil {
			return nil, err
		}
		hits[i] = eval.Hit{DocID: id, Score: float64(float32(acc.sum[d]))}
	}
	slices.SortFunc(hits, func(a, b eval.Hit) int {
		return cmp.Or(cmp.Compare(b.Score, a.Score), strings.Compare(a.DocID, b.DocID))
	})
	hits = hits[:min(k, len(hits))]
	scores := make([]float32, len(hits))
	for i, h := range hits {
		scores[i] = float32(h.Score)
	}
	adjustTies(scores)
	for i := range hits {
		hits[i].Score = float64(scores[i])
	}
	return hits, nil
}

// checkSearch compares search with ranked on random corpora, for every k
// from 1 to one past the corpus, queries following each other on one
// index; it returns the first difference.
func checkSearch(t *testing.T, search func(x *Index, q string, k int) ([]eval.Hit, error)) error {
	r := rand.New(rand.NewPCG(1, 2))
	for c := range 8 {
		x := buildFrom(t, randomCorpus(r, 10+5*c))
		for range 20 {
			q := randomCorpus(r, 1)[0].Text
			for k := 1; k <= int(x.stats.Docs)+1; k++ {
				got, err := search(x, q, k)
				if err != nil {
					return err
				}
				want, err := ranked(x, q, k)
				if err != nil {
					return err
				}
				if !slices.Equal(got, want) {
					return fmt.Errorf("corpus %d, query %q, k %d: got %v, want %v", c, q, k, got, want)
				}
			}
		}
	}
	return nil
}

func exhaustive(mut mutation) func(x *Index, q string, k int) ([]eval.Hit, error) {
	return func(x *Index, q string, k int) ([]eval.Hit, error) { return x.exhaustive(q, k, mut) }
}

func maxScore(mut mutation) func(x *Index, q string, k int) ([]eval.Hit, error) {
	return func(x *Index, q string, k int) ([]eval.Hit, error) { return x.maxScore(q, k, mut) }
}

func TestSearchEqualsFullRanking(t *testing.T) {
	for name, search := range map[string]func(x *Index, q string, k int) ([]eval.Hit, error){
		"exhaustive": exhaustive(mutation{}),
		"MaxScore":   maxScore(mutation{}),
	} {
		if err := checkSearch(t, search); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestBrokenSearchesAreCaught(t *testing.T) {
	for name, search := range map[string]func(x *Index, q string, k int) ([]eval.Hit, error){
		"ties at rank k dropped":           exhaustive(mutation{strictCut: true}),
		"cut from k − 1 scores":            exhaustive(mutation{shortHeap: true}),
		"ties by document number":          exhaustive(mutation{byDocNum: true}),
		"sums kept from the last query":    exhaustive(mutation{stale: true}),
		"MaxScore, bounds halved":          maxScore(mutation{lowBound: true}),
		"MaxScore, bound equal to the cut": maxScore(mutation{pruneTies: true}),
		"MaxScore, cut from k − 1 scores":  maxScore(mutation{shortHeap: true}),
	} {
		err := checkSearch(t, search)
		if err == nil {
			t.Errorf("%s: passed", name)
		}
		t.Logf("%s: %.200v", name, err)
	}
}
