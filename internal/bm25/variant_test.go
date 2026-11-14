package bm25

import (
	"math"
	"strings"
	"testing"

	index "github.com/blevesearch/bleve_index_api"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
)

// docs makes documents a, b, c, ... with the given texts.
func docs(texts ...string) []beir.Doc {
	ds := make([]beir.Doc, len(texts))
	for i, text := range texts {
		ds[i] = beir.Doc{ID: string(rune('a' + i)), Text: text}
	}
	return ds
}

func zebras(n int) string { return strings.Repeat(" zebra", n) }

// scoresOf returns the unrounded score of every document that matches
// query under v, by document ID.
func scoresOf(t *testing.T, x *Index, v Variant, query string) map[string]float32 {
	t.Helper()
	s, err := x.scoringFor(v)
	if err != nil {
		t.Fatal(err)
	}
	byID, err := x.scores(query, s)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]float32{}
	for id, score := range byID {
		ext, err := x.r.ExternalID(index.IndexInternalID(id))
		if err != nil {
			t.Fatal(err)
		}
		out[ext] = score
	}
	return out
}

func TestEachSwitchChangesTheOrder(t *testing.T) {
	for _, c := range []struct {
		name          string
		v             Variant
		docs          []beir.Doc
		query         string
		lucene, bleve string // document IDs in rank order
	}{
		// a holds cat 4 times in 20 tokens, b once in 1: a frequency of 4
		// puts a first, its square root 2 does not.
		{"sqrt-tf", Variant{SqrtTF: true}, docs("cat cat cat cat"+zebras(16), "cat"), "cat", "ab", "ba"},
		// Two terms in two documents make Bleve's avgdl 1, Lucene's 10.5:
		// a's 20 tokens weigh 20 times the average, not twice.
		{"avgdl", Variant{Avgdl: true}, docs("cat cat cat cat"+zebras(16), "cat"), "cat", "ab", "ba"},
		// cat is in 2 of 6 documents, zebra in 1: squared, zebra's idf
		// outweighs a's four cats.
		{"idf-twice", Variant{IDFTwice: true}, docs("cat cat cat cat", "zebra dog dog dog", "cat bird", "bird", "bird", "bird"), "cat zebra", "abc", "bac"},
		// a matches one of the two query tokens, so the factor halves its
		// score; b matches both.
		{"coord", Variant{Coord: true}, docs("cat cat cat", "cat dog"+zebras(30), "dog"), "cat dog", "abc", "bac"},
		// Four documents without a token: N = 7, not 3, brings dog's idf
		// (two documents) closer to cat's (one).
		{"all-docs", Variant{AllDocs: true}, docs("cat zebra zebra", "dog dog dog", "dog", "", "", "", ""), "cat dog", "abc", "bac"},
		// a has 41 tokens, b 40. Lucene's norm byte stores both as 40, and
		// the tie goes to a by document ID.
		{"exact-length", Variant{Length: ExactLength}, docs("cat"+zebras(40), "cat"+zebras(39)), "cat", "ab", "ba"},
	} {
		t.Run(c.name, func(t *testing.T) {
			x := buildDocs(t, c.docs)
			for _, w := range []struct {
				v    Variant
				want string
			}{{Variant{}, c.lucene}, {c.v, c.bleve}} {
				search, err := x.SearchVariant(w.v)
				if err != nil {
					t.Fatal(err)
				}
				hits, err := search(c.query, 10)
				if err != nil {
					t.Fatal(err)
				}
				var got strings.Builder
				for _, h := range hits {
					got.WriteString(h.DocID)
				}
				if got.String() != w.want {
					t.Errorf("%+v ranks %s, want %s: %v", w.v, got.String(), w.want, hits)
				}
			}
		})
	}
}

func TestBleveStats(t *testing.T) {
	// cat, dog, fish, bird and zebra over six documents, d among them
	// although it has no token: avgdl = ceil(5 / 6).
	st, err := build(t).BleveStats()
	if err != nil {
		t.Fatal(err)
	}
	if want := (BleveStats{Docs: 6, Terms: 5}); st != want || st.Avgdl() != 1 {
		t.Fatalf("Bleve's statistics = %+v with avgdl %v, want %+v with avgdl 1", st, st.Avgdl(), want)
	}
}

func TestIDFTwiceAndCoordCountTokens(t *testing.T) {
	// Bleve's match query makes one clause with boost 1 per query token: cat
	// twice, and unicorn, which no document holds. The port makes one
	// clause per distinct term, boosted by its count. Both must score alike.
	tokens := []string{"cat", "cat", "dog", "unicorn"}
	df := map[string]float64{"cat": 4, "dog": 2}
	tf := map[string]map[string]float64{"a": {"cat": 2, "dog": 1}, "b": {"cat": 1}, "c": {"dog": 1}, "e": {"cat": 1}, "f": {"cat": 1}}
	length := map[string]float64{"a": 3, "b": 2, "c": 2, "e": 2, "f": 50} // f's 51 tokens are stored as 50
	idf := func(t string) float64 { return math.Log(1 + (5-df[t]+0.5)/(df[t]+0.5)) }
	var sumSq float64
	for _, tok := range tokens {
		sumSq += idf(tok) * idf(tok)
	}
	queryNorm := 1 / math.Sqrt(sumSq)
	x := build(t)
	twice := scoresOf(t, x, Variant{IDFTwice: true}, strings.Join(tokens, " "))
	coord := scoresOf(t, x, Variant{Coord: true}, strings.Join(tokens, " "))
	if len(twice) != len(tf) || len(coord) != len(tf) {
		t.Fatalf("%d and %d documents scored, want %d", len(twice), len(coord), len(tf))
	}
	for doc, freqs := range tf {
		var sum, wantTwice float64
		matched := 0
		for _, tok := range tokens {
			f := freqs[tok]
			if f == 0 {
				continue
			}
			s := idf(tok) * f / (f + 0.9*(0.6+0.4*length[doc]/12))
			sum += s
			wantTwice += s * idf(tok) * queryNorm
			matched++
		}
		wantCoord := sum * float64(matched) / float64(len(tokens))
		for _, c := range []struct {
			name      string
			got, want float64
		}{{"idf-twice", float64(twice[doc]), wantTwice}, {"coord", float64(coord[doc]), wantCoord}} {
			if math.Abs(c.got-c.want) > 1e-6*c.want {
				t.Errorf("%s scores %s %v, want %v", c.name, doc, c.got, c.want)
			}
		}
	}
}

func TestK1ScalesEveryScore(t *testing.T) {
	x := build(t)
	lucene := scoresOf(t, x, Variant{}, "cat dog fish")
	scaled := scoresOf(t, x, Variant{K1: true}, "cat dog fish")
	if len(scaled) != len(lucene) {
		t.Fatalf("%d documents scored, want %d", len(scaled), len(lucene))
	}
	for doc, s := range lucene {
		if want := 0.9 * float64(s); math.Abs(float64(scaled[doc])-want) > 1e-6*want {
			t.Errorf("k1 scores %s %v, want 0.9 × %v", doc, scaled[doc], s)
		}
	}
}

func TestBleveLength(t *testing.T) {
	// a has 6 tokens. Bleve's posting returns the norm n = float32(1/sqrt(6)),
	// and 1/(n·n) is not 6 in float32.
	n := float64(float32(1 / math.Sqrt(6)))
	l := float32(1 / (n * n))
	if l == 6 {
		t.Fatalf("1/(n·n) = %v, want it to differ from 6", l)
	}
	x := buildDocs(t, docs("cat"+zebras(5), "cat"))
	exact := scoresOf(t, x, Variant{Length: ExactLength}, "cat")
	bleve := scoresOf(t, x, Variant{Length: BleveLength}, "cat")
	if want := termScore(idf(2, 2), 1, normInverse(l, x.lucene.avgdl)); bleve["a"] != want || exact["a"] == want {
		t.Errorf("a scores %v with Bleve's length and %v with the exact one, want %v and another", bleve["a"], exact["a"], want)
	}
	// b's length 1 is exact either way.
	if bleve["b"] != exact["b"] {
		t.Errorf("b scores %v with Bleve's length, want %v", bleve["b"], exact["b"])
	}
}

func TestParseVariant(t *testing.T) {
	v, err := ParseVariant("sqrt-tf,exact-length")
	if err != nil || v != (Variant{SqrtTF: true, Length: ExactLength}) {
		t.Fatalf("ParseVariant = %+v, %v", v, err)
	}
	if all, err := ParseVariant("all"); err != nil || all != (Variant{true, true, true, true, true, BleveLength, true}) {
		t.Fatalf("ParseVariant(all) = %+v, %v", all, err)
	}
	if _, err := ParseVariant("sqrt"); err == nil {
		t.Fatal("ParseVariant(sqrt) succeeded, want an error")
	}
}
