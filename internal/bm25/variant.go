package bm25

import (
	"fmt"
	"math"
	"strings"

	index "github.com/blevesearch/bleve_index_api"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// Variant switches on, alone or together, the ways in which Bleve's BM25
// scores differently from Lucene's BM25Similarity. The zero value is the
// Lucene port. References are to Bleve v2.6.3.
type Variant struct {
	SqrtTF   bool   // tf is the square root of the frequency (search/scorer/scorer_term.go:204-209)
	Avgdl    bool   // avgdl is Bleve's: BleveStats.Avgdl
	IDFTwice bool   // each clause is multiplied again by idf · queryNorm (scorer_term.go:121,223-224; search/searcher/search_disjunction_slice.go:104-116)
	Coord    bool   // a document's score is multiplied by its matched / all query tokens (search/scorer/scorer_disjunction.go:65-66)
	K1       bool   // each clause is multiplied by k1, as Bleve's numerator is (scorer_term.go:148-149)
	Length   Length // the document length
	AllDocs  bool   // idf's document count is every document, Bleve's DocCount (search/searcher/search_term.go:96,157)
}

// Length is the document length a score uses.
type Length int

const (
	// LuceneLength is the length through Lucene's one-byte norm.
	LuceneLength Length = iota
	// ExactLength is the number of tokens.
	ExactLength
	// BleveLength is 1/(norm·norm), the length Bleve's scorer computes from a
	// posting (scorer_term.go:146). The norm is 1/sqrt(length) rounded to
	// float32 (zapx v17 posting.go:894-896), so it is not a whole number.
	BleveLength
)

// variants are the names ParseVariant reads; "all" is every switch, with
// Bleve's length.
var variants = map[string]func(*Variant){
	"sqrt-tf":      func(v *Variant) { v.SqrtTF = true },
	"avgdl":        func(v *Variant) { v.Avgdl = true },
	"idf-twice":    func(v *Variant) { v.IDFTwice = true },
	"coord":        func(v *Variant) { v.Coord = true },
	"k1":           func(v *Variant) { v.K1 = true },
	"exact-length": func(v *Variant) { v.Length = ExactLength },
	"bleve-length": func(v *Variant) { v.Length = BleveLength },
	"all-docs":     func(v *Variant) { v.AllDocs = true },
	"all":          func(v *Variant) { *v = Variant{true, true, true, true, true, BleveLength, true} },
}

// ParseVariant reads a comma-separated list of the names in variants.
func ParseVariant(names string) (Variant, error) {
	var v Variant
	for n := range strings.SplitSeq(names, ",") {
		set, ok := variants[n]
		if !ok {
			return Variant{}, fmt.Errorf("unknown variant %q", n)
		}
		set(&v)
	}
	return v, nil
}

// BleveStats are the statistics Bleve's BM25 reads from an index
// (search/searcher/search_term.go:94-117).
type BleveStats struct {
	Docs  uint64 `json:"docs"`  // every document, with a token or not
	Terms int    `json:"terms"` // terms in the field's dictionary, summed over segments
}

// Avgdl is Bleve's average field length: Terms / Docs, rounded up.
func (s BleveStats) Avgdl() float64 { return math.Ceil(float64(s.Terms) / float64(s.Docs)) }

// BleveStats returns the statistics Bleve's BM25 reads from this index.
func (x *Index) BleveStats() (BleveStats, error) {
	docs, err := x.r.DocCount()
	if err != nil {
		return BleveStats{}, err
	}
	r, ok := x.r.(index.BM25Reader)
	if !ok {
		return BleveStats{}, fmt.Errorf("index reader is a %T, want an index.BM25Reader", x.r)
	}
	terms, err := r.FieldCardinality(field)
	return BleveStats{Docs: docs, Terms: terms}, err
}

// SearchVariant returns Search with the switches of v.
func (x *Index) SearchVariant(v Variant) (func(query string, k int) ([]eval.Hit, error), error) {
	s, err := x.scoringFor(v)
	if err != nil {
		return nil, err
	}
	return func(query string, k int) ([]eval.Hit, error) { return x.search(query, k, s) }, nil
}

func (x *Index) scoringFor(v Variant) (*scoring, error) {
	docCount, avgdl := x.lucene.docCount, x.lucene.avgdl
	if v.AllDocs || v.Avgdl {
		st, err := x.BleveStats()
		if err != nil {
			return nil, err
		}
		if v.AllDocs {
			docCount = int64(st.Docs)
		}
		if v.Avgdl {
			avgdl = float32(st.Avgdl())
		}
	}
	return newScoring(v, docCount, avgdl), nil
}

// scoring is a Variant with the statistics it scores with.
type scoring struct {
	Variant
	docCount int64 // idf's document count
	avgdl    float32
	norms    [256]float32 // normCache(avgdl)
}

func newScoring(v Variant, docCount int64, avgdl float32) *scoring {
	return &scoring{Variant: v, docCount: docCount, avgdl: avgdl, norms: normCache(avgdl)}
}

// freq is the term frequency a score uses.
func (s *scoring) freq(f uint64) float32 {
	if s.SqrtTF {
		return float32(math.Sqrt(float64(f)))
	}
	return float32(f)
}

// normInverse is normInverse of the length a score uses, from a posting's
// norm. Bleve stores the exact length; its posting returns 1/sqrt(length).
func (s *scoring) normInverse(norm float64) float32 {
	switch s.Length {
	case ExactLength:
		return normInverse(float32(math.Round(1/(norm*norm))), s.avgdl)
	case BleveLength:
		return normInverse(float32(1/(norm*norm)), s.avgdl)
	}
	return s.norms[intToByte4(int(math.Round(1/(norm*norm))))]
}
