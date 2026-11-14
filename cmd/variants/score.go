package main

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/bits"
	"slices"
	"strings"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/analysis"
	index "github.com/blevesearch/bleve_index_api"

	"github.com/sHyanorn/retrieval-lab/internal/bm25"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// This file is a copy of the scoring of internal/bm25 (lucene.go, and
// Search with its term loop in bm25.go) with Bleve's differences as
// switches. It is kept outside step 2's code, which stays as it was at
// bd072df (docs/PREREGISTRATION-AMENDMENTS.md, amendment 26). With no
// switch on it must rank and score as internal/bm25 does: make variants
// checks its run files byte for byte against results/bm25-flat.sha256.

// The names internal/bm25 gives the indexed field, its analyzer and the
// statistics it stores in the index.
const (
	field    = "contents"
	analyzer = "lucene_english"
	statsKey = "bm25_stats"
)

// k1 and b are Anserini's BM25 defaults, which Pyserini's --bm25 keeps.
const k1, b float32 = 0.9, 0.4

// Variant switches on, alone or together, the ways in which Bleve's BM25
// scores differently from Lucene's BM25Similarity. The zero value is the
// Lucene port. References are to Bleve v2.6.3.
type Variant struct {
	SqrtTF   bool   // tf is the square root of the frequency (search/scorer/scorer_term.go:204-209)
	Avgdl    bool   // avgdl is Bleve's: bleveStats.avgdl
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
	// float32 (zapx v17.2.5 posting.go:894-896), so it is not a whole number.
	BleveLength
)

// variants are the names parseVariant reads; "all" is every switch, with
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

// parseVariant reads a comma-separated list of the names in variants.
func parseVariant(names string) (Variant, error) {
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

// collection is an index made by internal/bm25.Build, open for scoring.
type collection struct {
	idx    bleve.Index
	r      index.IndexReader
	an     analysis.Analyzer
	stats  bm25.Stats
	lucene *scoring
}

// open opens an index made by internal/bm25.Build.
func open(path string) (*collection, error) {
	idx, err := bleve.Open(path)
	if err != nil {
		return nil, err
	}
	x := &collection{idx: idx, an: idx.Mapping().AnalyzerNamed(analyzer)}
	err = func() error {
		buf, err := idx.GetInternal([]byte(statsKey))
		if err != nil {
			return err
		}
		if err := json.Unmarshal(buf, &x.stats); err != nil {
			return fmt.Errorf("index statistics: %w", err)
		}
		if x.stats.NonEmpty == 0 {
			return fmt.Errorf("index statistics: no document has a token")
		}
		adv, err := idx.Advanced()
		if err != nil {
			return err
		}
		x.r, err = adv.Reader()
		return err
	}()
	if err != nil {
		idx.Close() //nolint:errcheck // the open error is the one to report
		return nil, err
	}
	// BM25Similarity.avgFieldLength.
	avgdl := float32(float64(x.stats.Tokens) / float64(x.stats.NonEmpty))
	x.lucene = newScoring(Variant{}, x.stats.NonEmpty, avgdl)
	return x, nil
}

func (x *collection) close() error {
	err := x.r.Close()
	return cmp.Or(x.idx.Close(), err)
}

// bleveStats are the statistics Bleve's BM25 reads from an index
// (search/searcher/search_term.go:94-117).
type bleveStats struct {
	Docs  uint64 `json:"docs"`  // every document, with a token or not
	Terms int    `json:"terms"` // terms in the field's dictionary, summed over segments
}

// avgdl is Bleve's average field length: Terms / Docs, rounded up.
func (s bleveStats) avgdl() float64 { return math.Ceil(float64(s.Terms) / float64(s.Docs)) }

func (x *collection) bleveStats() (bleveStats, error) {
	docs, err := x.r.DocCount()
	if err != nil {
		return bleveStats{}, err
	}
	r, ok := x.r.(index.BM25Reader)
	if !ok {
		return bleveStats{}, fmt.Errorf("index reader is a %T, want an index.BM25Reader", x.r)
	}
	terms, err := r.FieldCardinality(field)
	return bleveStats{Docs: docs, Terms: terms}, err
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

func (x *collection) scoringFor(v Variant) (*scoring, error) {
	docCount, avgdl := x.lucene.docCount, x.lucene.avgdl
	if v.AllDocs || v.Avgdl {
		st, err := x.bleveStats()
		if err != nil {
			return nil, err
		}
		if v.AllDocs {
			docCount = int64(st.Docs)
		}
		if v.Avgdl {
			avgdl = float32(st.avgdl())
		}
	}
	return newScoring(v, docCount, avgdl), nil
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

// terms analyzes a query as Anserini's BagOfWordsQueryGenerator does: one
// clause per distinct term, boosted by the number of times it occurs.
func (x *collection) terms(query string) map[string]int {
	counts := map[string]int{}
	for _, t := range x.an.Analyze([]byte(query)) {
		counts[string(t.Term)]++
	}
	return counts
}

// search returns the top k documents for query as internal/bm25's Search
// ranks and scores them, with the switches of s: ordered by score, then
// document ID ascending; scores then passed through adjustTies.
func (x *collection) search(query string, k int, s *scoring) ([]eval.Hit, error) {
	byID, err := x.scores(query, s)
	if err != nil {
		return nil, err
	}
	type cand struct {
		id    string
		score float32
	}
	cs := make([]cand, 0, len(byID))
	for id, score := range byID {
		cs = append(cs, cand{id, score})
	}
	slices.SortFunc(cs, func(a, b cand) int { return cmp.Compare(b.score, a.score) })
	if len(cs) > k {
		n := k
		for n < len(cs) && cs[n].score == cs[k-1].score {
			n++
		}
		cs = cs[:n]
	}
	for i := range cs {
		ext, err := x.r.ExternalID(index.IndexInternalID(cs[i].id))
		if err != nil {
			return nil, err
		}
		cs[i].id = ext
	}
	slices.SortFunc(cs, func(a, b cand) int {
		return cmp.Or(cmp.Compare(b.score, a.score), strings.Compare(a.id, b.id))
	})
	cs = cs[:min(k, len(cs))]
	scores := make([]float32, len(cs))
	for i, c := range cs {
		scores[i] = c.score
	}
	adjustTies(scores)
	hits := make([]eval.Hit, len(cs))
	for i, c := range cs {
		hits[i] = eval.Hit{DocID: c.id, Score: float64(scores[i])}
	}
	return hits, nil
}

// clause is one distinct query term: its postings, the number of times it
// occurs in the query (the clause's boost) and its idf.
type clause struct {
	tfr    index.TermFieldReader
	tokens int
	idf    float32
}

// scores returns the score of every document that holds a query term, keyed
// by Bleve's internal document ID: Lucene's BM25 summed over the query's
// terms, with the switches of s.
func (x *collection) scores(query string, s *scoring) (map[string]float32, error) {
	counts := x.terms(query)
	clauses := make([]clause, 0, len(counts))
	defer func() {
		for _, c := range clauses {
			c.tfr.Close() //nolint:errcheck // read-only
		}
	}()
	// Bleve's match query has one clause per token, with boost 1, a term the
	// index does not hold included: queryNorm and the coordination factor
	// count tokens.
	tokens, sumSq := 0, 0.0
	for _, t := range slices.Sorted(maps.Keys(counts)) {
		tfr, err := x.r.TermFieldReader(context.Background(), []byte(t), field, true, true, false)
		if err != nil {
			return nil, err
		}
		c := clause{tfr, counts[t], idf(int64(tfr.Count()), s.docCount)}
		clauses = append(clauses, c)
		tokens += c.tokens
		sumSq += float64(c.tokens) * float64(c.idf) * float64(c.idf)
	}
	queryNorm := 1 / math.Sqrt(sumSq)
	type sum struct {
		score  float64
		tokens int
	}
	sums := map[string]sum{}
	for _, c := range clauses {
		if c.tfr.Count() == 0 {
			continue
		}
		weight := float32(float32(c.tokens) * c.idf)
		var d index.TermFieldDoc
		for {
			p, err := c.tfr.Next(&d)
			if err != nil {
				return nil, err
			}
			if p == nil {
				break
			}
			score := termScore(weight, s.freq(p.Freq), s.normInverse(p.Norm))
			if s.K1 {
				score = float32(score * k1)
			}
			if s.IDFTwice {
				// The weight holds the boost once; Bleve's queryWeight is
				// boost · idf · queryNorm and its term score holds idf alone.
				score = float32(score * float32(float64(c.idf)*queryNorm))
			}
			a := sums[string(p.ID)]
			sums[string(p.ID)] = sum{a.score + float64(score), a.tokens + c.tokens}
		}
	}
	out := make(map[string]float32, len(sums))
	for id, a := range sums {
		if s.Coord {
			a.score *= float64(a.tokens) / float64(tokens)
		}
		// Lucene sums the clause scores in double and returns a float.
		out[id] = float32(a.score)
	}
	return out, nil
}

// numFreeValues is SmallFloat.NUM_FREE_VALUES: lengths below it are stored
// exactly.
const numFreeValues = 24

// intToByte4 is SmallFloat.intToByte4, the one-byte, lossy encoding of a
// document length that Lucene stores as the norm.
func intToByte4(i int) uint8 {
	if i < numFreeValues {
		return uint8(i)
	}
	return uint8(numFreeValues + longToInt4(int64(i-numFreeValues)))
}

// longToInt4 keeps the four most significant bits of i and a shift.
func longToInt4(i int64) int {
	n := bits.Len64(uint64(i))
	if n < 4 {
		return int(i)
	}
	shift := n - 4
	return int(i>>shift)&0x07 | (shift+1)<<3
}

// byte4ToInt is SmallFloat.byte4ToInt, the length a norm byte decodes to.
func byte4ToInt(e uint8) int {
	i := int(e)
	if i < numFreeValues {
		return i
	}
	return int(min(numFreeValues+int4ToLong(i-numFreeValues), math.MaxInt32))
}

func int4ToLong(i int) int64 {
	m, shift := int64(i&0x07), i>>3-1
	if shift == -1 {
		return m
	}
	return (m | 0x08) << shift
}

// idf is BM25Similarity.idf.
func idf(docFreq, docCount int64) float32 {
	return float32(math.Log(1 + (float64(docCount-docFreq)+0.5)/(float64(docFreq)+0.5)))
}

// normCache is the table BM25Similarity.scorer builds: normInverse of the
// length each norm byte decodes to.
func normCache(avgdl float32) [256]float32 {
	var c [256]float32
	for i := range c {
		c[i] = normInverse(float32(byte4ToInt(uint8(i))), avgdl)
	}
	return c
}

// normInverse is 1 / (k1 · ((1 − b) + b · length / avgdl)). The explicit
// float32 conversion keeps the compiler from fusing a multiply and an add,
// which would skip a rounding Java performs.
func normInverse(length, avgdl float32) float32 {
	return 1 / (k1 * ((1 - b) + float32(b*length)/avgdl))
}

// termScore is BM25Scorer.score: weight is boost · idf. It equals
// weight · freq / (freq + k1 · (1 − b + b · length / avgdl)).
func termScore(weight, freq, normInverse float32) float32 {
	return weight - weight/(1+float32(freq*normInverse))
}

// adjustTies is Anserini's ScoreTiesAdjusterReranker, applied to scores in
// rank order: round each to four decimals, then lower a score that is not
// more than 1e-4 below its predecessor by 1e-6 per tie so far, so that
// trec_eval keeps the rank order instead of reordering ties by document ID.
func adjustTies(s []float32) {
	dup := 0
	for i := range s {
		s[i] = float32(math.Round(float64(s[i])*1e4) / 1e4)
		if i == 0 || s[i-1]-s[i] > 1e-4 {
			dup = 0
		} else {
			dup++
			s[i] -= float32(1e-6 * float32(dup))
		}
	}
}
