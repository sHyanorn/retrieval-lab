// Package bm25 indexes BEIR documents with Bleve and ranks them with
// Lucene's BM25 as Pyserini runs it for its BEIR "BM25 flat" condition:
// Bleve does the analysis and holds the postings; the scoring is Lucene's,
// computed here from those postings (D005).
package bm25

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/analysis"
	"github.com/blevesearch/bleve/v2/analysis/analyzer/custom"
	"github.com/blevesearch/bleve/v2/analysis/lang/en"
	"github.com/blevesearch/bleve/v2/analysis/token/lowercase"
	"github.com/blevesearch/bleve/v2/analysis/token/porter"
	"github.com/blevesearch/bleve/v2/analysis/token/stop"
	"github.com/blevesearch/bleve/v2/analysis/tokenizer/unicode"
	"github.com/blevesearch/bleve/v2/analysis/tokenmap"
	"github.com/blevesearch/bleve/v2/index/scorch"
	"github.com/blevesearch/bleve/v2/index/scorch/mergeplan"
	"github.com/blevesearch/bleve/v2/mapping"
	index "github.com/blevesearch/bleve_index_api"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

const (
	field     = "contents"
	analyzer  = "lucene_english"
	statsKey  = "bm25_stats"
	batchSize = 1000
)

// stopWords is Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET.
var stopWords = []any{
	"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
	"no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
	"they", "this", "to", "was", "will", "with",
}

// newMapping indexes one field, contents, with the chain of Anserini's
// DefaultEnglishAnalyzer: a UAX#29 tokenizer, the English possessive filter,
// lowercasing, Lucene's English stop words and the Porter stemmer.
func newMapping() (*mapping.IndexMappingImpl, error) {
	m := bleve.NewIndexMapping()
	if err := m.AddCustomTokenMap("lucene_english_stop", map[string]any{
		"type": tokenmap.Name, "tokens": stopWords,
	}); err != nil {
		return nil, err
	}
	if err := m.AddCustomTokenFilter("lucene_english_stop", map[string]any{
		"type": stop.Name, "stop_token_map": "lucene_english_stop",
	}); err != nil {
		return nil, err
	}
	if err := m.AddCustomAnalyzer(analyzer, map[string]any{
		"type":          custom.Name,
		"tokenizer":     unicode.Name,
		"token_filters": []any{en.PossessiveName, lowercase.Name, "lucene_english_stop", porter.Name},
	}); err != nil {
		return nil, err
	}
	f := bleve.NewTextFieldMapping()
	f.Analyzer = analyzer
	f.Store = false
	f.IncludeInAll = false
	f.IncludeTermVectors = false
	f.DocValues = false
	doc := bleve.NewDocumentStaticMapping()
	doc.AddFieldMappingsAt(field, f)
	m.DefaultMapping = doc
	m.DefaultAnalyzer = analyzer
	// Only the reference ranking of cmd/bm25 -bleve-scoring reads this.
	m.ScoringModel = index.BM25Scoring
	return m, nil
}

// Stats are the collection statistics Lucene's BM25 needs, counted at index
// time from the analyzer's output.
type Stats struct {
	Docs     int64 `json:"docs"`      // documents indexed
	NonEmpty int64 `json:"non_empty"` // documents with at least one token: Lucene's docCount
	Tokens   int64 `json:"tokens"`    // tokens over all documents: Lucene's sumTotalTermFreq
}

// Index is a Bleve index with the statistics and tables for scoring.
type Index struct {
	idx   bleve.Index
	r     index.IndexReader
	an    analysis.Analyzer
	stats Stats
	norms [256]float32
}

// Contents is the text BM25 flat indexes for a document: Anserini's
// BeirFlatCollection joins title and text with a newline.
func Contents(d beir.Doc) string { return d.Title + "\n" + d.Text }

// Build indexes every document corpus yields into a new index at path.
func Build(path string, corpus func(func(beir.Doc) error) error) (*Index, error) {
	m, err := newMapping()
	if err != nil {
		return nil, err
	}
	idx, err := bleve.New(path, m)
	if err != nil {
		return nil, err
	}
	an := m.AnalyzerNamed(analyzer)
	var st Stats
	batch := idx.NewBatch()
	err = corpus(func(d beir.Doc) error {
		text := Contents(d)
		n := int64(len(an.Analyze([]byte(text))))
		st.Docs++
		st.Tokens += n
		if n > 0 {
			st.NonEmpty++
		}
		if err := batch.Index(d.ID, map[string]any{field: text}); err != nil {
			return err
		}
		if batch.Size() < batchSize {
			return nil
		}
		err := idx.Batch(batch)
		batch.Reset()
		return err
	})
	if err == nil {
		err = idx.Batch(batch)
	}
	if err == nil {
		var buf []byte
		if buf, err = json.Marshal(st); err == nil {
			err = idx.SetInternal([]byte(statsKey), buf)
		}
	}
	if err != nil {
		idx.Close() //nolint:errcheck // the build error is the one to report
		return nil, err
	}
	// Closing persists every batch, so the merge sees all of them.
	if err := idx.Close(); err != nil {
		return nil, err
	}
	if idx, err = bleve.Open(path); err != nil {
		return nil, err
	}
	if err := singleSegment(idx); err != nil {
		idx.Close() //nolint:errcheck // the merge error is the one to report
		return nil, err
	}
	return open(idx)
}

// singleSegment merges the index into one segment. Bleve's own BM25
// statistics depend on the segments (D008), so this makes the reference
// ranking independent of how the batches were cut; the Lucene scoring does
// not depend on segments.
func singleSegment(idx bleve.Index) error {
	adv, err := idx.Advanced()
	if err != nil {
		return err
	}
	s, ok := adv.(*scorch.Scorch)
	if !ok {
		return fmt.Errorf("index is a %T, want scorch", adv)
	}
	return s.ForceMerge(context.Background(), &mergeplan.SingleSegmentMergePlanOptions)
}

// Open opens an index made by Build.
func Open(path string) (*Index, error) {
	idx, err := bleve.Open(path)
	if err != nil {
		return nil, err
	}
	return open(idx)
}

func open(idx bleve.Index) (*Index, error) {
	x := &Index{idx: idx, an: idx.Mapping().AnalyzerNamed(analyzer)}
	err := func() error {
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
	x.norms = normCache(avgdl)
	return x, nil
}

// Stats returns the collection statistics.
func (x *Index) Stats() Stats { return x.stats }

// Close releases the index.
func (x *Index) Close() error {
	err := x.r.Close()
	return cmp.Or(x.idx.Close(), err)
}

// terms analyzes a query as Anserini's BagOfWordsQueryGenerator does: one
// clause per distinct term, boosted by the number of times it occurs.
func (x *Index) terms(query string) map[string]int {
	counts := map[string]int{}
	for _, t := range x.an.Analyze([]byte(query)) {
		counts[string(t.Term)]++
	}
	return counts
}

// Search returns the top k documents for query as Pyserini ranks and scores
// them: Lucene's BM25 summed over the query's terms; ordered by score, then
// document ID ascending; scores then passed through adjustTies.
func (x *Index) Search(query string, k int) ([]eval.Hit, error) {
	counts := x.terms(query)
	sums := map[string]float64{}
	for _, t := range slices.Sorted(maps.Keys(counts)) {
		if err := x.addTerm(sums, t, float32(counts[t])); err != nil {
			return nil, err
		}
	}
	type cand struct {
		id    string
		score float32
	}
	cs := make([]cand, 0, len(sums))
	for id, s := range sums {
		// Lucene sums the clause scores in double and returns a float.
		cs = append(cs, cand{id, float32(s)})
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

// addTerm adds one clause's BM25 score to every document holding term,
// keyed by Bleve's internal document ID.
func (x *Index) addTerm(sums map[string]float64, term string, boost float32) error {
	tfr, err := x.r.TermFieldReader(context.Background(), []byte(term), field, true, true, false)
	if err != nil {
		return err
	}
	defer tfr.Close() //nolint:errcheck // read-only
	df := int64(tfr.Count())
	if df == 0 {
		return nil
	}
	weight := float32(boost * idf(df, x.stats.NonEmpty))
	var d index.TermFieldDoc
	for {
		p, err := tfr.Next(&d)
		if err != nil {
			return err
		}
		if p == nil {
			return nil
		}
		// Bleve stores the exact length; its posting returns 1/sqrt(length).
		length := int(math.Round(1 / (p.Norm * p.Norm)))
		sums[string(p.ID)] += float64(termScore(weight, float32(p.Freq), x.norms[intToByte4(length)]))
	}
}

// SearchBleve ranks with Bleve's own BM25 instead of Lucene's, for the
// reference row of the alignment table (D008): a match query on the
// contents field, ordered by score and then document ID, scores passed
// through adjustTies. Bleve reads k1 and b from the package variables
// search.BM25_k1 and search.BM25_b.
func (x *Index) SearchBleve(query string, k int) ([]eval.Hit, error) {
	q := bleve.NewMatchQuery(query)
	q.SetField(field)
	req := bleve.NewSearchRequestOptions(q, k, 0, false)
	req.SortBy([]string{"-_score", "_id"})
	res, err := x.idx.Search(req)
	if err != nil {
		return nil, err
	}
	scores := make([]float32, len(res.Hits))
	for i, h := range res.Hits {
		scores[i] = float32(h.Score)
	}
	adjustTies(scores)
	hits := make([]eval.Hit, len(res.Hits))
	for i, h := range res.Hits {
		hits[i] = eval.Hit{DocID: h.ID, Score: float64(scores[i])}
	}
	return hits, nil
}
