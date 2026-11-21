// Package bm25 indexes BEIR documents with Bleve and ranks them with
// Lucene's BM25 as Pyserini runs it for its BEIR "BM25 flat" condition:
// Bleve does the analysis and holds the postings; the scoring is Lucene's,
// computed here from those postings (D005).
package bm25

import (
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"

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
	boundsKey = "bm25_bounds"
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
	idx    bleve.Index
	r      index.IndexReader
	an     analysis.Analyzer
	stats  Stats
	norms  [256]float32
	bounds map[string]float32 // per term, the highest product of its postings (D023)
	accs   sync.Pool          // *accumulator
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
	if err := storeBounds(idx, st); err != nil {
		idx.Close() //nolint:errcheck // the bounds error is the one to report
		return nil, err
	}
	return open(idx)
}

// avgdl is BM25Similarity.avgFieldLength.
func (s Stats) avgdl() float32 { return float32(float64(s.Tokens) / float64(s.NonEmpty)) }

// storeBounds reads every posting once and stores, per term, the highest
// freq · normInverse of its postings, rounded to float32: with a clause's
// weight, the highest score the term adds to a document (scoreOf).
func storeBounds(idx bleve.Index, st Stats) error {
	adv, err := idx.Advanced()
	if err != nil {
		return err
	}
	r, err := adv.Reader()
	if err != nil {
		return err
	}
	defer r.Close() //nolint:errcheck // read-only
	x := &Index{r: r, norms: normCache(st.avgdl())}
	dict, err := r.FieldDict(field)
	if err != nil {
		return err
	}
	defer dict.Close() //nolint:errcheck // read-only
	var buf []byte
	var d index.TermFieldDoc
	for {
		e, err := dict.Next()
		if err != nil {
			return err
		}
		if e == nil {
			return idx.SetInternal([]byte(boundsKey), buf)
		}
		tfr, err := r.TermFieldReader(context.Background(), []byte(e.Term), field, true, true, false)
		if err != nil {
			return err
		}
		var most float32
		for {
			p, err := tfr.Next(&d)
			if err != nil {
				tfr.Close() //nolint:errcheck // read-only
				return err
			}
			if p == nil {
				break
			}
			most = max(most, x.product(p))
		}
		if err := tfr.Close(); err != nil {
			return err
		}
		buf = binary.AppendUvarint(buf, uint64(len(e.Term)))
		buf = append(buf, e.Term...)
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(most))
	}
}

// readBounds decodes what storeBounds stored.
func readBounds(buf []byte) (map[string]float32, error) {
	bounds := map[string]float32{}
	for len(buf) > 0 {
		n, w := binary.Uvarint(buf)
		if w <= 0 || uint64(len(buf)-w) < n+4 {
			return nil, fmt.Errorf("term bounds: entry %d is cut short", len(bounds))
		}
		term := string(buf[w : w+int(n)])
		buf = buf[w+int(n):]
		bounds[term] = math.Float32frombits(binary.LittleEndian.Uint32(buf))
		buf = buf[4:]
	}
	return bounds, nil
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
		buf, err = idx.GetInternal([]byte(boundsKey))
		if err != nil {
			return err
		}
		if buf == nil {
			return fmt.Errorf("the index has no term bounds; build it again")
		}
		if x.bounds, err = readBounds(buf); err != nil {
			return err
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
	x.norms = normCache(x.stats.avgdl())
	x.accs.New = func() any {
		return &accumulator{sum: make([]float64, x.stats.Docs), seen: make([]bool, x.stats.Docs)}
	}
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
// document ID ascending; scores then passed through adjustTies. It
// evaluates the query with MaxScore (D023).
func (x *Index) Search(query string, k int) ([]eval.Hit, error) {
	return x.maxScore(query, k, mutation{})
}

// SearchExhaustive returns what Search returns, reading every posting of
// every query term into an array by document number.
func (x *Index) SearchExhaustive(query string, k int) ([]eval.Hit, error) {
	return x.exhaustive(query, k, mutation{})
}

// mutation breaks the evaluation on purpose, for tests that must catch it.
type mutation struct {
	strictCut bool // drop the documents tied with the k-th score
	shortHeap bool // take the cut from a heap of k − 1 scores
	byDocNum  bool // order tied scores by Bleve's document number, not by ID
	stale     bool // leave the sums of the last query in the accumulator
	lowBound  bool // halve MaxScore's bounds
	pruneTies bool // let MaxScore skip a document whose bound equals the cut
}

// cand is a document that may be in the top k.
type cand struct {
	doc   uint64
	id    string
	score float32
}

func (x *Index) exhaustive(query string, k int, mut mutation) ([]eval.Hit, error) {
	if k < 1 {
		return nil, fmt.Errorf("k = %d, want at least 1", k)
	}
	acc := x.accs.Get().(*accumulator)
	defer x.release(acc, mut)
	counts := x.terms(query)
	for _, t := range slices.Sorted(maps.Keys(counts)) {
		if err := x.addTerm(acc, t, float32(counts[t])); err != nil {
			return nil, err
		}
	}
	// Lucene sums the clause scores in double and returns a float.
	top := newTopK(k, mut)
	for _, d := range acc.touched {
		top.push(float32(acc.sum[d]))
	}
	cut := top.cut()
	var cs []cand
	for _, d := range acc.touched {
		if s := float32(acc.sum[d]); s >= cut {
			cs = append(cs, cand{doc: d, score: s})
		}
	}
	return x.rank(cs, cut, k, mut)
}

// rank orders the candidates scoring at least cut by score and then ID,
// keeps the first k and adjusts their ties.
func (x *Index) rank(cs []cand, cut float32, k int, mut mutation) ([]eval.Hit, error) {
	var kept []cand
	for _, c := range cs {
		if c.score < cut || mut.strictCut && c.score == cut {
			continue
		}
		id, err := x.r.ExternalID(index.NewIndexInternalID(nil, c.doc))
		if err != nil {
			return nil, err
		}
		c.id = id
		kept = append(kept, c)
	}
	slices.SortFunc(kept, func(a, b cand) int {
		if mut.byDocNum {
			return cmp.Or(cmp.Compare(b.score, a.score), cmp.Compare(a.doc, b.doc))
		}
		return cmp.Or(cmp.Compare(b.score, a.score), strings.Compare(a.id, b.id))
	})
	kept = kept[:min(k, len(kept))]
	scores := make([]float32, len(kept))
	for i, c := range kept {
		scores[i] = c.score
	}
	adjustTies(scores)
	hits := make([]eval.Hit, len(kept))
	for i, c := range kept {
		hits[i] = eval.Hit{DocID: c.id, Score: float64(scores[i])}
	}
	return hits, nil
}

// topK is a min-heap of the k highest scores pushed, equal scores counted
// apart.
type topK struct {
	k int
	h []float32
}

func newTopK(k int, mut mutation) *topK {
	if mut.shortHeap {
		k = max(k-1, 1)
	}
	return &topK{k: k, h: make([]float32, 0, k)}
}

func (t *topK) push(s float32) {
	h := t.h
	if len(h) < t.k {
		h = append(h, s)
		for i := len(h) - 1; i > 0 && h[(i-1)/2] > h[i]; i = (i - 1) / 2 {
			h[i], h[(i-1)/2] = h[(i-1)/2], h[i]
		}
		t.h = h
		return
	}
	if s <= h[0] {
		return
	}
	h[0] = s
	for i := 0; ; {
		c := 2*i + 1
		if c >= len(h) {
			return
		}
		if c+1 < len(h) && h[c+1] < h[c] {
			c++
		}
		if h[i] <= h[c] {
			return
		}
		h[i], h[c] = h[c], h[i]
		i = c
	}
}

// cut returns the k-th highest score pushed, or −∞ while fewer than k are
// in: a document scoring at least the cut may be in the top k, ties at rank
// k included.
func (t *topK) cut() float32 {
	if len(t.h) < t.k {
		return float32(math.Inf(-1))
	}
	return t.h[0]
}

// accumulator holds a query's sums of clause scores by Bleve's document
// number.
type accumulator struct {
	sum     []float64
	seen    []bool
	touched []uint64 // the documents seen, in the order first matched
}

func (x *Index) release(acc *accumulator, mut mutation) {
	if !mut.stale {
		for _, d := range acc.touched {
			acc.sum[d], acc.seen[d] = 0, false
		}
	}
	acc.touched = acc.touched[:0]
	x.accs.Put(acc)
}

// addTerm adds one clause's BM25 score to every document holding term.
func (x *Index) addTerm(acc *accumulator, term string, boost float32) error {
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
		doc := p.ID.Value()
		if doc >= uint64(len(acc.sum)) {
			return fmt.Errorf("document number %d, want below the %d documents indexed", doc, len(acc.sum))
		}
		if !acc.seen[doc] {
			acc.seen[doc] = true
			acc.touched = append(acc.touched, doc)
		}
		acc.sum[doc] += float64(scoreOf(weight, x.product(p)))
	}
}

// product is freq · normInverse of a posting, rounded to float32.
func (x *Index) product(p *index.TermFieldDoc) float32 {
	// Bleve stores the exact length; its posting returns 1/sqrt(length).
	length := int(math.Round(1 / (p.Norm * p.Norm)))
	return float32(float32(p.Freq) * x.norms[intToByte4(length)])
}

// clause is one query term's postings in MaxScore's evaluation.
type clause struct {
	tfr    index.TermFieldReader
	pos    int     // the term's place in the query's sorted terms, the order of summation
	weight float32 // boost · idf
	bound  float32 // the highest score the term adds to a document
	doc    uint64  // the current posting's document number
	done   bool    // no posting left
	buf    index.TermFieldDoc
	p      *index.TermFieldDoc
}

func (c *clause) next() error { return c.at(c.tfr.Next(&c.buf)) }

// advance moves to the first posting at or after doc, which must lie past
// the current one: Bleve reopens the reader to go back or stay.
func (c *clause) advance(doc uint64) error {
	return c.at(c.tfr.Advance(index.NewIndexInternalID(nil, doc), &c.buf))
}

func (c *clause) at(p *index.TermFieldDoc, err error) error {
	if err != nil {
		return err
	}
	if p == nil {
		c.done = true
		return nil
	}
	c.p, c.doc = p, p.ID.Value()
	return nil
}

// maxScore evaluates a query with MaxScore (Turtle and Flood, 1995): the
// clauses whose bounds sum below the cut are not essential, so only the
// postings of the others name candidates, and a candidate is scored on a
// non-essential clause only while its bound can still reach the cut. A
// document's score and every bound on it sum the same clauses in the same
// order, the query's sorted terms, with a bound in place of each clause not
// yet read; float rounding keeps that order, so no bound falls below the
// score it bounds.
func (x *Index) maxScore(query string, k int, mut mutation) ([]eval.Hit, error) {
	if k < 1 {
		return nil, fmt.Errorf("k = %d, want at least 1", k)
	}
	counts := x.terms(query)
	terms := slices.Sorted(maps.Keys(counts))
	var cls []*clause
	defer func() {
		for _, c := range cls {
			c.tfr.Close() //nolint:errcheck // read-only
		}
	}()
	for pos, t := range terms {
		tfr, err := x.r.TermFieldReader(context.Background(), []byte(t), field, true, true, false)
		if err != nil {
			return nil, err
		}
		df := int64(tfr.Count())
		if df == 0 {
			tfr.Close() //nolint:errcheck // read-only
			continue
		}
		b, ok := x.bounds[t]
		if !ok {
			tfr.Close() //nolint:errcheck // read-only
			return nil, fmt.Errorf("term %q has postings but no bound", t)
		}
		c := &clause{tfr: tfr, pos: pos, weight: float32(float32(counts[t]) * idf(df, x.stats.NonEmpty))}
		c.bound = scoreOf(c.weight, b)
		if mut.lowBound {
			c.bound /= 2
		}
		cls = append(cls, c)
		if err := c.next(); err != nil {
			return nil, err
		}
	}
	slices.SortStableFunc(cls, func(a, b *clause) int { return cmp.Compare(a.bound, b.bound) })

	vals := make([]float32, len(terms)) // a clause's score or bound, by place in terms
	sum := func() float32 {
		var s float64
		for _, v := range vals {
			s += float64(v)
		}
		return float32(s)
	}
	// optimistic is the sum with the bounds of cls[:n] in place of their scores.
	optimistic := func(n int) float32 {
		for _, c := range cls[:n] {
			vals[c.pos] = c.bound
		}
		s := sum()
		for _, c := range cls[:n] {
			vals[c.pos] = 0
		}
		return s
	}
	top := newTopK(k, mut)
	cut := top.cut()
	ne := 0 // cls[:ne] are not essential
	var cs []cand
docs:
	for {
		doc, found := uint64(0), false
		for _, c := range cls[ne:] {
			if !c.done && (!found || c.doc < doc) {
				doc, found = c.doc, true
			}
		}
		if !found {
			break
		}
		clear(vals)
		for _, c := range cls[ne:] {
			if !c.done && c.doc == doc {
				vals[c.pos] = scoreOf(c.weight, x.product(c.p))
				if err := c.next(); err != nil {
					return nil, err
				}
			}
		}
		for i := ne - 1; i >= 0; i-- {
			if b := optimistic(i + 1); b < cut || mut.pruneTies && b == cut {
				continue docs
			}
			c := cls[i]
			if !c.done && c.doc < doc {
				if err := c.advance(doc); err != nil {
					return nil, err
				}
			}
			if !c.done && c.doc == doc {
				vals[c.pos] = scoreOf(c.weight, x.product(c.p))
			}
		}
		s := sum()
		if s < cut {
			continue
		}
		cs = append(cs, cand{doc: doc, score: s})
		top.push(s)
		cut = top.cut()
		clear(vals)
		for ne < len(cls) && optimistic(ne+1) < cut {
			ne++
		}
	}
	return x.rank(cs, top.cut(), k, mut)
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
