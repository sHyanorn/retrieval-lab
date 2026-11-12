package dense

import (
	"cmp"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
	"github.com/sHyanorn/retrieval-lab/internal/hnsw"
)

// Index holds document vectors for exact search, row i for IDs[i], all
// of one dimension.
type Index struct {
	IDs  []string
	Vecs []float32
}

// score is Lucene's DOT_PRODUCT similarity, as Anserini's flat dense search
// uses it: max((1 + q·d) / 2, 0) in float32, the dot product summed in
// float64 and rounded.
func score(q, d []float32) float32 {
	var s float64
	for j, v := range d {
		s += float64(q[j]) * float64(v)
	}
	return max((1+float32(s))/2, 0)
}

// byScore orders hits by score, then document ID ascending, as Anserini's
// FlatDenseSearcher sorts them.
func byScore(a, b eval.Hit) int {
	return cmp.Or(cmp.Compare(b.Score, a.Score), strings.Compare(a.DocID, b.DocID))
}

// hit scores document i for q; the rows are as wide as the vectors are
// long per ID.
func (x *Index) hit(q []float32, i int) eval.Hit {
	d := len(x.Vecs) / len(x.IDs)
	return eval.Hit{DocID: x.IDs[i], Score: float64(score(q, x.Vecs[i*d:(i+1)*d]))}
}

// Search scores every document and returns the top k, with the scores as
// they are: Anserini's flat dense search does not adjust ties (D011).
func (x *Index) Search(q []float32, k int) []eval.Hit {
	hits := make([]eval.Hit, len(x.IDs))
	for i := range hits {
		hits[i] = x.hit(q, i)
	}
	slices.SortFunc(hits, byScore)
	return slices.Clone(hits[:min(k, len(hits))])
}

// GraphConfig is recserve's default graph (internal/shard/set.go at
// c802a68); one inserting goroutine builds the same graph every time.
var GraphConfig = hnsw.Config{M: 16, EfConstruction: 200, Seed: 1, Workers: 1}

// EFs are the ef values of the approximate runs, the default list of
// recserve's cmd/annbench at c802a68.
var EFs = []int{10, 20, 40, 80, 120, 200, 400, 800}

// GraphDepth is the number of hits of the approximate run at ef: 10 below
// ef 100, else 100, as the graph searches layer 0 with max(ef, k).
func GraphDepth(ef int) int {
	if ef < 100 {
		return 10
	}
	return 100
}

// SearchGraph returns the k documents that the HNSW graph g, built over
// x.Vecs, finds for q at ef, scored and ordered as Search scores and orders
// them (D014). The document whose ID is skip is not returned.
func (x *Index) SearchGraph(g *hnsw.Index, q []float32, k, ef int, skip string) []eval.Hit {
	found := g.Search(q, k, ef, func(i int32) bool { return x.IDs[i] == skip })
	hits := make([]eval.Hit, len(found))
	for r, h := range found {
		hits[r] = x.hit(q, int(h.ID))
	}
	slices.SortFunc(hits, byScore)
	return hits
}
