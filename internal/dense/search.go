package dense

import (
	"cmp"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// Index holds document vectors for exact search, row i for IDs[i].
type Index struct {
	IDs  []string
	Vecs []float32
}

// Search scores every document as Anserini's flat dense search does, with
// Lucene's DOT_PRODUCT similarity: max((1 + q·d) / 2, 0) in float32, the
// dot product summed in float64 and rounded. It returns the top k by score,
// then document ID ascending, with the scores as they are: that path does
// not adjust ties (D011).
func (x *Index) Search(q []float32, k int) []eval.Hit {
	n := len(x.IDs)
	scores := make([]float32, n)
	for i := range n {
		var d float64
		for j, v := range x.Vecs[i*Dim : (i+1)*Dim] {
			d += float64(q[j]) * float64(v)
		}
		scores[i] = max((1+float32(d))/2, 0)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		return cmp.Or(cmp.Compare(scores[b], scores[a]), strings.Compare(x.IDs[a], x.IDs[b]))
	})
	hits := make([]eval.Hit, min(k, n))
	for r := range hits {
		hits[r] = eval.Hit{DocID: x.IDs[order[r]], Score: float64(scores[order[r]])}
	}
	return hits
}
