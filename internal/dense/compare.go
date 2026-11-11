package dense

import (
	"fmt"
	"math"
	"slices"
)

// Agreement summarises the cosine similarity between two embeddings of
// the same documents. Quantiles are the cosine at rank ⌊q·n⌋ of the n
// cosines sorted in ascending order.
type Agreement struct {
	Docs                  int
	Min, P001, P01, P50   float64 // minimum and the 0.1 %, 1 % and 50 % quantiles
	Below9999, Below99999 int     // documents with a cosine below 0.9999 and 0.99999
}

// Compare aligns two sets of vectors of dimension dim by document ID and
// summarises their cosines; both must hold the same documents.
func Compare(aIDs []string, a []float32, bIDs []string, b []float32, dim int) (Agreement, error) {
	if len(aIDs) != len(bIDs) {
		return Agreement{}, fmt.Errorf("%d documents against %d", len(aIDs), len(bIDs))
	}
	row := make(map[string]int, len(bIDs))
	for i, id := range bIDs {
		row[id] = i
	}
	cos := make([]float64, len(aIDs))
	for i, id := range aIDs {
		j, ok := row[id]
		if !ok {
			return Agreement{}, fmt.Errorf("document %s is missing from the second set", id)
		}
		cos[i] = cosine(a[i*dim:(i+1)*dim], b[j*dim:(j+1)*dim])
	}
	slices.Sort(cos)
	at := func(q float64) float64 { return cos[min(int(q*float64(len(cos))), len(cos)-1)] }
	g := Agreement{Docs: len(cos), Min: cos[0], P001: at(0.001), P01: at(0.01), P50: at(0.5)}
	for _, c := range cos {
		if c < 0.9999 {
			g.Below9999++
		}
		if c < 0.99999 {
			g.Below99999++
		}
	}
	return g, nil
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}
