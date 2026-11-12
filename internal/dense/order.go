package dense

import (
	"math/rand/v2"

	"github.com/sHyanorn/retrieval-lab/internal/hnsw"
)

// OrderSeed is the first word of the PCG that draws a build's insertion
// order; the build seed is the second (docs/PREREGISTRATION.md, Section 6).
const OrderSeed = 20261111

// InsertionOrder returns the order in which the build of the given seed
// inserts n documents: seed 0 inserts them in corpus order; from seed 1 the
// k-th document inserted is document p[k] of the corpus order, with
// p = rand.New(rand.NewPCG(OrderSeed, seed)).Perm(n).
func InsertionOrder(n int, seed uint64) []int {
	if seed == 0 {
		p := make([]int, n)
		for i := range p {
			p[i] = i
		}
		return p
	}
	return rand.New(rand.NewPCG(OrderSeed, seed)).Perm(n)
}

// Graph is the configuration of the build of the given seed with m links
// and efConstruction efc: level seed seed + 1, on one goroutine, so that
// seed 0 is GraphConfig at M 16 and efConstruction 200.
func Graph(m, efc int, seed uint64) hnsw.Config {
	return hnsw.Config{M: m, EfConstruction: efc, Seed: seed + 1, Workers: 1}
}

// Arrange moves x's rows, IDs and vectors together and in place, into the
// insertion order of the given seed, so that row k holds document p[k] of
// the corpus order. It starts from the corpus order, wherever an earlier
// call left the rows. Exact search does not depend on the order of the
// rows; a graph built over x.Vecs then has node k for document x.IDs[k].
func (x *Index) Arrange(seed uint64) {
	x.arrange(seed, false)
}

// arrange is Arrange; inverse puts row p[k] where row k should go, a
// broken variant for the tests.
func (x *Index) arrange(seed uint64, inverse bool) {
	n := len(x.IDs)
	if x.order != nil {
		back := make([]int, n) // row back[i] holds corpus document i
		for k, i := range x.order {
			back[i] = k
		}
		x.gather(back)
		x.order = nil
	}
	if seed == 0 {
		return
	}
	p := InsertionOrder(n, seed)
	if inverse {
		q := make([]int, n)
		for k, i := range p {
			q[i] = k
		}
		p = q
	}
	x.gather(p)
	x.order = p
}

// gather sets row k to the row that was at src[k], following the cycles
// of the permutation src with one row of scratch.
func (x *Index) gather(src []int) {
	n := len(x.IDs)
	if n == 0 {
		return
	}
	d := len(x.Vecs) / n
	row := func(i int) []float32 { return x.Vecs[i*d : (i+1)*d] }
	tmp := make([]float32, d)
	done := make([]bool, n)
	for start := range n {
		if done[start] {
			continue
		}
		id := x.IDs[start]
		copy(tmp, row(start))
		k := start
		for {
			done[k] = true
			s := src[k]
			if s == start {
				x.IDs[k] = id
				copy(row(k), tmp)
				break
			}
			x.IDs[k] = x.IDs[s]
			copy(row(k), row(s))
			k = s
		}
	}
}

// BuildGraph arranges x in the insertion order of the given seed and
// builds its graph with m links and efConstruction efc.
func (x *Index) BuildGraph(m, efc int, seed uint64) (*hnsw.Index, error) {
	x.Arrange(seed)
	return hnsw.Build(x.Vecs, len(x.Vecs)/max(1, len(x.IDs)), Graph(m, efc, seed))
}
